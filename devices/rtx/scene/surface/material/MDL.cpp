/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "MDL.h"

#include "fmt/base.h"
#include "gpu/gpu_objects.h"
#include "optix_visrtx.h"
#include "scene/surface/material/Material.h"

#include "libmdl/ArgumentBlockDescriptor.h"
#include "libmdl/helpers.h"
#include "scene/surface/material/sampler/Sampler.h"

#include <anari/frontend/anari_enums.h>

#include <mi/base/handle.h>
#include <mi/neuraylib/ivalue.h>

#include <fmt/core.h>

#include <algorithm>
#include <string>
namespace visrtx {

MDL::MDL(DeviceGlobalState *d) : Material(d)
{
  auto &mdl = d->mdl;
}

MDL::~MDL()
{
  for (auto sampler : m_samplers) {
    deviceState()->mdl->samplerRegistry.releaseSampler(sampler);
  }
}

void MDL::commit()
{
  Material::commit();
}

void MDL::syncSource()
{
  auto sourceType = getParamString("sourceType", "module");
  auto source = getParamString("source", "::visrtx::default::diffusePink");
  auto uuid = libmdl::Uuid{};

  // Handle source changes separately as it probably implies a full material
  // change.
  if (source != m_source || sourceType != m_sourceType) {
    // Equivalent using the material registry instead of the material manager
    if (sourceType == "module") {
      auto &&[moduleName, materialName] = libmdl::parseCmdArgumentMaterialName(
          source, &deviceState()->mdl->core);
      uuid = deviceState()->mdl->materialRegistry.acquireMaterial(
          moduleName, materialName);
    } else if (sourceType == "code") {
      uuid = {};
      reportMessage(ANARI_SEVERITY_ERROR,
          "MDL::commit(): sourceType 'code' not supported yet");
    } else {
      uuid = {};
      reportMessage(ANARI_SEVERITY_ERROR,
          "MDL::commit(): sourceType must be either 'module' or 'code'");
    }

    if (uuid != libmdl::Uuid{}) {
      // We have successfully loaded a material, release the previous one and
      // use it instead.
      deviceState()->mdl->materialRegistry.releaseMaterial(m_uuid);
      m_argumentBlockInstance =
          deviceState()->mdl->materialRegistry.createArgumentBlock(uuid);
      m_uuid = uuid;
    }

    m_source = source;
    m_sourceType = sourceType;
  }
}

void MDL::syncParameters()
{
  if (m_argumentBlockInstance.has_value()) {
    // Reset resource mapping
    m_argumentBlockInstance->resetResources();

    auto transaction = mi::base::make_handle(
        deviceState()->mdl->materialRegistry.createTransaction());
    auto factory = deviceState()->mdl->materialRegistry.getMdlFactory();

    auto &argumentBlockInstance = *m_argumentBlockInstance;
    for (auto &&[name, type] : argumentBlockInstance.enumerateArguments()) {
      auto sourceParamAny = getParamDirect(name);
      switch (type) {
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Bool: {
        if (sourceParamAny.type() == ANARI_BOOL) {
          argumentBlockInstance.setValue(
              name, sourceParamAny.get<bool>(), transaction.get(), factory);
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Float: {
        if (sourceParamAny.type() == ANARI_FLOAT32) {
          argumentBlockInstance.setValue(
              name, sourceParamAny.get<float>(), transaction.get(), factory);
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Int: {
        if (sourceParamAny.type() == ANARI_INT32) {
          argumentBlockInstance.setValue(
              name, sourceParamAny.get<int>(), transaction.get(), factory);
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Color: {
        if (sourceParamAny.type() == ANARI_FLOAT32_VEC3) {
          auto value = sourceParamAny.get<glm::vec3>();
          argumentBlockInstance.setColorValue(
              name, {value.r, value.g, value.b}, transaction.get(), factory);
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Texture: {
        if (sourceParamAny.type() == ANARI_STRING) {
          // FIXME: Deal with colorspace, possibly by reading a
          // `name.colorspace` attribute linearlizing on load here, or by
          // creating a matching sampler below.
          argumentBlockInstance.setTextureValue(name,
              sourceParamAny.getString().c_str(),
              transaction.get(),
              factory);
        }
        break;
      }
      default: {
        reportMessage(ANARI_SEVERITY_WARNING, "Don't know how to set '{}' (unsupported type)", name);
      }
      }
    }

    // Traverse resources to related samplers. Allocate actual ones and release
    // previous ones. Note that unchanged samplers, compared to the previous
    // commit will therby be reference both in samplers and newSamplers before
    // being dereference when cleaning actual samplers.
    std::vector<const Sampler *> newSamplers;
    for (const auto &name :
        m_argumentBlockInstance->getTextureResourceNames()) {
      if (name.empty()) {
        newSamplers.push_back({});
      } else {
        // FIXME:  How to deduce the shape from the type/resource/target code?
        // FIXME: Handle colorspace (for instance, normals should be raw, not
        // srgb) here or at load time (see above)
        auto textureShape =
            mi::neuraylib::ITarget_code::Texture_shape::Texture_shape_2d;

        newSamplers.push_back(
            deviceState()->mdl->samplerRegistry.acquireSampler(
                name, textureShape, transaction.get()));
      }
    }

    for (auto sampler : m_samplers) {
      deviceState()->mdl->samplerRegistry.releaseSampler(sampler);
    }
    m_samplers.assign(next(cbegin(newSamplers)), cend(newSamplers));

    // Pretty dumb, but we don't yet maintain a list of used resource, and those
    // are for now only used to populate anari samplers, so we should be OK
    // dropping everything that's in the current transaction. Note that this
    // will imply reloading texture in between different commits...
    transaction->commit();
  }
}

void MDL::syncImplementationIndex()
{
  m_implementationIndex =
      deviceState()->mdl->materialRegistry.getMaterialImplementationIndex(
          m_uuid);
}

MaterialGPUData MDL::gpuData() const
{
  MaterialGPUData retval;
  retval.materialType = MaterialType::MDL;
  retval.mdl.implementationIndex = m_implementationIndex;
  retval.mdl.numSamplers =
      std::min(std::size(retval.mdl.samplers), size(m_samplers));
  std::fill(std::begin(retval.mdl.samplers),
      std::end(retval.mdl.samplers),
      DeviceObjectIndex(~0));
  std::transform(cbegin(m_samplers),
      cend(m_samplers),
      std::begin(retval.mdl.samplers),
      [](const auto &v) { return v->index(); });

  if (const auto &argBlockData =
          m_argumentBlockInstance->getArgumentBlockData();
      !argBlockData.empty()) {
    m_argBlockBuffer.upload(data(argBlockData), size(argBlockData));
  } else {
    m_argBlockBuffer.reset();
  }
  retval.mdl.argBlock =
      m_argBlockBuffer.bytes() ? m_argBlockBuffer.ptrAs<const char>() : nullptr;

  return retval;
}

void MDL::markCommitted()
{
  Material::markCommitted();
  deviceState()->objectUpdates.lastMDLObjectChange = helium::newTimeStamp();
}

} // namespace visrtx
