/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/gpu_objects.h"
#include "libmdl/ArgumentBlockInstance.h"
#include "optix_visrtx.h"
#include "sampler/Sampler.h"
#include "scene/surface/material/Material.h"

#include "libmdl/ArgumentBlockDescriptor.h"
#include "libmdl/helpers.h"
#include "scene/surface/material/sampler/Sampler.h"

#include <anari/frontend/anari_enums.h>

#include <anari/frontend/type_utility.h>
#include <helium/utility/IntrusivePtr.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/ivalue.h>

#include <cstdio>
#include <iterator>
#include <nonstd/scope.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <set>
#include <string>
#include <string_view>

using namespace std::string_view_literals;

namespace visrtx {

MDL::MDL(DeviceGlobalState *d) : Material(d) {}

MDL::~MDL()
{
  clearSamplers();
}

void MDL::clearSamplers()
{
  auto &samplerRegistry = deviceState()->mdl->samplerRegistry;
  for (auto &samplerDesc : m_samplers) {
    if (samplerDesc.isFromRegistry) {
      samplerRegistry.releaseSampler(samplerDesc.sampler);
    } else {
      samplerDesc.sampler->refDec(helium::INTERNAL);
    }
  }

  m_samplers.clear();
}

void MDL::commitParameters()
{
  Material::commitParameters();
}

void MDL::finalize()
{
  syncSource();
  syncImplementationIndex();
  syncParameters();
  upload();
}

void MDL::syncSource()
{
  auto sourceType = getParamString("sourceType", "module");
  auto source = getParamString("source", "::visrtx::default::diffuseWhite");
  auto uuid = libmdl::Uuid{};
  auto argumentBlockDescriptor = libmdl::ArgumentBlockDescriptor{};

  auto &materialRegistry = deviceState()->mdl->materialRegistry;
  auto &samplerRegistry = deviceState()->mdl->samplerRegistry;

  // Handle source changes separately as it probably implies a full material
  // change.
  if (source != m_source || sourceType != m_sourceType) {
    // Equivalent using the material registry instead of the material manager
    if (sourceType == "module") {
      auto &&[moduleName, materialName] =
          libmdl::parseMaterialSourceName(source, &deviceState()->mdl->core);
      std::tie(uuid, argumentBlockDescriptor) =
          materialRegistry.acquireMaterial(moduleName, materialName);
      if (uuid == libmdl::Uuid{}) {
        reportMessage(ANARI_SEVERITY_ERROR,
            "Failed to acquire material %s, falling back to %s",
            source.c_str(),
            "diffuseWhite");
        std::tie(uuid, argumentBlockDescriptor) =
            materialRegistry.acquireMaterial(
                "::visrtx::default", "diffuseWhite");
      }
    } else if (sourceType == "code") {
      uuid = {};
      reportMessage(ANARI_SEVERITY_ERROR,
          "MDL::commitParameters(): sourceType 'code' not supported yet");
    } else {
      uuid = {};
      reportMessage(ANARI_SEVERITY_ERROR,
          "MDL::commitParameters(): sourceType must be either 'module' or 'code'");
    }

    if (uuid != libmdl::Uuid{}) {
      // We have successfully loaded a material, release the previous one and
      // use it instead.
      if (m_uuid != libmdl::Uuid{}) {
        materialRegistry.releaseMaterial(m_uuid);
      }
      m_argumentBlockInstance =
          materialRegistry.createArgumentBlock(argumentBlockDescriptor);
      m_uuid = uuid;

      clearSamplers();

      for (auto textureDesc :
          argumentBlockDescriptor.m_defaultAndBodyTextureDescriptors) {
        auto sampler = samplerRegistry.acquireSampler(textureDesc);
        auto index = textureDesc.knownIndex;
        if (m_samplers.size() <= index) {
          m_samplers.resize(index + 1);
        }
        m_samplers[textureDesc.knownIndex] = {sampler, textureDesc.url, true};
      }
    }

    m_source = source;
    m_sourceType = sourceType;
  }
}

void MDL::syncParameters()
{
  if (m_argumentBlockInstance.has_value()) {
    auto &argumentBlockInstance = *m_argumentBlockInstance;
    for (auto param = params_begin(); param != params_end(); ++param) {
      const auto &name = param->first;
      if (name == "source"sv || name == "sourceType"sv) {
        // Skip these two parameters, they are not part of the argument block
        continue;
      }

      if (argumentBlockInstance.hasArgument(name) == 0) {
        reportMessage(ANARI_SEVERITY_WARNING,
            "%s is not a valid parameter for an MDL material %s",
            name.c_str(),
            m_source.c_str());
      }
    }

    for (auto &&[name, type] : argumentBlockInstance.enumerateArguments()) {
      auto sourceParamAny = getParamDirect(name);

      switch (type) {
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Bool: {
        if (sourceParamAny.type() == ANARI_BOOL) {
          argumentBlockInstance.setValue(name, sourceParamAny.get<bool>());
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Float: {
        if (sourceParamAny.type() == ANARI_FLOAT32) {
          argumentBlockInstance.setValue(name, sourceParamAny.get<float>());
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Float2: {
        if (sourceParamAny.type() == ANARI_FLOAT32_VEC2) {
          auto value = sourceParamAny.get<glm::vec2>();
          argumentBlockInstance.setValue(name, {value.x, value.y});
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Float3: {
        if (sourceParamAny.type() == ANARI_FLOAT32_VEC3) {
          auto value = sourceParamAny.get<glm::vec3>();
          argumentBlockInstance.setValue(name, {value.x, value.y, value.z});
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Float4: {
        if (sourceParamAny.type() == ANARI_FLOAT32_VEC4) {
          auto value = sourceParamAny.get<glm::vec4>();
          argumentBlockInstance.setValue(
              name, {value.x, value.y, value.z, value.w});
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Int: {
        if (sourceParamAny.type() == ANARI_INT32) {
          argumentBlockInstance.setValue(name, sourceParamAny.get<int>());
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Int2: {
        if (sourceParamAny.type() == ANARI_INT32_VEC2) {
          auto value = sourceParamAny.get<glm::ivec2>();
          argumentBlockInstance.setValue(name, {value.x, value.y});
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Int3: {
        if (sourceParamAny.type() == ANARI_INT32_VEC3) {
          auto value = sourceParamAny.get<glm::ivec3>();
          argumentBlockInstance.setValue(name, {value.x, value.y, value.z});
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Int4: {
        if (sourceParamAny.type() == ANARI_INT32_VEC4) {
          auto value = sourceParamAny.get<glm::ivec4>();
          argumentBlockInstance.setValue(
              name, {value.x, value.y, value.z, value.w});
        }
        break;
      }

      case libmdl::ArgumentBlockDescriptor::ArgumentType::Color: {
        if (sourceParamAny.type() == ANARI_FLOAT32_VEC3) {
          auto value = sourceParamAny.get<glm::vec3>();
          argumentBlockInstance.setValue(name, {value.r, value.g, value.b});
        }
        break;
      }
      case libmdl::ArgumentBlockDescriptor::ArgumentType::Texture: {
        Sampler *sampler = nullptr;
        auto &samplerRegistry = deviceState()->mdl->samplerRegistry;

        if (!sourceParamAny.valid()) {
          if (auto it = find_if(begin(m_samplers),
                  end(m_samplers),
                  [name = name](auto &p) { return p.name == name; });
              it != end(m_samplers)) {
            // We have a sampler for this parameter, but it is not set.
            // Set it to 0, which means no sampler.
            if (it->isFromRegistry) {
              samplerRegistry.releaseSampler(it->sampler);
            } else {
              it->sampler->refDec(helium::INTERNAL);
            }
            *it = {};
          }

          // Return the sampler to an undefined state in the argument block.
          argumentBlockInstance.setValue(name, 0);
          continue;
        }

        bool samplerIsFromRegistry = false;

        switch (sourceParamAny.type()) {
        case ANARI_STRING: {
          auto colorspaceStr = getParamString(name + ".colorspace", "srgb");
          auto colorspace = libmdl::ColorSpace::sRGB;
          if (colorspaceStr != "raw" && colorspaceStr != "srgb") {
            reportMessage(ANARI_SEVERITY_WARNING,
                "Unknown colorspace type %s for %s. Falling back to srgb",
                colorspaceStr,
                name);
            colorspaceStr = "srgb"s;
          }
          if (colorspaceStr == "raw"sv) {
            colorspace = libmdl::ColorSpace::Linear;
          } else if (colorspaceStr == "srgb"sv) {
            colorspace = libmdl::ColorSpace::sRGB;
          }

          sampler = samplerRegistry.acquireSampler(
              sourceParamAny.getString(), colorspace);
          samplerIsFromRegistry = true;
          break;
        }
        case ANARI_SAMPLER: {
          sampler = sourceParamAny.getObject<Sampler>();
          // We need to hold a reference to the sampler
          if (sampler)
            sampler->refInc(helium::INTERNAL);
          break;
        }
        }

        if (sampler) {
          // Find a valid slot for out sampler.
          auto it =
              std::find(begin(m_samplers), end(m_samplers), SamplerDesc{});
          if (it == end(m_samplers)) {
            it = m_samplers.insert(it, {sampler, name, samplerIsFromRegistry});
          } else {
            *it = {sampler, name, samplerIsFromRegistry};
          }

          int index = distance(std::begin(m_samplers), it);
          argumentBlockInstance.setValue(
              name, index + 1); // MDL starts counting at 1, 0 being invalid.
        }
        break;
      }
      default: {
        reportMessage(ANARI_SEVERITY_WARNING,
            "Don't know how to set '%s' (unsupported type %i)",
            name.c_str(),
            type);
      }
      }
    }
  }
}

void MDL::syncImplementationIndex()
{
  m_implementationIndex = static_cast<unsigned int>(MaterialType::MDL)
      + deviceState()->mdl->materialRegistry.getMaterialImplementationIndex(
          m_uuid);
}

MaterialGPUData MDL::gpuData() const
{
  MaterialGPUData retval = {};

  retval.implementationIndex = m_implementationIndex;

  if (m_argumentBlockInstance.has_value()) {
    retval.materialData.mdl.numSamplers =
        std::min(std::size(retval.materialData.mdl.samplers), size(m_samplers));
    std::fill(std::begin(retval.materialData.mdl.samplers),
        std::end(retval.materialData.mdl.samplers),
        DeviceObjectIndex(~0));
    std::transform(cbegin(m_samplers),
        cend(m_samplers),
        std::begin(retval.materialData.mdl.samplers),
        [](const auto &v) {
          return v.sampler ? v.sampler->index() : DeviceObjectIndex(~0);
        });

    if (const auto &argBlockData =
            m_argumentBlockInstance->getArgumentBlockData();
        !argBlockData.empty()) {
      m_argBlockBuffer.upload(data(argBlockData), size(argBlockData));
    } else {
      m_argBlockBuffer.reset();
    }
    retval.materialData.mdl.argBlock = m_argBlockBuffer.bytes()
        ? m_argBlockBuffer.ptrAs<const char>()
        : nullptr;
  }

  return retval;
}

} // namespace visrtx
