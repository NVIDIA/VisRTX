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

#include "MaterialRegistry.h"

#include "ptx.h"

#include "libmdl/ArgumentBlockDescriptor.h"
#include "libmdl/ArgumentBlockInstance.h"
#include "libmdl/TimeStamp.h"
#include "libmdl/ptx.h"
#include "material/PhysicallyBasedMDL.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/types.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/version.h>

#include <fstream>
#include <glm/fwd.hpp>

#include <nonstd/scope.hpp>

#include <fmt/base.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std::string_literals;

extern "C" const char VISRTX_DEFAULT_MDL[];
extern "C" const char VISRTX_PHYSICALLY_BASED_MDL[];

namespace visrtx::mdl {

MaterialRegistry::MaterialRegistry(libmdl::Core *core)
    : m_core(core),
      m_scope(m_core->createScope("VisRTXMaterialResgistryScope"s
          + std::to_string(std::uintptr_t(this))))
{
  m_core->addBuiltinModule("::visrtx::default", VISRTX_DEFAULT_MDL);
  m_core->addBuiltinModule(
      "::visrtx::physically_based", VISRTX_PHYSICALLY_BASED_MDL);
}

MaterialRegistry::~MaterialRegistry()
{
  m_core->removeScope(m_scope.get());
}

std::tuple<libmdl::Uuid, libmdl::ArgumentBlockDescriptor>
MaterialRegistry::acquireMaterial(
    std::string_view moduleName, std::string_view materialName)
{
  using mi::base::make_handle;

  // First check if the material has already been compiled.
  auto fullMaterialName = fmt::format("{}::{}", moduleName, materialName);
  m_core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
      "Acquiring material {}",
      fullMaterialName);

  if (auto uuidIt = m_materialNameToUuid.find(fullMaterialName);
      uuidIt != cend(m_materialNameToUuid)) {
    if (auto indexIt = m_uuidToIndex.find(std::get<0>(uuidIt->second));
        indexIt != cend(m_uuidToIndex)) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
          "Reusing compiled material {}",
          fullMaterialName);
      m_targetCodes[indexIt->second].refCount++;
      return uuidIt->second;
    }
  }

  // If not, try and get a compiled version of it.
  auto transaction = make_handle(m_core->createTransaction(m_scope.get()));
  bool doCommit = false;
  auto finalizeTransaction =
      nonstd::make_scope_exit([transaction, &doCommit]() {
        if (doCommit) {
          transaction->commit();
        } else {
          transaction->abort();
        }
      });

  auto module = make_handle(m_core->loadModule(moduleName, transaction.get()));
  if (!module.is_valid_interface()) {
    m_core->logMessage(
        mi::base::MESSAGE_SEVERITY_ERROR, "Cannot find module {}", moduleName);
    return {};
  }

  auto functionDef = make_handle(m_core->getFunctionDefinition(
      module.get(), materialName, transaction.get()));
  if (!functionDef.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot find function {} definition in module {}",
        materialName,
        moduleName);
    return {};
  }

  auto compiledMaterial =
      make_handle(m_core->getCompiledMaterial(functionDef.get()));
  if (!compiledMaterial.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Failed compiling material {} for module {}",
        materialName,
        moduleName);
    return {};
  }

  // Get the compiled material hash and its target code. Get the default and
  // body resources state from that.
  auto uuid = compiledMaterial->get_hash();
  auto targetCode = make_handle(
      m_core->getPtxTargetCode(compiledMaterial.get(), transaction.get()));
  std::vector<libmdl::TextureDescriptor> textureDescs;

  for (auto i = 1ul; i < targetCode->get_texture_count(); ++i) {
    libmdl::TextureDescriptor textureDesc{i - 1, targetCode->get_texture(i)};

    switch (targetCode->get_texture_shape(i)) {
    case mi::neuraylib::ITarget_code::Texture_shape_2d: {
      textureDesc.shape = libmdl::Shape::TwoD;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_3d: {
      textureDesc.shape = libmdl::Shape::ThreeD;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_cube: {
      textureDesc.shape = libmdl::Shape::Cube;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data: {
      textureDesc.shape = libmdl::Shape::BsdfData;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_ptex: {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "Ptex textures are not supported by VisRTX");
      textureDesc.shape = libmdl::Shape::Unknown;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_invalid:
    default: {
      textureDesc.shape = libmdl::Shape::Unknown;
      break;
    }
    }

    if (targetCode->get_texture_shape(i)
        == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data) {
      mi::Size x, y, z;
      const char *pixelFormat = "Float32";
#if MI_NEURAYLIB_API_VERSION >= 56
      textureDesc.bsdf.data =
          targetCode->get_texture_df_data(i, x, y, z, pixelFormat);
#else
      textureDesc.bsdf.data = targetCode->get_texture_df_data(i, x, y, z);
#endif
      textureDesc.bsdf.dims[0] = x;
      textureDesc.bsdf.dims[1] = y;
      textureDesc.bsdf.dims[2] = z;
      textureDesc.bsdf.pixelFormat = pixelFormat;
      textureDesc.url =
          fmt::format("bsdf_data_{}", fmt::ptr(textureDesc.bsdf.data));
    } else {
      auto moduleOwner = targetCode->get_texture_owner_module(i);
      auto url = std::string(targetCode->get_texture_url(i));
      url = m_core->resolveResource(url.c_str(), moduleOwner);
      if (url.empty()) {
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
            "Failed to resolve texture resource {} for material {}",
            targetCode->get_texture_url(i),
            fullMaterialName);
        textureDesc.url = {};
      } else
        textureDesc.url = url;
    }

    switch (targetCode->get_texture_gamma(i)) {
    case mi::neuraylib::ITarget_code::GM_GAMMA_DEFAULT:
    case mi::neuraylib::ITarget_code::GM_GAMMA_LINEAR:
    case mi::neuraylib::ITarget_code::GM_GAMMA_UNKNOWN: {
      textureDesc.colorSpace = libmdl::ColorSpace::Linear;
      break;
    }
    case mi::neuraylib::ITarget_code::GM_GAMMA_SRGB: {
      textureDesc.colorSpace = libmdl::ColorSpace::sRGB;
      break;
    }
#if MI_NEURAYLIB_API_VERSION < 57
    case mi::neuraylib::ITarget_code::GM_FORCE_32_BIT: {
      assert(false);
      break;
    }
#endif
    }

    textureDescs.push_back(textureDesc);
  }

  libmdl::ArgumentBlockDescriptor argBlockDesc(m_core,
      compiledMaterial.get(),
      targetCode.get(),
      std::move(textureDescs));

  // Reuse an existing targetCode and its matching ptx generated code if we
  // already have it.
  if (auto it = m_uuidToIndex.find(uuid); it != std::cend(m_uuidToIndex)) {
    ++m_targetCodes[it->second].refCount;
    return {uuid, argBlockDesc};
  }

  // First time we hit this. Build a complete PTX shader from the generated
  // blob.
  auto ptxBlob = libmdl::stitchPTXs(std::vector{
      nonstd::span{reinterpret_cast<const char *>(ptx::MDLTexture.ptr),
          ptx::MDLTexture.size},
      nonstd::span{targetCode->get_code(), targetCode->get_code_size()},
      nonstd::span{
          reinterpret_cast<const char *>(ptx::MDLShaderEvalSurfaceMaterial.ptr),
          ptx::MDLShaderEvalSurfaceMaterial.size},
  });

  // Find an empty slot if possible
  auto targetIt = std::find_if(std::begin(m_targetCodes),
      std::end(m_targetCodes),
      [](const auto &v) { return v.refCount == 0; });

  if (targetIt == std::end(m_targetCodes)) {
    targetIt = m_targetCodes.insert(std::end(m_targetCodes), {ptxBlob, 1});
  } else {
    targetIt->ptxBlob = ptxBlob;
    targetIt->refCount = 1;
  }

  auto targetIndex = std::distance(std::begin(m_targetCodes), targetIt);

  m_uuidToIndex.insert({uuid, targetIndex});
  m_materialNameToUuid.insert(
      {fullMaterialName, std::tuple{uuid, argBlockDesc}});

  m_lastUpdateTS = libmdl::newTimeStamp();

  // Make sure we commit the transaction.
  m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
      "Acquired material {} with uuid {:04x}-{:04x}-{:04x}-{:04x}",
      fullMaterialName,
      uuid.m_id1,
      uuid.m_id2,
      uuid.m_id3,
      uuid.m_id4);
  doCommit = true;
  return {uuid, argBlockDesc};
}

std::optional<libmdl::ArgumentBlockInstance>
MaterialRegistry::createArgumentBlock(
    const libmdl::ArgumentBlockDescriptor &argumentBlockDescriptor) const
{
  auto result = std::optional<libmdl::ArgumentBlockInstance>{};

  result = libmdl::ArgumentBlockInstance(argumentBlockDescriptor, m_core);

  return result;
}

void MaterialRegistry::releaseMaterial(const Uuid &uuid)
{
  if (auto it = m_uuidToIndex.find(uuid); it != std::end(m_uuidToIndex)) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
        "Releasing material with uuid {:04x}-{:04x}-{:04x}-{:04x}",
        uuid.m_id1,
        uuid.m_id2,
        uuid.m_id3,
        uuid.m_id4);
    if (--m_targetCodes[it->second].refCount == 0) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
          "Destroying material with uuid {:04x}-{:04x}-{:04x}-{:04x}",
          uuid.m_id1,
          uuid.m_id2,
          uuid.m_id3,
          uuid.m_id4);
      m_targetCodes[it->second] = {};
      m_uuidToIndex.erase(it);
      m_lastUpdateTS = libmdl::newTimeStamp();
    }
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot release material with UUID {:04x}-{:04x}-{:04x}-{:04x}",
        uuid.m_id1,
        uuid.m_id2,
        uuid.m_id3,
        uuid.m_id4);
  }
}

} // namespace visrtx::mdl
