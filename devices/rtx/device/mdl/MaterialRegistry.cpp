/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "libmdl/source_name_utils.h"
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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std::string_literals;

extern "C" const char VISRTX_DEFAULT_MDL[];
extern "C" const std::size_t VISRTX_DEFAULT_MDL_size;
extern "C" const char VISRTX_PHYSICALLY_BASED_MDL[];
extern "C" const std::size_t VISRTX_PHYSICALLY_BASED_MDL_size;

namespace visrtx::mdl {

MaterialRegistry::MaterialRegistry(libmdl::Core *core)
    : m_core(core),
      m_scope(m_core->createScope("VisRTXMaterialResgistryScope"s
          + std::to_string(std::uintptr_t(this))))
{
  // Build the string_view from the embedded array and its exact size: the
  // embedded resource is not a C string, so a strlen-based view would overrun.
  m_core->addBuiltinModule("::visrtx::default",
      std::string_view(VISRTX_DEFAULT_MDL, VISRTX_DEFAULT_MDL_size));
  m_core->addBuiltinModule("::visrtx::physically_based",
      std::string_view(
          VISRTX_PHYSICALLY_BASED_MDL, VISRTX_PHYSICALLY_BASED_MDL_size));
}

MaterialRegistry::~MaterialRegistry()
{
  m_core->removeScope(m_scope.get());
}

std::optional<MaterialRegistry::AcquiredMaterial>
MaterialRegistry::reuseCompiledMaterial(const std::string &fullMaterialName)
{
  auto uuidIt = m_materialNameToUuid.find(fullMaterialName);
  if (uuidIt == cend(m_materialNameToUuid))
    return std::nullopt;

  auto indexIt = m_uuidToIndex.find(uuidIt->second.uuid);
  if (indexIt == cend(m_uuidToIndex))
    return std::nullopt;

  m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
      "Reusing compiled material {}",
      fullMaterialName);
  m_targetCodes[indexIt->second].refCount++;
  return uuidIt->second;
}

std::uint32_t MaterialRegistry::maxConcurrentCompiles() const
{
  return m_maxConcurrentCompiles.load(std::memory_order_relaxed);
}

MaterialRegistry::PtxSnapshot MaterialRegistry::ptxSnapshot() const
{
  PtxSnapshot snapshot;
  snapshot.timestamp = m_lastUpdateTS;
  snapshot.blobs.reserve(m_targetCodes.size());
  for (const auto &target : m_targetCodes)
    snapshot.blobs.push_back(target.ptxBlob);
  return snapshot;
}

std::optional<std::string> MaterialRegistry::preloadModule(
    const std::string &moduleOrSource, bool fromCode)
{
  using mi::base::make_handle;

  auto transaction = make_handle(m_core->createTransaction(m_scope.get()));
  auto abortTransaction = nonstd::make_scope_exit([&] {
    if (transaction->is_open())
      transaction->abort();
  });

  std::string moduleDbName;
  {
    auto module = fromCode
        ? make_handle(m_core->loadModuleFromString(
              libmdl::makeInlineModuleName(moduleOrSource),
              moduleOrSource,
              transaction.get()))
        : make_handle(m_core->loadModule(moduleOrSource, transaction.get()));
    if (!module.is_valid_interface()) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
          "Cannot load module {}",
          moduleOrSource);
      return std::nullopt;
    }
    const char *name = transaction->name_of(module.get());
    if (!name) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
          "Cannot find database name for module {}",
          moduleOrSource);
      return std::nullopt;
    }
    moduleDbName = name;
  } // Release the module handle before committing its transaction.
  if (transaction->commit() != 0) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot commit module {}",
        moduleOrSource);
    return std::nullopt;
  }
  return moduleDbName;
}

std::optional<MaterialRegistry::CompileProduct>
MaterialRegistry::compileMaterial(const std::string &moduleDbName,
    const std::string &materialName,
    MdlCompileCoordinator &coordinator)
{
  using mi::base::make_handle;

  const auto active =
      m_activeCompiles.fetch_add(1, std::memory_order_relaxed) + 1;
  auto finishCompile = nonstd::make_scope_exit(
      [&] { m_activeCompiles.fetch_sub(1, std::memory_order_relaxed); });
  auto peak = m_maxConcurrentCompiles.load(std::memory_order_relaxed);
  while (peak < active
      && !m_maxConcurrentCompiles.compare_exchange_weak(
          peak, active, std::memory_order_relaxed)) {
  }

  auto transaction = make_handle(m_core->createTransaction(m_scope.get()));
  bool doCommit = false;
  auto finalizeTransaction = nonstd::make_scope_exit([&] {
    if (doCommit)
      transaction->commit();
    else
      transaction->abort();
  });

  // Access only: even an idempotent load can enter the shared entity resolver.
  // The coordinator already selected and committed the exact module to use.
  auto module = make_handle(
      transaction->access<mi::neuraylib::IModule>(moduleDbName.c_str()));
  if (!module.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot access preloaded module {}",
        moduleDbName);
    return std::nullopt;
  }
  const std::string moduleName = module->get_mdl_name();
  const auto fullMaterialName = fmt::format("{}::{}", moduleName, materialName);

  auto functionDef = make_handle(m_core->getFunctionDefinition(
      module.get(), materialName, transaction.get()));
  if (!functionDef.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot find function {} definition in module {}",
        materialName,
        moduleName);
    return std::nullopt;
  }

  auto compiledMaterial =
      make_handle(m_core->getCompiledMaterial(functionDef.get()));
  if (!compiledMaterial.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Failed compiling material {} for module {}",
        materialName,
        moduleName);
    return std::nullopt;
  }

  // Get the compiled material hash and its target code. Get the default and
  // body resources state from that.
  auto uuid = compiledMaterial->get_hash();
  auto targetCode = make_handle(
      m_core->getPtxTargetCode(compiledMaterial.get(), transaction.get()));
  std::vector<libmdl::TextureDescriptor> textureDescs;
  if (!targetCode.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Failed generating PTX target code for material {}",
        fullMaterialName);
    return std::nullopt;
  }

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
      // Relative resource paths (e.g. "Textures/foo.png") resolve against the
      // owner module's directory. Pass the compiled module's on-disk path so
      // resolution doesn't depend on the owner module name being reachable
      // through the MDL search paths.
      auto moduleOwner = std::string(targetCode->get_texture_owner_module(i));
      auto ownerName =
          moduleOwner.empty() ? std::string(moduleName) : moduleOwner;
      const char *moduleFile = module->get_filename();
      auto url = std::string(targetCode->get_texture_url(i));
      // The entity resolver is shared, so serialize resolution on the
      // coordinator while the worker waits.
      auto resolve = [&] {
        return m_core->resolveResource(
            url.c_str(), ownerName, moduleFile ? moduleFile : "");
      };
      url = coordinator.run(resolve);
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

  // Build the complete PTX shader on the worker.
  auto ptxBlob = libmdl::stitchPTXs(std::vector{
      nonstd::span{reinterpret_cast<const char *>(ptx::MDLTexture.ptr),
          ptx::MDLTexture.size},
      nonstd::span{targetCode->get_code(), targetCode->get_code_size()},
      nonstd::span{
          reinterpret_cast<const char *>(ptx::MDLShaderEvalSurfaceMaterial.ptr),
          ptx::MDLShaderEvalSurfaceMaterial.size},
  });

  // Extract the emission IR now, while the compiled material is alive — it is
  // not retained. The material folds this IR against its live arguments at
  // finalize to publish an emission descriptor (ADR 0007).
  auto emission =
      libmdl::buildEmissionIR(compiledMaterial.get(), transaction.get());

  doCommit = true;
  return CompileProduct{uuid,
      std::move(ptxBlob),
      std::move(emission),
      std::move(argBlockDesc),
      libmdl::SceneDataDescriptor(*targetCode)};
}

MaterialRegistry::AcquiredMaterial MaterialRegistry::insertCompiled(
    const std::string &fullMaterialName, CompileProduct product)
{
  // Reuse an existing slot for the same compiled hash: a different name (or a
  // concurrent duplicate compile) mapping to the same content shares one slot.
  if (auto it = m_uuidToIndex.find(product.uuid);
      it != std::cend(m_uuidToIndex)) {
    auto &target = m_targetCodes[it->second];
    ++target.refCount;
    return AcquiredMaterial{
        product.uuid, product.argBlockDescriptor, target.sceneData};
  }

  // Find an empty slot if possible. Slots are assigned here on the coordinator
  // thread as each worker's compile completes, so slot indices follow
  // compile-completion order (not submission order) and are not stable across
  // runs. That is fine: every consumer looks a material up by uuid via
  // getMaterialImplementationIndex, and the PTX fingerprint XORs slots so it is
  // order-independent.
  auto targetIt = std::find_if(std::begin(m_targetCodes),
      std::end(m_targetCodes),
      [](const auto &v) { return v.refCount == 0; });

  if (targetIt == std::end(m_targetCodes)) {
    targetIt = m_targetCodes.insert(
        std::end(m_targetCodes), {std::move(product.ptxBlob), 1});
  } else {
    targetIt->ptxBlob = std::move(product.ptxBlob);
    targetIt->refCount = 1;
  }
  targetIt->emission = std::move(product.emission);
  targetIt->sceneData = std::move(product.sceneData);

  auto targetIndex = std::distance(std::begin(m_targetCodes), targetIt);

  m_uuidToIndex.insert({product.uuid, targetIndex});
  const AcquiredMaterial acquired{
      product.uuid, product.argBlockDescriptor, targetIt->sceneData};
  m_materialNameToUuid.insert({fullMaterialName, acquired});

  m_lastUpdateTS = libmdl::newTimeStamp();

  m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
      "Acquired material {} with uuid {:04x}-{:04x}-{:04x}-{:04x}",
      fullMaterialName,
      product.uuid.m_id1,
      product.uuid.m_id2,
      product.uuid.m_id3,
      product.uuid.m_id4);
  return acquired;
}

std::future<MaterialRegistry::AcquiredMaterial>
MaterialRegistry::acquireMaterialAsync(MdlCompileCoordinator &coordinator,
    std::string moduleOrSource,
    std::string materialName,
    bool fromCode)
{
  return coordinator.submit([this,
                                &coordinator,
                                moduleOrSource = std::move(moduleOrSource),
                                materialName = std::move(materialName),
                                fromCode]() -> AcquiredMaterial {
    const std::string moduleName = fromCode
        ? std::string(libmdl::makeInlineModuleName(moduleOrSource))
        : moduleOrSource;
    const auto fullMaterialName =
        fmt::format("{}::{}", moduleName, materialName);

    // Dedup, preload and insert mutate or read the registry / SDK database, so
    // they run on the coordinator thread; only the compile runs here on the
    // worker, in parallel with other materials' compiles.
    if (auto cached = coordinator.run(
            [&] { return reuseCompiledMaterial(fullMaterialName); }))
      return *cached;
    auto moduleDbName = coordinator.run(
        [&] { return preloadModule(moduleOrSource, fromCode); });
    if (!moduleDbName)
      return {};
    auto product = compileMaterial(*moduleDbName, materialName, coordinator);
    if (!product)
      return {};
    return coordinator.run(
        [&] { return insertCompiled(fullMaterialName, std::move(*product)); });
  });
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
      // Prune the name cache with the slot: a stale name entry pins the
      // descriptor (MDL-SDK handles included) forever under distinct-source
      // churn, and its insert() would silently no-op if the same name later
      // resolved to a different content hash.
      for (auto nameIt = std::begin(m_materialNameToUuid);
          nameIt != std::end(m_materialNameToUuid);) {
        if (nameIt->second.uuid == uuid)
          nameIt = m_materialNameToUuid.erase(nameIt);
        else
          ++nameIt;
      }
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
