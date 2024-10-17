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

#pragma once

#include "MDLMaterialInfo.h"
#include "VisRTXDevice.h"
#include "gpu/gpu_objects.h"
#include "renderer/MaterialSbtData.cuh"

#include "optix_visrtx.h"

#include <anari/frontend/anari_enums.h>

#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/types.h>
#include <mi/base/uuid.h>
#include <mi/mdl_sdk.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_compiler.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>

#ifdef MI_PLATFORM_WINDOWS
#include <direct.h>
#include <mi/base/miwindows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace visrtx {

class Sampler;
class VisRTXDevice;

struct MDLParamDesc
{
  enum class Semantic
  {
    Scalar, // Treat inputs as singular values
    Range, // The set of inputs is to be considered as a range ()
    Color,
  };
  std::string name;
  std::string displayName;
  ANARIDataType type;
  Semantic semantic;
};

class MDLCompiler
{
  friend class VisRTXDevice;
public:
  static MDLCompiler* getMDLCompiler(DeviceGlobalState* deviceState) {
    auto it = s_instances.find(deviceState);
    return it != s_instances.end() ? it->second : nullptr;
  }

  using Uuid = mi::base::Uuid;

  struct CompilationResult
  {
    // Uuid of the compiled material.
    Uuid uuid;

    // The generated target code object.
    mi::base::Handle<mi::neuraylib::ITarget_code const> targetCode;

    // FIXME: Handle some MDLMaterialInfo struct holding a description of the
    // compiled material and related argblocks.
  };

  /// Compiles a material from a given source file.
  std::optional<CompilationResult> compileMaterial(
      mi::neuraylib::ITransaction *transaction,
      const std::string &material_name,
      std::vector<mi::neuraylib::Target_function_description> &descs,
      const ptx_blob& llvmRenderModule = {},
      bool class_compilation = true);

  /// Create an MDL SDK transaction.
  mi::base::Handle<mi::neuraylib::ITransaction> createTransaction()
  {
    return mi::base::make_handle(m_deviceState->mdl.globalScope->create_transaction());
  }

  /// Get the image API component.
  mi::base::Handle<mi::neuraylib::IImage_api> getImageApi()
  {
    return m_deviceState->mdl.imageApi;
  }

  /// Get the import export API component.
  mi::base::Handle<mi::neuraylib::IMdl_impexp_api> getImpexpApi()
  {
    return mi::base::make_handle(m_deviceState->mdl.neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
  }

  void addMdlSearchPath(const std::filesystem::path &path);
  void removeMdlSearchPath(const std::filesystem::path &path);

  bool isValid() const
  {
    return m_deviceState && m_deviceState->mdl.neuray.is_valid_interface();
  }

  using Index = uint64_t;
  static const constexpr auto INVALID_ID = std::numeric_limits<Index>::max();

  Uuid acquireModule(const char *modulePath);
  void releaseModule(Uuid moduleUuid);
  Index getModuleIndex(Uuid moduleUuid) const
  {
    auto it = m_uuidToIndex.find(moduleUuid);
    if (it == m_uuidToIndex.cend()) {
      return INVALID_ID;
    }
    return it->second;
  }

  std::vector<MDLParamDesc> getModuleParameters(Uuid moduleUuid);
  std::vector<ptx_blob> getPTXBlobs();
  std::vector<MaterialSbtData> getMaterialSbtEntries();

  std::vector<const Sampler*> getModuleSamplers(Uuid moduleUuid) const {
    return m_materialImplementations[getModuleIndex(moduleUuid)].materialInfo.getSamplers();
  }
 private:
   MDLCompiler(const MDLCompiler&) = default;
   MDLCompiler(MDLCompiler&&) = default;
   MDLCompiler& operator=(const MDLCompiler&) = default;
   MDLCompiler& operator=(MDLCompiler&&) = default;

   friend class std::unordered_map<DeviceGlobalState* const, MDLCompiler>;
   friend class std::pair<DeviceGlobalState* const, MDLCompiler>;
   friend class std::pair<DeviceGlobalState*, visrtx::MDLCompiler>;
   friend class std::allocator<std::pair<visrtx::DeviceGlobalState* const, visrtx::MDLCompiler> >;

   MDLCompiler(DeviceGlobalState* deviceState);
  ~MDLCompiler();

  static bool setUp(DeviceGlobalState* deviceState);
  static void tearDown(DeviceGlobalState* deviceState);

  static std::unordered_map<DeviceGlobalState*, MDLCompiler*> s_instances;

  DeviceGlobalState* m_deviceState;


  struct UuidHasher
  {
    std::size_t operator()(const mi::base::Uuid &uuid) const noexcept
    {
      return mi::base::uuid_hash32(uuid);
    }
  };

  struct MaterialImplementation
  {
    MDLMaterialInfo materialInfo;
    std::vector<unsigned char> ptxBlob;
  };
  std::vector<MaterialImplementation> m_materialImplementations;
  std::unordered_map<mi::base::Uuid, uint64_t, UuidHasher> m_uuidToIndex;

  DeviceBuffer m_argblocksBuffer;

#ifdef MI_PLATFORM_WINDOWS
  using DllHandle = HMODULE;
#else
  using DllHandle = void*;
#endif

  // MDL API helpers
  static DllHandle loadMdlSdk(const DeviceGlobalState* deviceState, const char *filename = nullptr);
  static bool unloadMdlSdk(const DeviceGlobalState* deviceState, DllHandle handle);
  static mi::neuraylib::INeuray* getINeuray(const DeviceGlobalState* deviceState, DllHandle handle);
  
  bool parseCmdArgumentMaterialName(
    const std::string &argument,
    std::string &out_module_name,
    std::string &out_material_name,
    bool prepend_colons_if_missing);

  static std::string addMissingMaterialSignature(const mi::neuraylib::IModule *module, const std::string &material_name);
  static std::string messageKindToString(mi::neuraylib::IMessage::Kind messageKind);
  bool logExecutionContextMessages(mi::neuraylib::IMdl_execution_context *context);

  // Textures
  Sampler*  prepareTexture(
    mi::neuraylib::ITransaction* transaction,
    char const* texture_db_name,
    mi::neuraylib::ITarget_code::Texture_shape textureShape);


  // ANARI reporting
  template<typename... Args>
  void reportMessage(ANARIStatusSeverity severity, const char *fmt, Args &&...args) const;

  template<typename... Args>
  static void reportMessage(const DeviceGlobalState* deviceState, ANARIStatusSeverity severity, const char *fmt, Args &&...args);
};

} // namespace visrtx
