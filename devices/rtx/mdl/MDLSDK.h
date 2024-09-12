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

#include "VisRTXDevice.h"
#include "mdl/MDLMaterialManager.h"

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
#include <utility>
#include <vector>

namespace visrtx {

class MDLMaterialManager;

class MDLSDK
{
  class Logger;

  friend class MDLMaterialManager;

 public:
  MDLSDK(VisRTXDevice *device);
  ~MDLSDK();

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

  std::optional<MDLSDK::CompilationResult> compileMaterial(
      mi::neuraylib::ITransaction *transaction,
      const std::string &material_name,
      std::vector<mi::neuraylib::Target_function_description> &descs,
      bool class_compilation = true);

  /// Create an MDL SDK transaction.
  mi::base::Handle<mi::neuraylib::ITransaction> createTransaction()
  {
    return mi::base::make_handle<mi::neuraylib::ITransaction>(
        m_globalScope->create_transaction());
  }

  /// Get the image API component.
  mi::base::Handle<mi::neuraylib::IImage_api> getImagApi()
  {
    return m_imageApi;
  }

  /// Get the import export API component.
  mi::base::Handle<mi::neuraylib::IMdl_impexp_api> getImpexpApi()
  {
    return mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(
        m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
  }

  void addMdlSearchPath(const std::filesystem::path &path);
  void removeMdlSearchPath(const std::filesystem::path &path);

  bool isValid() const
  {
    return m_neuray && m_neuray.is_valid_interface()
        && m_imageApi.is_valid_interface();
  }

 private:
  VisRTXDevice *m_device;

  mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
  mi::base::Handle<mi::base::ILogger> m_logger;
  mi::base::Handle<mi::neuraylib::IMdl_compiler> m_mdlCompiler;
  mi::base::Handle<mi::neuraylib::IMdl_configuration> m_mdlConfiguration;
  mi::base::Handle<mi::neuraylib::IDatabase> m_database;
  mi::base::Handle<mi::neuraylib::IScope> m_globalScope;
  mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdlFactory;

  mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_executionContext;

  mi::base::Handle<mi::neuraylib::IMdl_backend> m_backendCudaPtx;
  mi::base::Handle<mi::neuraylib::IImage_api> m_imageApi;

  mi::neuraylib::INeuray *loadAndGetINeuray(const char *filename = nullptr);
  bool unloadINeuray();
  mi::Sint32 loadPlugin(mi::neuraylib::INeuray *neuray, const char *path);

  template <typename... Args>
  void reportMessage(
      ANARIStatusSeverity severity, const char *format, Args &&...args)
  {
    m_device->reportMessage(severity, format, std::forward<Args>(args)...);
  }

  bool logMessages(mi::neuraylib::IMdl_execution_context *context);
  bool parseCmdArgumentMaterialName(const std::string &argument,
    std::string &out_module_name,
    std::string &out_material_name,
    bool prepend_colons_if_missing);

  struct UuidHasher
  {
    std::size_t operator()(const mi::base::Uuid &uuid) const noexcept
    {
      return mi::base::uuid_hash32(uuid);
    }
  };

  using TargetCodeCache = std::unordered_map<mi::base::Uuid,
      mi::base::Handle<mi::neuraylib::ITarget_code const>,
      UuidHasher>;

  /// Maps a compiled material hash to a target code object to avoid generation
  /// of duplicate code.
  TargetCodeCache m_targetCodeCache;

#ifdef MI_PLATFORM_WINDOWS
  HMODULE m_dso_handle = {};
#else
  void *m_dso_handle = {};
#endif
};

} // namespace visrtx
