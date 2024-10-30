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

#include "MDLCompiler.h"

#include "MDLShaderEvalSurfaceMaterial_ptx.h"
#include "MDLTexture_ptx.h"
#include "array/Array2D.h"
#include "array/Array3D.h"
#include "optix_visrtx.h"

#include "scene/surface/material/sampler/Image2D.h"
#include "scene/surface/material/sampler/Image3D.h"

#include <anari/frontend/anari_enums.h>
#include <helium/BaseDevice.h>
#include <mi/base/config.h>
#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/uuid.h>
#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/ilogging_configuration.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/iversion.h>
#include <mi/neuraylib/version.h>
#include <anari/anari_cpp.hpp>

#include <string_view>

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#include <direct.h>
#ifdef UNICODE
#define FMT_LPTSTR "%ls"
#else // UNICODE
#define FMT_LPTSTR "%s"
#endif // UNICODE
#else
#include <dlfcn.h>
#endif


#include <fstream>

using namespace std::string_view_literals;

namespace {

// Returns a string-representation of the given message severity
inline ANARIStatusSeverity miSeverityToAnari(
    mi::base::Message_severity severity)
{
  switch (severity) {
  case mi::base::MESSAGE_SEVERITY_ERROR:
    return ANARI_SEVERITY_ERROR;
  case mi::base::MESSAGE_SEVERITY_WARNING:
    return ANARI_SEVERITY_WARNING;
  case mi::base::MESSAGE_SEVERITY_INFO:
    return ANARI_SEVERITY_INFO;
  case mi::base::MESSAGE_SEVERITY_VERBOSE:
    return ANARI_SEVERITY_INFO;
  case mi::base::MESSAGE_SEVERITY_DEBUG:
    return ANARI_SEVERITY_DEBUG;
  default:
    return ANARI_SEVERITY_INFO;
  }
}

inline std::string string_printf(const char *fmt, ...)
{
  std::string s;
  va_list args, args2;
  va_start(args, fmt);
  va_copy(args2, args);

  s.resize(vsnprintf(nullptr, 0, fmt, args2) + 1);
  va_end(args2);
  vsprintf(s.data(), fmt, args);
  va_end(args);
  s.pop_back();
  return s;
}

} // namespace

namespace visrtx {

class Logger : public mi::base::Interface_implement<mi::base::ILogger>
{
 public:
  Logger(DeviceGlobalState *deviceState) : m_deviceState(deviceState) {}

  void message(mi::base::Message_severity level,
      const char *moduleCategory,
      const mi::base::Message_details &,
      const char *message) override
  {
    m_deviceState->messageFunction(miSeverityToAnari(level),
        string_printf("[MDL:%s]  %s", moduleCategory, message),
        ANARI_UNKNOWN,
        nullptr);
  }

 private:
  DeviceGlobalState *m_deviceState;
};

std::unordered_map<DeviceGlobalState*, MDLCompiler*> MDLCompiler::s_instances;

MDLCompiler::MDLCompiler(DeviceGlobalState *deviceState) : m_deviceState(deviceState) {}
MDLCompiler::~MDLCompiler() = default;

template <typename... Args>
inline void MDLCompiler::reportMessage(
    ANARIStatusSeverity severity, const char *fmt, Args &&...args) const
{
  m_deviceState->messageFunction(severity,
      string_printf(fmt, std::forward<Args>(args)...),
      ANARI_DEVICE,
      this);
}

template <typename... Args>
inline void MDLCompiler::reportMessage(const DeviceGlobalState *deviceState,
    ANARIStatusSeverity severity,
    const char *fmt,
    Args &&...args)
{
  deviceState->messageFunction(severity,
      string_printf(fmt, std::forward<Args>(args)...),
      ANARI_DEVICE,
      deviceState);
}

bool MDLCompiler::setUp(DeviceGlobalState *deviceState)
{
  // Create the neuray library instance.
  auto dllHandle = loadMdlSdk(deviceState);
  if (!dllHandle) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Failed to load the MDL SDK library. MDL support will be disabled.");
    return false;
  }
  auto neuray = mi::base::make_handle(getINeuray(deviceState, dllHandle));
  if (!neuray.is_valid_interface()) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Failed to load the MDL SDK. MDL support will be disabled.");
    return false;
  }

  // Early initialization of the logger.
  auto loggingConfig = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
  auto logger = mi::base::make_handle(new Logger(deviceState));
  loggingConfig->set_receiving_logger(logger.get());

  // Plugins
  auto pluginConf = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
  if (mi::Sint32 res = pluginConf->load_plugin_library(
          "nv_openimageio" MI_BASE_DLL_FILE_EXT);
      res == 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_INFO,
        "Successfully loaded the plugin library nv_openimageio");
  } else {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to load the nv_openimageio plugin.");
  }

  if (mi::Sint32 res =
          pluginConf->load_plugin_library("dds" MI_BASE_DLL_FILE_EXT);
      res == 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_INFO,
        "Successfully loaded the plugin library dds");
  } else {
    reportMessage(
        deviceState, ANARI_SEVERITY_WARNING, "Failed to load the dds plugin.");
  }

  // Handle user path.
  // FIXME: Get some additional paths from the device parameters.
  auto mdlConfiguration = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
  if (!mdlConfiguration.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Retrieving MDL configuration failed.");
    return false;
  }

  mdlConfiguration->add_mdl_system_paths();
  mdlConfiguration->add_mdl_user_paths();

  // Start the MDL SDK
  if (auto ret = neuray->start()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to initialize the SDK. Result code: %d",
        ret);
    return false;
  }

  // Setup the compiler
  auto mdlCompiler = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::IMdl_compiler>());
  if (!mdlCompiler.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Initialization of MDL compiler failed.");
    return false;
  }

  auto database = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::IDatabase>());
  if (!database.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to retrieve neuray database component.");
    return false;
  }
  auto globalScope = mi::base::make_handle(database->get_global_scope());
  if (!globalScope.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to get databae global scope.");
    return false;
  }

  auto mdlFactory = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::IMdl_factory>());
  if (!mdlFactory.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to retrieve MDL factory component.");
    return false;
  }

  auto executionContext =
      mi::base::make_handle(mdlFactory->create_execution_context());
  if (!executionContext.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to retrieve MDL factory execution context component.");
    return false;
  }

  // FIXME This needs to be configurable. Per device parameters?
  const int numTextureSpaces = 4; // ANARI attributes 0 to 3 
  const int numTextureResults = 16; // Number of actually supported textures. Let's assume this is enough for now.
  const bool enable_derivatives = false;

  auto mdlBackendApi = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
  if (!mdlBackendApi.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to retrieve MDL backend API component.");
    return false;
  }
  auto backendCudaPtx = mi::base::make_handle(
      mdlBackendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX));
  if (!backendCudaPtx.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to retrieve MDL PTX backend component.");
    return false;
  }

  if (backendCudaPtx->set_option(
          "num_texture_spaces", std::to_string(numTextureSpaces).c_str())
      != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Setting PTX option num_texture_spaces failed.");
    return false;
  }

  if (backendCudaPtx->set_option(
          "num_texture_results", std::to_string(numTextureResults).c_str())
      != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Setting PTX option num_texture_results failed.");
    return false;
  }
  if (backendCudaPtx->set_option("sm_version", "52") != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Setting PTX option sm_version failed.");
  }
  if (backendCudaPtx->set_option("tex_lookup_call_mode", "direct_call") != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Setting PTX option tex_lookup_call_mode failed.");
    return false;
  }
  if (backendCudaPtx->set_option("lambda_return_mode", "value") != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Setting PTX option lambda_return_mode failed.");
    return false;
  }
  if (enable_derivatives) {
    if (backendCudaPtx->set_option("texture_runtime_with_derivs", "on") != 0) {
      reportMessage(deviceState,
          ANARI_SEVERITY_WARNING,
          "Setting PTX option texture_runtime_with_derivs failed.");
    }
  }
  if (backendCudaPtx->set_option("inline_aggressively", "on") != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_WARNING,
        "Setting PTX option inline_aggressively failed.");
  }

  if (backendCudaPtx->set_option("opt_level", "2") != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_WARNING,
        "Setting PTX option opt_level failed.");
  }

  if (backendCudaPtx->set_option("enable_exceptions", "off") != 0) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Setting PTX option enable_exceptions failed.");
    return false;
  }

  auto imageApi = mi::base::make_handle(
      neuray->get_api_component<mi::neuraylib::IImage_api>());
  if (!imageApi.is_valid_interface()) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "Failed to retrieve image API component.");
    return false;
  }

  deviceState->mdl = DeviceGlobalState::MDLContext{
      neuray,
      logger,
      mdlCompiler,
      mdlConfiguration,
      database,
      globalScope,
      mdlFactory,

      executionContext,

      backendCudaPtx,
      imageApi,

      {},

      dllHandle,
  };

  s_instances.insert({deviceState, new MDLCompiler(deviceState)});

  return true;
}

void MDLCompiler::tearDown(DeviceGlobalState *deviceState)
{
  auto it = s_instances.find(deviceState);
  if (it != end(s_instances)) {
    delete it->second;
    s_instances.erase(it);
  }

  // FIXME: Add support.
  deviceState->mdl.targetCodeCache.clear();

  deviceState->mdl.imageApi.reset();
  deviceState->mdl.backendCudaPtx.reset();
  deviceState->mdl.executionContext.reset();
  deviceState->mdl.mdlFactory.reset();
  deviceState->mdl.globalScope.reset();
  deviceState->mdl.database.reset();

  deviceState->mdl.mdlConfiguration.reset();
  deviceState->mdl.mdlCompiler.reset();

  deviceState->mdl.logger.reset();

  // Shut down the MDL SDK
  if (deviceState->mdl.neuray->shutdown() != 0) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Failed to shutdown the SDK.");
  }

  deviceState->mdl.neuray.reset();

  // Unload the MDL SDK
  if (!unloadMdlSdk(deviceState, deviceState->mdl.dllHandle)) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Failed to unload the SDK.");
  }
  deviceState->mdl.dllHandle = {};
}

bool MDLCompiler::parseCmdArgumentMaterialName(const std::string &argument,
    std::string &outModuleName,
    std::string &outMaterialName,
    bool prepend_colons_if_missing)
{
  outModuleName = "";
  outMaterialName = "";
  std::size_t p_left_paren = argument.rfind('(');
  if (p_left_paren == std::string::npos)
    p_left_paren = argument.size();
  std::size_t p_last = argument.rfind("::", p_left_paren - 1);

  bool starts_with_colons =
      argument.length() > 2 && argument[0] == ':' && argument[1] == ':';

  // check for mdle
  if (!starts_with_colons) {
    std::string potential_path = argument;
    std::string potential_material_name = "main";

    // input already has ::main attached (optional)
    if (p_last != std::string::npos) {
      potential_path = argument.substr(0, p_last);
      potential_material_name =
          argument.substr(p_last + 2, argument.size() - p_last);
    }

    // is it an mdle?
    if (potential_path.length() >= 5
        && potential_path.substr(potential_path.length() - 5) == ".mdle") {
      if (potential_material_name != "main") {
        reportMessage(ANARI_SEVERITY_ERROR,
            "Material and module name cannot be extracted from "
            "'%s'.\nThe module was detected as MDLE but the selected material is "
            "different from 'main'.\n",
            argument.c_str());
        return false;
      }
      outModuleName = potential_path;
      outMaterialName = potential_material_name;
      return true;
    }
  }

  if (p_last == std::string::npos || p_last == 0
      || p_last == argument.length() - 2
      || (!starts_with_colons && !prepend_colons_if_missing)) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Material and module name cannot be extracted from '%s'.\n"
        "An absolute fully-qualified material name of form "
        "'[::<package>]::<module>::<material>' is expected.\n",
        argument.c_str());
    return false;
  }

  if (!starts_with_colons && prepend_colons_if_missing) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "The provided argument '%s' is not an absolute fully-qualified"
        " material name, a leading '::' has been added.\n",
        argument.c_str());
    outModuleName = "::";
  }

  outModuleName.append(argument.substr(0, p_last));
  outMaterialName = argument.substr(p_last + 2, argument.size() - p_last);
  return true;
}

std::string MDLCompiler::addMissingMaterialSignature(
    const mi::neuraylib::IModule *module, const std::string &material_name)
{
  // Return input if it already contains a signature.
  if (material_name.back() == ')')
    return material_name;

  auto result = mi::base::make_handle(
      module->get_function_overloads(material_name.c_str()));
  if (!result || result->get_length() != 1)
    return std::string();

  auto overloads = mi::base::make_handle(
      result->get_element<mi::IString>(static_cast<mi::Size>(0)));
  return overloads->get_c_str();
}

std::string MDLCompiler::messageKindToString(
    mi::neuraylib::IMessage::Kind messageKind)
{
  switch (messageKind) {
  case mi::neuraylib::IMessage::MSG_INTEGRATION:
    return "[MDL:SDK]";
  case mi::neuraylib::IMessage::MSG_IMP_EXP:
    return "[MDL:Importer/Exporter]";
  case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
    return "[MDL:Compiler Backend]";
  case mi::neuraylib::IMessage::MSG_COMILER_CORE:
    return "[MDL:Compiler Core]";
  case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
    return "[MDL:Compiler Archive Tool]";
  case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
    return "[MDL:Compiler DAG generator]";
  default:
    return {};
  }
}

// Prints the messages of the given context.
// Returns true, if the context does not contain any error messages, false
// otherwise.
bool MDLCompiler::logExecutionContextMessages(mi::neuraylib::IMdl_execution_context *context)
{
  for (mi::Size i = 0; i < context->get_messages_count(); ++i) {
    auto message = mi::base::make_handle(context->get_message(i));
    reportMessage(miSeverityToAnari(message->get_severity()),
        "%s: %s",
        messageKindToString(message->get_kind()).c_str(),
        message->get_string());
  }
  return context->get_error_messages_count() == 0;
}

std::optional<MDLCompiler::CompilationResult> MDLCompiler::compileMaterial(
    mi::neuraylib::ITransaction *transaction,
    std::string const &materialName,
    std::vector<mi::neuraylib::Target_function_description> &descs,
    const ptx_blob& llvmRenderModule,
    bool classCompilation)
// Material_info **out_mat_info)
{
  // Split module and material name
  std::string moduleName, materialSimpleName;
  if (!parseCmdArgumentMaterialName(
          materialName, moduleName, materialSimpleName, true)) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to parse material name %s",
        materialName.c_str());
    return {};
  }

  // Load the module
  auto mdlImpexpApi = getImpexpApi();
  mdlImpexpApi->load_module(transaction,
      moduleName.c_str(),
      m_deviceState->mdl.executionContext.get());
  if (!logExecutionContextMessages(m_deviceState->mdl.executionContext.get()))
    return {};

  // Get the database name for the module we loaded
  mi::base::Handle<const mi::IString> moduleDbName(
      m_deviceState->mdl.mdlFactory->get_db_module_name(moduleName.c_str()));
  mi::base::Handle<const mi::neuraylib::IModule> module(
      transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str()));
  if (!module) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to access the loaded module %s",
        moduleName.c_str());
    return {};
  }

  // Append the material name
  std::string materialDbName =
      std::string(moduleDbName->get_c_str()) + "::" + materialSimpleName;
  materialDbName = addMissingMaterialSignature(module.get(), materialDbName);
  if (materialDbName.empty()) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to find the material %s in the module %s",
        materialSimpleName.c_str(),
        moduleName.c_str());
    return {};
  }

  // Create a material instance from the material definition
  // with the default arguments.
  mi::base::Handle<const mi::neuraylib::IFunction_definition>
      functionDefinition(
          transaction->access<mi::neuraylib::IFunction_definition>(
              materialDbName.c_str()));
  if (!functionDefinition) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Cannot find material definition %s",
        materialDbName.c_str());
    return {};
  }

  mi::Sint32 ret = 0;
  mi::base::Handle<mi::neuraylib::IFunction_call> functionCall(
      functionDefinition->create_function_call(0, &ret));
  if (ret != 0) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed instantiating material %s",
        materialName.c_str());
    return {};
  }
  transaction->store(functionCall.get(), materialDbName.c_str());

  mi::base::Handle<mi::neuraylib::ICompiled_material const> compiledMaterial;
  // Create a compiled material
  mi::Uint32 flags = classCompilation
      ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
      : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

  mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance(
      functionCall->get_interface<mi::neuraylib::IMaterial_instance>());
  compiledMaterial = materialInstance->create_compiled_material(
      flags, m_deviceState->mdl.executionContext.get());

  if (!logExecutionContextMessages(m_deviceState->mdl.executionContext.get()))
    return {};

  mi::base::Handle<mi::neuraylib::ITarget_code const> targetCode;
  // Reuse old target code, if possible
  mi::base::Uuid materialHash = compiledMaterial->get_hash();
  auto it = m_deviceState->mdl.targetCodeCache.find(materialHash);
  if (it != m_deviceState->mdl.targetCodeCache.end()) {
    targetCode = it->second;
  } else {
    // Configure bitcode embdedding if any.
    if (llvmRenderModule.ptr && llvmRenderModule.size) {
      if (m_deviceState->mdl.backendCudaPtx->set_option_binary("llvm_renderer_module", reinterpret_cast<const char*>(llvmRenderModule.ptr), llvmRenderModule.size) != 0) {
        reportMessage(ANARI_SEVERITY_ERROR, "Failed to set llvm renderer module");
      }
    } else {
      if (m_deviceState->mdl.backendCudaPtx->set_option_binary("llvm_renderer_module", nullptr, 0) != 0) {
        reportMessage(ANARI_SEVERITY_ERROR, "Failed to reset llvm renderer module");
      }
    }

    // Generate target code for the compiled material
    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(
        m_deviceState->mdl.backendCudaPtx->create_link_unit(
            transaction, m_deviceState->mdl.executionContext.get()));
    linkUnit->add_material(compiledMaterial.get(),
        descs.data(),
        descs.size(),
        m_deviceState->mdl.executionContext.get());

    if (!logExecutionContextMessages(m_deviceState->mdl.executionContext.get()))
      return {};


    // m_deviceState->mdl.backendCudaPtx->set_option("visible_functions", "__direct_callable__evalSurfaceMaterial");
    targetCode = mi::base::Handle<const mi::neuraylib::ITarget_code>(
        m_deviceState->mdl.backendCudaPtx->translate_link_unit(
            linkUnit.get(), m_deviceState->mdl.executionContext.get()));
    if (!logExecutionContextMessages(m_deviceState->mdl.executionContext.get())) {
      return {};
    }

    m_deviceState->mdl.targetCodeCache[materialHash] = targetCode;

    transaction->store(functionCall.get(), materialDbName.c_str());
  }

  auto argblock = targetCode->get_argument_block_count() == 1
      ? targetCode->get_argument_block(0)
      : nullptr;

  return CompilationResult{
      materialHash,
      targetCode,
  };
}

void MDLCompiler::addMdlSearchPath(const std::filesystem::path &path)
{
  m_deviceState->mdl.mdlConfiguration->add_mdl_path(path.u8string().c_str());
}

void MDLCompiler::removeMdlSearchPath(const std::filesystem::path &path)
{
  m_deviceState->mdl.mdlConfiguration->remove_mdl_path(path.u8string().c_str());
}

void MDLCompiler::setMdlSearchPaths(const std::vector<std::filesystem::path> &paths)
{
  m_deviceState->mdl.mdlConfiguration->clear_mdl_paths();
  m_deviceState->mdl.mdlConfiguration->add_mdl_system_paths();
  m_deviceState->mdl.mdlConfiguration->add_mdl_user_paths();
  for (const auto& p : paths) addMdlSearchPath(p);
}

MDLCompiler::DllHandle MDLCompiler::loadMdlSdk(const DeviceGlobalState* deviceState, const char* filename) {
  if (!filename)
    filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;

  DllHandle handle = {};

#ifdef MI_PLATFORM_WINDOWS
  handle = LoadLibraryA(filename);
  if (handle == nullptr) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Failed to load MDL SDK.");
    LPTSTR buffer = 0;
    LPCTSTR message = TEXT("unknown failure");
    DWORD errorCode = GetLastError();
    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            0, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
      message = buffer;
    reportMessage(deviceState, ANARI_SEVERITY_ERROR,
        "Failed to load %s library (%u): " FMT_LPTSTR,
        filename,
        errorCode,
        message);
    if (buffer)
      LocalFree(buffer);
  }
#else
  handle = dlopen(filename, RTLD_LAZY);
  if (handle == nullptr) {
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR, "Failed load the MDL SDK library: %s", dlerror());
  }
#endif
  return handle;
}

bool MDLCompiler::unloadMdlSdk(const DeviceGlobalState* deviceState, DllHandle dllHandle) {
  if (dllHandle == nullptr) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Trying to unload MDL SDK while it is not loaded.");
    return false;
  }

  #ifdef MI_PLATFORM_WINDOWS
  if (!FreeLibrary(dllHandle)) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Failed unloading the MDL SDK library");
    return false;
  }
#else
  if (dlclose(dllHandle)) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Failed unloading the MDL SDK library: %s", dlerror());
    return false;
  }
#endif

  return true;
}

mi::neuraylib::INeuray* MDLCompiler::getINeuray(const DeviceGlobalState* deviceState, DllHandle dllHandle) {
#ifdef MI_PLATFORM_WINDOWS
  void *symbol = GetProcAddress(dllHandle, "mi_factory");
  if (!symbol) {
    LPTSTR buffer = 0;
    LPCTSTR message = TEXT("unknown failure");
    DWORD error_code = GetLastError();
    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
                | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            0,
            error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&buffer,
            0,
            0))
      message = buffer;
    reportMessage(deviceState,
        ANARI_SEVERITY_ERROR,
        "GetProcAddress error (%u): " FMT_LPTSTR,
        error_code,
        message);
    if (buffer)
      LocalFree(buffer);
    return nullptr;
  }
#else
  void *symbol = dlsym(dllHandle, "mi_factory");
  if (!symbol) {
    reportMessage(deviceState, ANARI_SEVERITY_ERROR,
        "Failed getting mi_factory from the MDL SDK library: %s",
        dlerror());
    return nullptr;
  }
#endif

  mi::neuraylib::INeuray *neuray =
      mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
  // Check if we have a valid neuray instance, otherwise check why.
  if (!neuray) {
    auto version = mi::base::make_handle(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
    if (!version)
      reportMessage(deviceState, ANARI_SEVERITY_ERROR, "Cannot get MDL SDK library version");
    else
      reportMessage(deviceState, ANARI_SEVERITY_ERROR, "MDL SDK library version %s does not match header version %s",
          version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
    return nullptr;
  }

  return neuray;
}

MDLCompiler::Uuid MDLCompiler::acquireModule(const char *materialName)
{
  auto transaction = createTransaction();

  // The entry points we want expose.
  std::vector<mi::neuraylib::Target_function_description> descs{
      {"surface.scattering", "mdlBsdf"},
      {"thin_walled", "mdl_isThinWalled"},
  };

#if 0
  // Some other entry points. Just there for the sake of not searching for them again.
  std::vector<mi::neuraylib::Target_function_description> descs;
  descs.push_back(
      mi::neuraylib::Target_function_description("init"));
  descs.push_back(
      mi::neuraylib::Target_function_description("surface.scattering"));
  descs.push_back(
      mi::neuraylib::Target_function_description("surface.emission.emission"));
  descs.push_back(
      mi::neuraylib::Target_function_description("surface.emission.intensity"));
  descs.push_back(
      mi::neuraylib::Target_function_description("volume.absorption_coefficient"));
  descs.push_back(
      mi::neuraylib::Target_function_description("thin_walled"));
  descs.push_back(
      mi::neuraylib::Target_function_description("geometry.cutout_opacity"));
  descs.push_back(
      mi::neuraylib::Target_function_description("hair"));
  descs.push_back(
      mi::neuraylib::Target_function_description("backface.scattering"));
  descs.push_back(
      mi::neuraylib::Target_function_description("backface.emission.emission"));
  descs.push_back(
      mi::neuraylib::Target_function_description("backface.emission.intensity"));
#endif


  reportMessage(ANARI_SEVERITY_INFO, "Compiling material %s", materialName);
  auto compilationResult = compileMaterial(transaction.get(), materialName, descs, {}, true);

  if (!compilationResult.has_value() || !compilationResult->targetCode.is_valid_interface()) {
    reportMessage(ANARI_SEVERITY_ERROR, "Failed compiling MDL: %s", materialName);
    return {};
  }
  reportMessage(ANARI_SEVERITY_INFO, "Successfully compiled %s", materialName);

  Uuid uuid = compilationResult->uuid;

  // PTX blob from the compiled code.
  auto targetCode = compilationResult->targetCode->get_code();
  auto targetCodeSize = compilationResult->targetCode->get_code_size();
  std::vector<unsigned char> ptxBlob(
      reinterpret_cast<const uint8_t *>(targetCode),
      reinterpret_cast<const uint8_t *>(targetCode) + targetCodeSize
  );


  // Hack time. An alternative would be to compile shaderMain as LLVM byte code
  // and then embed it using "llvm_render_module" parameter to MDL's ptx
  // backend. Lets save that for later if we want and add such a dependency on
  // LLVM.
  std::vector<unsigned char> shaderMain(MDLShaderEvalSurfaceMaterial_ptx, MDLShaderEvalSurfaceMaterial_ptx + sizeof(MDLShaderEvalSurfaceMaterial_ptx));
  std::vector<unsigned char> shaderTexture(MDLTexture_ptx, MDLTexture_ptx + sizeof(MDLTexture_ptx));

  // As we are blending to separate compilation unit PTXs, make sure headers are
  // not conflicting.
  static constexpr const auto versionKw = "\n.version "sv;
  static constexpr const auto targetKw = "\n.target "sv;
  static constexpr const auto addressSizeKw = "\n.address_size "sv;

  // Cleanup both ptxBlob and shaderMain for headers. Keep the highest version and target and insert those after the final string
  // is built.
  std::string version;
  std::string target;
  std::string addressSize = ".address_size 64";

  for (auto blobIt: { &shaderMain, &shaderTexture, &ptxBlob }) {
    auto& blob = *blobIt;
    for (auto dotkword : { versionKw, targetKw, addressSizeKw }) {
      if (auto it = std::search(begin(blob), end(blob), cbegin(dotkword), cend(dotkword));
          it != end(blob)) {
        auto eolIt = std::find(it + 1, end(blob), '\n');
        std::string sub(it + 1, eolIt);
        blob.erase(it, eolIt);
        if (dotkword == versionKw) { // .version
          if (sub > version)
            version = sub;
        } else if (dotkword == targetKw) { //.target
          if (sub > target)
            target = sub;
        }
      }
    }
  }

  // Same cleanup with forward decls possibly conflicting with actual
  // declarations.
  static constexpr const std::string_view externPrefixes[] = {
      "\n.extern .func "sv,
  };

  for (auto blobIt: { &shaderMain, &ptxBlob}) {
    auto& blob = *blobIt;
    for (const auto &externPrefix : externPrefixes) {
      for (auto it = std::search(begin(blob), end(blob), cbegin(externPrefix), cend(externPrefix));
          it != end(blob);) {
        auto semiColonIt = std::find(it, end(blob), ';');
        if (semiColonIt != end(blob)) {
          it = blob.erase(it, ++semiColonIt);
        }
        it = std::search(it, end(blob), cbegin(externPrefix), cend(externPrefix));
      }
    }
  }


  // And ditto with colliding symbol names (focusing on constant strings for
  // now).
  for (auto&& [blobIt, prefix]: {
        std::tuple{ &shaderMain, "_main_"sv },
        std::tuple{ &shaderTexture, "_texture_"sv },
  }) {
    auto& blob = *blobIt;
    static const std::string strPrefix = "$str";
    for (auto it = std::search(begin(blob), end(blob),cbegin(strPrefix), cend(strPrefix));
        it != end(blob);) {
      // Insertion might invalidate the iterator. Make sure to get the after insertion value.
      it = blob.insert(++it, cbegin(prefix), cend(prefix));
      // And search for next occurrence from there. Note that current $str has been broken to $_something_str and is not a match anymore
      // for the search, so we can resume at current position.
      it = std::search(it, end(blob), cbegin(strPrefix), cend(strPrefix));
    }
  }

  // Remove .visible qualifier in from of all functions
  for (auto blobIt: { &shaderMain, &shaderTexture, &ptxBlob }) {
    auto& blob = *blobIt;
    static constexpr std::string_view dotVisible = "\n.visible ";
    static constexpr std::string_view directCallableSemantic = "__direct_callable__";

    for (auto it = std::search(begin(blob), end(blob),cbegin(dotVisible), cend(dotVisible));
        it != end(blob);) {
      it = next(it); // Skip the new line.
      auto eolIt = std::find(it, end(blob), '\n');
      if (std::search(it, eolIt, cbegin(directCallableSemantic), cend(directCallableSemantic)) == eolIt) {
        it = blob.erase(it, it + dotVisible.size() - 1);
      }
      // And search for next occurrence from there. Note that current $str has been broken to $_something_str and is not a match anymore
      // for the search, so we can resume at current position.
      it = std::search(it, end(blob), cbegin(dotVisible), cend(dotVisible));
    }
  }

  // Last pass: remove duplicates between the texture shader and main shader.
  static constexpr const std::string_view weakGlobalPrefix[] = {
    ".weak .global "sv,
    ".weak .const "sv,
  };
  {
    std::vector<std::string> decls;
    for (const auto& prefix : weakGlobalPrefix) {
      for (auto it = std::search(begin(shaderTexture), end(shaderTexture),cbegin(prefix), cend(prefix));
        it != end(shaderTexture);) {
          auto equalSignIt = std::find(it, end(shaderTexture), '=');
          if (equalSignIt != end(shaderTexture)) {
            decls.emplace_back(it, equalSignIt);
          } else {
            // FIXME: Unexpected...
          }
          it = std::search(++it, end(shaderTexture),cbegin(prefix), cend(prefix));
      }
    }
    for (const auto& decl : decls) {
      auto it = std::search(begin(shaderMain), end(shaderMain),cbegin(decl), cend(decl));
      if (it != end(shaderMain)) {
        auto eolIt = std::find(it, end(shaderMain), '\n');
        if (eolIt != end(shaderMain)) {
          shaderMain.erase(it, ++eolIt);
        }
      }
    }
  }

  std::vector<unsigned char> finalPtx;
  std::string header = "// Generated\n" + version + "\n" + target + "\n" + addressSize + "\n\n";
  finalPtx.insert(end(finalPtx), cbegin(header), cend(header));
  header = "//:FILE: shaderTexture\n\n";
  finalPtx.insert(end(finalPtx), cbegin(header), cend(header));
  finalPtx.insert(end(finalPtx), cbegin(shaderTexture), cend(shaderTexture));
  header = "//:FILE: mdlGen\n\n";
  finalPtx.insert(end(finalPtx), cbegin(header), cend(header));
  finalPtx.insert(end(finalPtx), cbegin(ptxBlob), cend(ptxBlob));
  header = "//:FILE: shaderMain\n\n";
  finalPtx.insert(end(finalPtx), cbegin(header), cend(header));
  finalPtx.insert(end(finalPtx), cbegin(shaderMain), cend(shaderMain));

  // Handle textures
  std::vector<const Sampler*> samplers;
  for (auto i = 1; i < compilationResult->targetCode->get_texture_count(); ++i) {
    auto textureDbName = compilationResult->targetCode->get_texture(i);
    auto textureShape = compilationResult->targetCode->get_texture_shape(i);

    auto texture= mi::base::make_handle(transaction->access<mi::neuraylib::ITexture>(textureDbName));
    auto imageDBName = texture->get_image();
    auto image = mi::base::make_handle(transaction->access<mi::neuraylib::IImage>(imageDBName));
    auto canvas = image->get_canvas(0, 0 ,0);

    switch (textureShape) {
    case mi::neuraylib::ITarget_code::Texture_shape::Texture_shape_2d:
    case mi::neuraylib::ITarget_code::Texture_shape::Texture_shape_3d:
    case mi::neuraylib::ITarget_code::Texture_shape::Texture_shape_bsdf_data: {
        samplers.push_back(prepareTexture(transaction.get(), textureDbName, textureShape));
        break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape::Texture_shape_cube:
      reportMessage(ANARI_SEVERITY_WARNING, "Unsupported sampler of type cube map. Ignoring.");
    default:
      break;
    }
  }

  // MaterialInfo
  MDLMaterialInfo materialInfo(compilationResult->targetCode.get(), samplers);

  // Get a free slot in the material implementations list and get its index.
  auto it = std::find_if(begin(m_materialImplementations),
      end(m_materialImplementations),
      [](const auto &v) { return v.ptxBlob.empty(); });
  if (it == end(m_materialImplementations)) {
    it = m_materialImplementations.insert(it,
        {
            std::move(materialInfo),
            std::move(finalPtx),
        });
  } else {
    *it = {
        std::move(materialInfo),
        std::move(finalPtx), 
    };
  }

  m_uuidToIndex[uuid] = std::distance(begin(m_materialImplementations), it);

  transaction->commit();

  return uuid;
}


std::vector<ptx_blob> MDLCompiler::getPTXBlobs()
{
  std::vector<ptx_blob> blobs;

  for (const auto &materialImplementation : m_materialImplementations) {
    blobs.push_back({
        data(materialImplementation.ptxBlob),
        size(materialImplementation.ptxBlob),
    });
  }

  return blobs;
}

std::vector<MaterialSbtData> MDLCompiler::getMaterialSbtEntries()
{
  if (m_materialImplementations.empty())
    return {};

  std::vector<MaterialSbtData> entries;
  std::vector<uint64_t> offsets;

  std::vector<char> payloads; // Akin to MDLMaterialData, serialized below.

  for (const auto &materialImpl : m_materialImplementations) {
    if (auto argblockData = materialImpl.materialInfo.getArgumentBlockData();
        !argblockData.empty()) {
      offsets.push_back(payloads.size());
      payloads.insert(payloads.end(), cbegin(argblockData), cend(argblockData));
    } else {
      offsets.push_back(-1ul);
    }
  }

  m_argblocksBuffer.upload(payloads.data(), payloads.size());

  const uint8_t *baseAddr = m_argblocksBuffer.ptrAs<const uint8_t>();

  for (auto &offset : offsets) {
    entries.push_back(
        { {offset != -1ul
                 ? reinterpret_cast<const MDLMaterialData *>(baseAddr + offset)
                 : nullptr}});
  }

  return entries;
}

// Prepare the texture identified by the texture_index for use by the texture access functions
// on the GPU.
Sampler* MDLCompiler::prepareTexture(
    mi::neuraylib::ITransaction* transaction,
    char const* texture_db_name,
    mi::neuraylib::ITarget_code::Texture_shape textureShape)
{
    // Get access to the texture data by the texture database name from the target code.
    auto texture = mi::base::make_handle(transaction->access<mi::neuraylib::ITexture>(texture_db_name));
    auto imageApi = getImageApi();

    // FIXME: Needs a texture cache.

    // Access image and canvas via the texture object
    auto image = mi::base::make_handle(transaction->access<mi::neuraylib::IImage>(texture->get_image()));
    auto canvas = mi::base::make_handle(image->get_canvas(0, 0, 0));
    mi::Uint32 tex_width = canvas->get_resolution_x();
    mi::Uint32 tex_height = canvas->get_resolution_y();
    mi::Uint32 tex_layers = canvas->get_layers_size();
    std::string image_type = image->get_type(0, 0);

    if (image->is_uvtile() || image->is_animated()) {
        reportMessage(ANARI_SEVERITY_WARNING, "Unsupported uvtile and/or animated textures.");
        return {};
    }

    // Convert to linear color space if necessary
    if (texture->get_effective_gamma(0, 0) != 1.0f) {
        // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
        mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(imageApi->convert(canvas.get(), "Color"));
        gamma_canvas->set_gamma(texture->get_effective_gamma(0, 0));
        imageApi->adjust_gamma(gamma_canvas.get(), 1.0f);
        canvas = gamma_canvas;
    }

    image_type = canvas->get_type();

    ANARIDataType dataType;
    if (image_type == "Sint8"sv) {
      dataType = ANARI_FIXED8;
    } else if (image_type == "Sint32"sv) {
      dataType = ANARI_FIXED32;
    } else if (image_type == "Float32"sv) {
      dataType = ANARI_FLOAT32;
    } else if (image_type == "Float32<2>"sv) {
      dataType = ANARI_FLOAT32_VEC2;
    } else if (image_type == "Float32<3>"sv) {
      dataType = ANARI_FLOAT32_VEC3;
    } else if (image_type == "Float32<4>"sv || image_type == "Color"sv) {
      dataType = ANARI_FLOAT32_VEC4;
    } else if (image_type == "Rgb"sv) {
      dataType = ANARI_UFIXED8_VEC3;
    } else if (image_type == "Rgba"sv) {
      dataType = ANARI_UFIXED8_VEC4;
    } else if (image_type == "Rgb_16"sv) {
      dataType = ANARI_UFIXED16_VEC3;
    } else if (image_type == "Rgba_16"sv) {
      dataType = ANARI_UFIXED16_VEC4;
    } else if (image_type == "Rgb_fp"sv) {
      dataType = ANARI_FLOAT32_VEC3;
    } else if (image_type == "Color") {
      dataType = ANARI_FLOAT32_VEC3;
    } else { // rgbe, rgbea, 
      reportMessage(ANARI_SEVERITY_WARNING, "Unsupported image type '%s'", image_type.c_str());
      return {};
    }

    Sampler* sampler = {};
    switch (textureShape) {
      case mi::neuraylib::ITarget_code::Texture_shape_2d: {
        Array2DMemoryDescriptor desc = {
          {
            canvas->get_tile(0)->get_data(),
            nullptr,
            nullptr,
            dataType,
          },
         tex_width,
         tex_height,
        };

        auto array2d = new Array2D(m_deviceState, desc);
        array2d->commit();
        auto image2d = new Image2D(m_deviceState);
        image2d->setParam("image", array2d);
        image2d->commit();
        sampler = image2d;
        break;
      }
      case mi::neuraylib::ITarget_code::Texture_shape_3d:
      case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data:
      {
        Array3DMemoryDescriptor desc = {
          {
            canvas->get_tile(0)->get_data(),
            nullptr,
            nullptr,
            dataType,
          },
         tex_width,
         tex_height,
         tex_layers,
        };

        auto array3d = new Array3D(m_deviceState, desc);
        array3d->commit();
        auto image3d = new Image3D(m_deviceState);
        image3d->setParam("image", array3d);
        image3d->commit();
        sampler = image3d;
        break;
      }

      default: {
        reportMessage(ANARI_SEVERITY_WARNING, "Unsupported sampler type %i", int(textureShape));
        break;
      }
    }

    if (!sampler) return nullptr;

    return sampler;
}

void MDLCompiler::releaseModule(Uuid uuid) {
  // FIXME: To implement.
}

} // namespace visrtx
