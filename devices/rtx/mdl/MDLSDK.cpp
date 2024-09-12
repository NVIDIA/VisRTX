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

#include "MDLSDK.h"

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
#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#endif

#ifdef MI_PLATFORM_WINDOWS
#include <direct.h>
#else
#include <dlfcn.h>
#endif

#include <optional>

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

} // namespace

namespace visrtx {

class MDLSDK::Logger : public mi::base::Interface_implement<mi::base::ILogger>
{
  MDLSDK *m_sdk;

 public:
  Logger(MDLSDK *sdk) : m_sdk(sdk) {}
  virtual void message(mi::base::Message_severity level,
      const char *moduleCategory,
      const mi::base::Message_details &,
      const char *message)
  {
    m_sdk->reportMessage(
        miSeverityToAnari(level), "%s %s", moduleCategory, message);
  }
};

inline mi::Sint32 MDLSDK::loadPlugin(
    mi::neuraylib::INeuray *neuray, const char *path)
{
  mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
      neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

  // try to load the requested plugin before adding any special handling
  mi::Sint32 res = plugin_conf->load_plugin_library(path);
  if (res == 0) {
    reportMessage(ANARI_SEVERITY_INFO,
        "Successfully loaded the plugin library '%s'",
        path);
    return 0;
  }

  // return the failure code
  reportMessage(
      ANARI_SEVERITY_ERROR, "Failed to load the plugin library '%s'", path);
  return res;
}

bool MDLSDK::parseCmdArgumentMaterialName(const std::string &argument,
    std::string &out_module_name,
    std::string &out_material_name,
    bool prepend_colons_if_missing)
{
  out_module_name = "";
  out_material_name = "";
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
      out_module_name = potential_path;
      out_material_name = potential_material_name;
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
    out_module_name = "::";
  }

  out_module_name.append(argument.substr(0, p_last));
  out_material_name = argument.substr(p_last + 2, argument.size() - p_last);
  return true;
}

MDLSDK::MDLSDK(VisRTXDevice *device) : m_device(device)
{
  m_neuray = loadAndGetINeuray();
  if (!m_neuray.is_valid_interface())
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to load the MDL SDK. MDL support will be disabled.");

  mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_config(
      m_neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
  mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

  // set user defined or default logger
  logging_config->set_receiving_logger(
      mi::base::make_handle(new Logger(this)).get());

  m_logger = logging_config->get_forwarding_logger();

  mdl_config->add_mdl_system_paths();
  mdl_config->add_mdl_user_paths();

  if (loadPlugin(m_neuray.get(), "nv_openimageio" MI_BASE_DLL_FILE_EXT) != 0) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to load the nv_openimageio plugin. Might create some issue with image loading.");
  }

  if (loadPlugin(m_neuray.get(), "dds" MI_BASE_DLL_FILE_EXT) != 0) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to load the dds plugin. Might create some issue with image loading.");
  }

  // Start the MDL SDK
  mi::Sint32 ret = m_neuray->start();
  if (ret != 0)
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to initialize the SDK. Result code: %d",
        ret);

  // Setup the compiler
  m_mdlCompiler = m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>();
  if (!m_mdlCompiler)
    reportMessage(
        ANARI_SEVERITY_ERROR, "Initialization of MDL compiler failed!");

  m_mdlConfiguration =
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>();
  if (!m_mdlConfiguration)
    reportMessage(ANARI_SEVERITY_ERROR, "Retrieving MDL configuration failed!");

  m_database = m_neuray->get_api_component<mi::neuraylib::IDatabase>();
  m_globalScope = m_database->get_global_scope();

  m_mdlFactory = m_neuray->get_api_component<mi::neuraylib::IMdl_factory>();
  m_executionContext = m_mdlFactory->create_execution_context();

#if 0
  m_executionContext->set_option("internal_space", "coordinate_world");  // equals default
  m_executionContext->set_option("bundle_resources", false);             // equals default
  m_executionContext->set_option("meters_per_scene_unit", 1.0f);         // equals default
  m_executionContext->set_option("mdl_wavelength_min", 380.0f);          // equals default
  m_executionContext->set_option("mdl_wavelength_max", 780.0f);          // equals default
  m_executionContext->set_option("include_geometry_normal", true);       // equals default
#endif

  // FIXME To handle
  const int numTextureSpaces = 2;
  const int numTextureResults = 2;
  const bool enable_derivatives = false;

  mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
      m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
  m_backendCudaPtx = mdl_backend_api->get_backend(
      mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX);
  if (m_backendCudaPtx->set_option(
          "num_texture_spaces", std::to_string(numTextureSpaces).c_str())
      != 0)
    reportMessage(
        ANARI_SEVERITY_ERROR, "Setting PTX option num_texture_spaces failed");
  if (m_backendCudaPtx->set_option(
          "num_texture_results", std::to_string(numTextureResults).c_str())
      != 0)
    reportMessage(
        ANARI_SEVERITY_ERROR, "Setting PTX option num_texture_results failed");
  if (m_backendCudaPtx->set_option("sm_version", "75") != 0)
    reportMessage(ANARI_SEVERITY_ERROR, "Setting PTX option sm_version failed");
  if (m_backendCudaPtx->set_option("tex_lookup_call_mode", "direct_call") != 0)
    reportMessage(
        ANARI_SEVERITY_ERROR, "Setting PTX option tex_lookup_call_mode failed");
  if (m_backendCudaPtx->set_option("lambda_return_mode", "value") != 0)
    reportMessage(
        ANARI_SEVERITY_ERROR, "Setting PTX option lambda_return_mode failed");
  if (enable_derivatives) {
    if (m_backendCudaPtx->set_option("texture_runtime_with_derivs", "on") != 0)
      reportMessage(ANARI_SEVERITY_ERROR,
          "Setting PTX option texture_runtime_with_derivs failed");
  }
  if (m_backendCudaPtx->set_option("inline_aggressively", "off") != 0)
    reportMessage(
        ANARI_SEVERITY_ERROR, "Setting PTX option inline_aggressively failed");

  if (m_backendCudaPtx->set_option("opt_level", "0") != 0)
    reportMessage(ANARI_SEVERITY_ERROR, "Setting PTX option opt_level failed");

  if (m_backendCudaPtx->set_option("enable_exceptions", "off"))
    reportMessage(
        ANARI_SEVERITY_ERROR, "Setting PTX option enable_exceptions failed");

  m_imageApi = m_neuray->get_api_component<mi::neuraylib::IImage_api>();
}

MDLSDK::~MDLSDK()
{
  m_targetCodeCache.clear();

  m_imageApi.reset();
  m_backendCudaPtx.reset();
  m_executionContext.reset();
  m_mdlFactory.reset();
  m_globalScope.reset();
  m_database.reset();

  m_mdlConfiguration.reset();
  m_mdlCompiler.reset();

  m_logger.reset();
  // Shut down the MDL SDK
  if (m_neuray->shutdown() != 0)
    reportMessage(ANARI_SEVERITY_ERROR, "Failed to shutdown the SDK.");

  m_neuray.reset();

  // Unload the MDL SDK
  if (!unloadINeuray())
    reportMessage(ANARI_SEVERITY_ERROR, "Failed to unload the SDK.");
}

// Returns a string-representation of the given message severity
inline const char *message_severity_to_string(
    mi::base::Message_severity severity)
{
  switch (severity) {
  case mi::base::MESSAGE_SEVERITY_ERROR:
    return "error";
  case mi::base::MESSAGE_SEVERITY_WARNING:
    return "warning";
  case mi::base::MESSAGE_SEVERITY_INFO:
    return "info";
  case mi::base::MESSAGE_SEVERITY_VERBOSE:
    return "verbose";
  case mi::base::MESSAGE_SEVERITY_DEBUG:
    return "debug";
  default:
    break;
  }
  return "";
}

// Returns a string-representation of the given message category
inline const char *message_kind_to_string(
    mi::neuraylib::IMessage::Kind message_kind)
{
  switch (message_kind) {
  case mi::neuraylib::IMessage::MSG_INTEGRATION:
    return "MDL SDK";
  case mi::neuraylib::IMessage::MSG_IMP_EXP:
    return "Importer/Exporter";
  case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
    return "Compiler Backend";
  case mi::neuraylib::IMessage::MSG_COMILER_CORE:
    return "Compiler Core";
  case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
    return "Compiler Archive Tool";
  case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
    return "Compiler DAG generator";
  default:
    break;
  }
  return "";
}

// Prints the messages of the given context.
// Returns true, if the context does not contain any error messages, false
// otherwise.
bool MDLSDK::logMessages(mi::neuraylib::IMdl_execution_context *context)
{
  for (mi::Size i = 0; i < context->get_messages_count(); ++i) {
    mi::base::Handle<const mi::neuraylib::IMessage> message(
        context->get_message(i));
    reportMessage(miSeverityToAnari(message->get_severity()),
        "%s: %s",
        message_kind_to_string(message->get_kind()),
        message->get_string());
  }
  return context->get_error_messages_count() == 0;
}

inline std::string addMissingMaterialSignature(
    const mi::neuraylib::IModule *module, const std::string &material_name)
{
  // Return input if it already contains a signature.
  if (material_name.back() == ')')
    return material_name;

  mi::base::Handle<const mi::IArray> result(
      module->get_function_overloads(material_name.c_str()));
  if (!result || result->get_length() != 1)
    return std::string();

  mi::base::Handle<const mi::IString> overloads(
      result->get_element<mi::IString>(static_cast<mi::Size>(0)));
  return overloads->get_c_str();
}

std::optional<MDLSDK::CompilationResult> MDLSDK::compileMaterial(
    mi::neuraylib::ITransaction *transaction,
    std::string const &materialName,
    std::vector<mi::neuraylib::Target_function_description> &descs,
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
  mdlImpexpApi->load_module(
      transaction, moduleName.c_str(), m_executionContext.get());
  if (!logMessages(m_executionContext.get()))
    return {};

  // Get the database name for the module we loaded
  mi::base::Handle<const mi::IString> moduleDbName(
      m_mdlFactory->get_db_module_name(moduleName.c_str()));
  mi::base::Handle<const mi::neuraylib::IModule> module(
      transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str()));
  if (!module) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to access the loaded module %s",
        moduleName.c_str());
    return {};
  }

  // FIXME: Check for iModule->get_material(0) and see what names it returns
  // FIXME: Use the argument editor to get a description of the material

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
      flags, m_executionContext.get());

  if (!logMessages(m_executionContext.get()))
    return {};

  mi::base::Handle<mi::neuraylib::ITarget_code const> targetCode;
  // Reuse old target code, if possible
  mi::base::Uuid materialHash = compiledMaterial->get_hash();
  auto it = m_targetCodeCache.find(materialHash);
  if (it != m_targetCodeCache.end()) {
    targetCode = it->second;
  } else {
    // Generate target code for the compiled material
    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(
        m_backendCudaPtx->create_link_unit(
            transaction, m_executionContext.get()));
    linkUnit->add_material(compiledMaterial.get(),
        descs.data(),
        descs.size(),
        m_executionContext.get());

    if (!logMessages(m_executionContext.get()))
      return {};

    targetCode = mi::base::Handle<const mi::neuraylib::ITarget_code>(
        m_backendCudaPtx->translate_link_unit(
            linkUnit.get(), m_executionContext.get()));
    if (!logMessages(m_executionContext.get())) {
      return {};
    }

    m_targetCodeCache[materialHash] = targetCode;

    transaction->store(functionCall.get(), materialDbName.c_str());
  }

  auto argblock = targetCode->get_argument_block_count() == 1
      ? targetCode->get_argument_block(0)
      : nullptr;

  return CompilationResult{
      .uuid = materialHash,
      .targetCode = targetCode,
  };
}

void MDLSDK::addMdlSearchPath(const std::filesystem::path &path)
{
  m_mdlConfiguration->add_mdl_path(path.u8string().c_str());
}

void MDLSDK::removeMdlSearchPath(const std::filesystem::path &path)
{
  m_mdlConfiguration->remove_mdl_path(path.u8string().c_str());
}

mi::neuraylib::INeuray *MDLSDK::loadAndGetINeuray(const char *filename)
{
  if (!filename)
    filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;

#ifdef MI_PLATFORM_WINDOWS
  HMODULE handle = LoadLibraryA(filename);
  if (!handle) {
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
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed to load %s library (%u): " FMT_LPTSTR,
        filename,
        error_code,
        message);
    if (buffer)
      LocalFree(buffer);
    return 0;
  }
  void *symbol = GetProcAddress(handle, "mi_factory");
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
    reportMessage(ANARI_SEVERITY_ERROR,
        "GetProcAddress error (%u): " FMT_LPTSTR,
        error_code,
        message);
    if (buffer)
      LocalFree(buffer);
    return nullptr;
  }
#else // MI_PLATFORM_WINDOWS
  void *handle = dlopen(filename, RTLD_LAZY);
  if (!handle) {
    reportMessage(
        ANARI_SEVERITY_ERROR, "Failed load the MDL SDK library: %s", dlerror());
    return nullptr;
  }
  void *symbol = dlsym(handle, "mi_factory");
  if (!symbol) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed getting mi_factory from the MDL SDK library: %s",
        dlerror());
    return nullptr;
  }
#endif // MI_PLATFORM_WINDOWS
  m_dso_handle = handle;

  mi::neuraylib::INeuray *neuray =
      mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
  if (!neuray) {
    mi::base::Handle<mi::neuraylib::IVersion> version(
        mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
    if (!version)
      reportMessage(ANARI_SEVERITY_ERROR, "Error: Incompatible library.\n");
    else
      reportMessage(ANARI_SEVERITY_ERROR,
          "Error: Library version %s does not match header version %s.\n",
          version->get_product_version(),
          MI_NEURAYLIB_PRODUCT_VERSION_STRING);
    return nullptr;
  }

  return neuray;
}

bool MDLSDK::unloadINeuray()
{
#ifdef MI_PLATFORM_WINDOWS
  BOOL result = FreeLibrary(g_dso_handle);
  if (!result) {
    reportMessage(ANARI_SEVERITY_ERROR, "Failed unloading the MDL SDK library");
    return false;
  }
  return true;
#else
  int result = dlclose(m_dso_handle);
  if (result != 0) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "Failed unloading the MDL SDK library: %s",
        dlerror());
    return false;
  }
  return true;
#endif
}

} // namespace visrtx
