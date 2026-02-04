// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "Core.h"

#include <fmt/core.h>
#include <fmt/std.h>

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/types.h>
#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/ilogging_configuration.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_compiler.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_distiller_api.h>
#include <mi/neuraylib/imdl_entity_resolver.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iplugin_api.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/iscene_element.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>
#include <mi/neuraylib/iversion.h>

#include <string>

#ifdef MI_PLATFORM_WINDOWS
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
static_assert(sizeof(HMODULE) <= sizeof(void *));

#define loadLibrary(s) reinterpret_cast<void *>(LoadLibrary(s))
#define freeLibrary(l) FreeLibrary(reinterpret_cast<HMODULE>(l))
#define getProcAddress(l, s) GetProcAddress(reinterpret_cast<HMODULE>(l), s)

#else
#include <dlfcn.h>

#define loadLibrary(s) dlopen(s, RTLD_LAZY)
#define freeLibrary(l) dlclose(l)
#define getProcAddress(l, s) dlsym(l, s)

#endif

#include <nonstd/scope.hpp>

#include <stdexcept>

using namespace std::string_literals;
using mi::base::make_handle;

namespace visrtx::libmdl {

Core::Core() : Core({}, {}) {}

Core::Core(mi::base::ILogger *logger) : Core({}, logger) {}

Core::Core(mi::neuraylib::INeuray *neuray) : Core(neuray, {}) {}

Core::Core(mi::neuraylib::INeuray *neuray, mi::base::ILogger *logger)
{
  if (neuray && logger) {
    throw std::runtime_error(
        "Only one of neuray or logger can be provided to libmdl::Core");
  }

  static constexpr const auto filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;

  nonstd::scope_fail handleCleanup([this]() {
    m_executionContext = {};
    m_globalScope = {};
    m_neuray = {};

    if (m_dllHandle)
      freeLibrary(m_dllHandle);
    m_dllHandle = {};
  });

  if (neuray) {
    m_neuray = mi::base::make_handle_dup(neuray);
    m_dllHandle = {};
  } else {
    // Load library
    m_dllHandle = loadLibrary(filename);

    if (m_dllHandle == nullptr)
      throw std::runtime_error("Failed to load MDL SDK library "s + filename);

    // Get neuray main entry point
    void *symbol = getProcAddress(m_dllHandle, "mi_factory");
    if (symbol == nullptr)
      throw std::runtime_error("Failed to find MDL SDK mi_factory symbol");

    m_neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
    if (m_neuray == nullptr) {
      // Check if we have a valid neuray instance, otherwise check why.
      auto version = make_handle(
          mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
      if (!version) {
        throw std::runtime_error("Cannot get MDL SDK library version");
      } else {
        throw std::runtime_error(
            "Cannot get INeuray interface from mi_factory, either there is a version mismatch or the interface has already been acquired: "s
            "Expected version is " MI_NEURAYLIB_PRODUCT_VERSION_STRING
            ", library version is "
            + version->get_product_version());
      }
    }

    // Get the MDL configuration component so main path can be added.
    auto mdlConfiguration = make_handle(
        m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mdlConfiguration->add_mdl_system_paths();
    mdlConfiguration->add_mdl_user_paths();

    auto loggingConfig = make_handle(
        m_neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
    if (logger) {
      loggingConfig->set_receiving_logger(logger);
    } else {
      logger = loggingConfig->get_receiving_logger();
    }

    m_logger = logger;

    auto pluginConf = make_handle(
        m_neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
    if (mi::Sint32 res = pluginConf->load_plugin_library(
            "nv_openimageio" MI_BASE_DLL_FILE_EXT);
        res != 0) {
      logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "Failed to load the nv_openimageio plugin");
    }
    if (mi::Sint32 res =
            pluginConf->load_plugin_library("dds" MI_BASE_DLL_FILE_EXT);
        res != 0) {
      logMessage(
          mi::base::MESSAGE_SEVERITY_WARNING, "Failed to load the dds plugin");
    }

    if (mi::Sint32 res = pluginConf->load_plugin_library(
            "mdl_distiller" MI_BASE_DLL_FILE_EXT);
        res != 0) {
      logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "Failed to load the mdl_distiller plugin");
    }

    m_neuray->start();
  }

  // Get the global scope from the database
  auto database =
      make_handle(m_neuray->get_api_component<mi::neuraylib::IDatabase>());
  if (!database.is_valid_interface())
    throw std::runtime_error("Failed to retrieve neuray database component");

  m_globalScope = make_handle(database->get_global_scope());
  if (!m_globalScope.is_valid_interface())
    throw std::runtime_error("Failed to acquire neuray database global scope");

  // Get an execution context for later use.
  m_mdlFactory =
      make_handle(m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());
  if (!m_mdlFactory.is_valid_interface()) {
    throw std::runtime_error("Failed to retrieve MDL factory component");
  }

  m_executionContext = make_handle(m_mdlFactory->create_execution_context());
  if (!m_executionContext.is_valid_interface()) {
    throw std::runtime_error("Failed acquiring an execution context");
  }

  // Some default options that other cloned contexts will use.
  // We will load resources ourselves. Let's save us from autoloading things
  // that might not be used.
  m_executionContext->set_option("resolve_resources", false);
}

Core::~Core()
{
  m_executionContext = {};
  m_mdlFactory = {};
  m_globalScope = {};
  if (m_dllHandle) {
    m_neuray->shutdown();
    m_neuray = {};
    freeLibrary(m_dllHandle);
  }
  m_dllHandle = {};
}

mi::neuraylib::IScope *Core::createScope(
    std::string_view scopeName, mi::neuraylib::IScope *parent)
{
  auto database =
      make_handle(m_neuray->get_api_component<mi::neuraylib::IDatabase>());

  return database->create_scope(parent);
}

void Core::removeScope(mi::neuraylib::IScope *scope)
{
  auto database =
      make_handle(m_neuray->get_api_component<mi::neuraylib::IDatabase>());

  database->remove_scope(scope->get_id());
}

mi::neuraylib::ITransaction *Core::createTransaction(
    mi::neuraylib::IScope *scope)
{
  if (!scope)
    scope = m_globalScope.get();
  return scope->create_transaction();
}

void Core::addBuiltinModule(
    std::string_view moduleName, std::string_view moduleSource)
{
  auto impexpApi = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

  auto executionContext =
      make_handle(m_mdlFactory->clone(m_executionContext.get()));

  auto transaction = make_handle(createTransaction());
  nonstd::scope_exit finalizeTransaction(
      [transaction]() { transaction->commit(); });

  auto result = impexpApi->load_module_from_string(transaction.get(),
      std::string(moduleName).c_str(),
      std::string(moduleSource).c_str(),
      executionContext.get());

  switch (result) {
  case 0: {
    logMessage(mi::base::MESSAGE_SEVERITY_INFO,
        "Added builtin module {} from source",
        moduleName);
    break;
  }
  case 1: {
    logMessage(mi::base::MESSAGE_SEVERITY_INFO,
        "Builtin module {} already exists",
        moduleName);
    break;
  }
  case -1:
    logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Invalid name {} or module source for builtin",
        moduleName);
    break;
  case -2:
    logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
        "Ignoring builtin {} would shadow a file based definition",
        moduleName);
    break;
  default:
    logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unknown error while adding builtin module {}",
        moduleName);
    logExecutionContextMessages(executionContext.get());
    break;
  }
}

const mi::neuraylib::IModule *Core::loadModule(
    std::string_view moduleOrFileName, mi::neuraylib::ITransaction *transaction)
{
  auto impexpApi = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

  auto moduleName = std::string(moduleOrFileName);

  // If that fails, try and resolve it as a file name.
  // First considering  the module name from the MDL file name.
  if (auto name =
          make_handle(impexpApi->get_mdl_module_name(moduleName.c_str()));
      name.is_valid_interface()) {
    moduleName = name->get_c_str();
  } else {
    // Check if this is a single MDL name, such as OmniPBR.mdl and
    // resolve it to its equivalent module name, such as ::OmniPBR.
    if (auto len = moduleName.length(); len > 4) {
      auto extension = moduleName.substr(len - 4);
      if (moduleName.find('/') == std::string::npos && extension == ".mdl") {
        moduleName = "::"s + moduleName.substr(0, len - 4);
      }
    } else {
      moduleName.clear();
    }
  }

  if (moduleName.empty()) {
    logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot resolve module name from {}",
        std::string(moduleOrFileName));
    return {};
  }

  // Clone the context so we can go and at least have message isolation.
  auto executionContext =
      make_handle(m_mdlFactory->clone(m_executionContext.get()));

  if (impexpApi->load_module(
          transaction, moduleName.c_str(), executionContext.get())
      < 0)
    return {};

  // Get the database name for the module we loaded
  auto moduleDbName = make_handle(
      m_mdlFactory->get_db_module_name(std::string(moduleName).c_str()));
  return transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str());
}

const mi::neuraylib::IFunction_definition *Core::getFunctionDefinition(
    const mi::neuraylib::IModule *module,
    std::string_view functionName,
    mi::neuraylib::ITransaction *transaction)
{
  std::string functionQualifiedName;
  if (functionName.back() == ')') {
    // Already a qualified function signature. Make sure it includes the
    // namespacing.
    functionQualifiedName = functionName;
    if (functionQualifiedName.front() != ':') {
      functionQualifiedName =
          "mdl"s + module->get_mdl_name() + "::" + functionQualifiedName;
    }
  } else { // Needs more work to get what we need.
    functionQualifiedName = functionName;
    auto overloads = make_handle(
        module->get_function_overloads(functionQualifiedName.c_str()));
    if (!overloads.is_valid_interface() || overloads->get_length() != 1)
      return {};

    auto theOneOverload = make_handle(
        overloads->get_element<mi::IString>(static_cast<mi::Size>(0)));
    functionQualifiedName = theOneOverload->get_c_str();
    logMessage(mi::base::MESSAGE_SEVERITY_INFO,
        "Deducing fully qualified name {} from provided {}",
        functionQualifiedName,
        functionName);
  }

  return transaction->access<mi::neuraylib::IFunction_definition>(
      functionQualifiedName.c_str());
}

mi::neuraylib::ICompiled_material *Core::getCompiledMaterial(
    const mi::neuraylib::IFunction_definition *functionDefinition,
    bool classCompilation)
{
  mi::Sint32 ret = 0;
  auto functionCall =
      make_handle(functionDefinition->create_function_call(0, &ret));
  if (ret != 0)
    return {};

  auto executionContext =
      make_handle(m_mdlFactory->clone(m_executionContext.get()));

  auto materialInstance = make_handle(
      functionCall->get_interface<mi::neuraylib::IMaterial_instance>());
  auto compiledMaterial = materialInstance->create_compiled_material(
      classCompilation ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
                       : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS,
      executionContext.get());

  if (!logExecutionContextMessages(executionContext.get())) {
    return {};
  }

  return compiledMaterial;
}

mi::neuraylib::ICompiled_material *Core::getDistilledToDiffuse(
    const mi::neuraylib::ICompiled_material *compiledMaterial)
{
  auto distiller_api = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
  mi::Sint32 result = 0;
  auto distilledMaterial = distiller_api->distill_material(
      compiledMaterial, "diffuse", nullptr, &result);
  if (result != 0) {
    logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Failed to distill material: %i\n",
        result);
  }

  return distilledMaterial;
}

const mi::neuraylib::ITarget_code *Core::getPtxTargetCode(
    const mi::neuraylib::ICompiled_material *compiledMaterial,
    mi::neuraylib::ITransaction *transaction)
{
  auto backendApi = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

  auto ptxBackend = make_handle(
      backendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX));
  auto executionContext =
      make_handle(m_mdlFactory->clone(m_executionContext.get()));

  auto distilledMaterial = make_handle(getDistilledToDiffuse(compiledMaterial));

  // ANARI attributes 0 to 3
  const int numTextureSpaces = 4;
  // Number of actually supported textures. MDL's default, let's assume this is
  // enough for now
  const int numTextureResults = 32;

  ptxBackend->set_option(
      "num_texture_spaces", std::to_string(numTextureSpaces).c_str());
  ptxBackend->set_option(
      "num_texture_results", std::to_string(numTextureResults).c_str());
  ptxBackend->set_option_binary("llvm_renderer_module", nullptr, 0);
  ptxBackend->set_option("visible_functions", "");

  ptxBackend->set_option("sm_version", "52");
  ptxBackend->set_option("tex_lookup_call_mode", "direct_call");
  ptxBackend->set_option("lambda_return_mode", "value");
  ptxBackend->set_option("texture_runtime_with_derivs", "off");
  ptxBackend->set_option("inline_aggressively", "on");
  ptxBackend->set_option("opt_level", "2");
  ptxBackend->set_option("enable_exceptions", "off");

  // For now, only consider surface scattering.
  static mi::neuraylib::Target_function_description materialFunctions[] = {
      {"init", "mdlInit"},
      {"thin_walled", "mdlThinWalled"},

      {"surface.scattering", "mdlBsdf"},
      {"surface.emission.emission", "mdlEmission"},
      {"surface.emission.intensity", "mdlEmissionIntensity"},
      {"surface.emission.mode", "mdlEmissionMode"},

      {"volume.scattering_coefficient", "mdlTransmission"},

      {"geometry.cutout_opacity", "mdlOpacity"},
  };

  static mi::neuraylib::Target_function_description distilledFunctions[] = {
      {"surface.scattering.tint", "mdlTint"},
  };

  // Generate target code for the compiled material
  auto linkUnit = make_handle(
      ptxBackend->create_link_unit(transaction, executionContext.get()));

  // Add main material functions (BSDF, emission, and auxiliary
  // albedo/normal/roughness)
  linkUnit->add_material(compiledMaterial,
      std::data(materialFunctions),
      std::size(materialFunctions),
      executionContext.get());

  if (!logExecutionContextMessages(executionContext.get()))
    return {};

  linkUnit->add_material(distilledMaterial.get(),
      std::data(distilledFunctions),
      std::size(distilledFunctions),
      executionContext.get());

  if (!logExecutionContextMessages(executionContext.get()))
    return {};

  auto targetCode =
      ptxBackend->translate_link_unit(linkUnit.get(), executionContext.get());
  if (!logExecutionContextMessages(executionContext.get()))
    return {};

  return targetCode;
}

bool Core::logExecutionContextMessages(
    const mi::neuraylib::IMdl_execution_context *executionContext)
{
  for (auto i = 0ull, messageCount = executionContext->get_messages_count();
      i < messageCount;
      ++i) {
    auto message = make_handle(executionContext->get_message(i));
    logMessage(message->get_severity(), "{}", message->get_string());
  }

  for (auto i = 0ull,
            messageCount = executionContext->get_error_messages_count();
      i < messageCount;
      ++i) {
    auto message = make_handle(executionContext->get_error_message(i));
    logMessage(message->get_severity(), "{}", message->get_string());
  }

  return executionContext->get_error_messages_count() == 0;
}

auto Core::getINeuray() const -> mi::neuraylib::INeuray *
{
  return m_neuray.get();
}
auto Core::getMdlFactory() const -> mi::neuraylib::IMdl_factory *
{
  return m_mdlFactory.get();
}

auto Core::getLogger() const -> mi::base::ILogger *
{
  return m_logger.get();
}

auto Core::setMdlSearchPaths(nonstd::span<std::filesystem::path> paths) -> void
{
  auto mdlConfiguration = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
  mdlConfiguration->clear_mdl_paths();
  for (const auto &path : paths) {
    mdlConfiguration->add_mdl_path(path.generic_string().c_str());
  }
  mdlConfiguration->add_mdl_system_paths();
  mdlConfiguration->add_mdl_user_paths();
}

auto Core::setMdlResourceSearchPaths(nonstd::span<std::filesystem::path> paths)
    -> void
{
  auto mdlConfiguration = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
  mdlConfiguration->clear_resource_paths();
  for (const auto &path : paths) {
    mdlConfiguration->add_resource_path(path.string().c_str());
  }
}

auto Core::resolveResource(
    std::string_view resourceId, std::string_view ownerId) -> std::string
{
  auto mdlConfiguration = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
  auto entityResolver = make_handle(mdlConfiguration->get_entity_resolver());
  auto resolvedResource = make_handle(
      entityResolver->resolve_resource(std::string(resourceId).c_str(),
          ownerId.empty() ? nullptr : std::string(ownerId).c_str(),
          nullptr,
          0,
          0));

  if (resolvedResource.is_valid_interface()) {
    auto firstResolvedResourceElement =
        make_handle(resolvedResource->get_element(0));
    if (firstResolvedResourceElement.is_valid_interface()) {
      auto res = firstResolvedResourceElement->get_filename(0);
      return res ? std::string(res) : std::string();
    }
  }

  return {};
}

auto Core::resolveModule(std::string_view moduleId) -> std::string
{
  auto mdlConfiguration = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
  auto entityResolver = make_handle(mdlConfiguration->get_entity_resolver());

  auto executionContext =
      make_handle(m_mdlFactory->clone(m_executionContext.get()));
  auto resolvedModule =
      make_handle(entityResolver->resolve_module(std::string(moduleId).c_str(),
          nullptr,
          nullptr,
          0,
          0,
          executionContext.get()));
  logExecutionContextMessages(executionContext.get());

  if (resolvedModule.is_valid_interface()) {
    return resolvedModule->get_module_name();
  } else {
    logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
        "Failed to resolve module `{}` using entityResolver\n",
        moduleId);
  }

  return {};
}

} // namespace visrtx::libmdl
