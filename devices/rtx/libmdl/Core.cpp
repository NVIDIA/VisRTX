// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "Core.h"
#include "MDLBackendConfig.h"

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
#include <mi/neuraylib/iexpression.h>
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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

#ifdef MI_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
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

mi::Sint32 Core::loadModuleSource(std::string_view moduleName,
    std::string_view moduleSource,
    mi::neuraylib::ITransaction *transaction)
{
  auto impexpApi = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
  auto executionContext =
      make_handle(m_mdlFactory->clone(m_executionContext.get()));

  auto result = impexpApi->load_module_from_string(transaction,
      std::string(moduleName).c_str(),
      std::string(moduleSource).c_str(),
      executionContext.get());

  if (result < 0)
    logExecutionContextMessages(executionContext.get());

  return result;
}

const mi::neuraylib::IModule *Core::loadModuleFromString(
    std::string_view moduleName,
    std::string_view moduleSource,
    mi::neuraylib::ITransaction *transaction)
{
  if (loadModuleSource(moduleName, moduleSource, transaction) < 0)
    return nullptr;
  return accessModule(moduleName, transaction);
}

const mi::neuraylib::IModule *Core::accessModule(
    std::string_view moduleName, mi::neuraylib::ITransaction *transaction)
{
  auto dbName = make_handle(
      m_mdlFactory->get_db_module_name(std::string(moduleName).c_str()));
  return transaction->access<mi::neuraylib::IModule>(dbName->get_c_str());
}

void Core::addBuiltinModule(
    std::string_view moduleName, std::string_view moduleSource)
{
  auto transaction = make_handle(createTransaction());
  nonstd::scope_exit finalizeTransaction(
      [transaction]() { transaction->commit(); });

  auto result = loadModuleSource(moduleName, moduleSource, transaction.get());

  switch (result) {
  case 0:
    logMessage(mi::base::MESSAGE_SEVERITY_INFO,
        "Added builtin module {} from source",
        moduleName);
    break;
  case 1:
    logMessage(mi::base::MESSAGE_SEVERITY_INFO,
        "Builtin module {} already exists",
        moduleName);
    break;
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
      < 0) {
    // A scene may ship a .mdl without its resources (e.g. a vMaterials module
    // copied without its ./textures), so the local copy fails to compile. A
    // complete copy usually exists on the MDL search path -- recover it by
    // canonical name before giving up.
    return loadModuleByCanonicalName(moduleOrFileName, transaction);
  }

  // Get the database name for the module we loaded
  auto moduleDbName = make_handle(
      m_mdlFactory->get_db_module_name(std::string(moduleName).c_str()));
  return transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str());
}

const mi::neuraylib::IModule *Core::loadModuleByCanonicalName(
    std::string_view filePath, mi::neuraylib::ITransaction *transaction)
{
  std::string path(filePath);
  if (path.find('/') == std::string::npos)
    return {}; // already a module name, nothing to recover

  auto impexpApi = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
  auto mdlConfiguration = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

  // If the failing path passes through a directory whose basename matches a
  // search root (e.g. ".../vMaterials_2/Concrete/Concrete_Precast.mdl" with a
  // "/data/mdl/vMaterials_2" root), the canonical module name is the tail after
  // it ("::Concrete::Concrete_Precast"). Loading that by name lets the entity
  // resolver pick a complete copy on the search path.
  auto pathsCount = mdlConfiguration->get_mdl_paths_length();
  for (auto i = decltype(pathsCount)(0); i < pathsCount; ++i) {
    auto rootName =
        std::filesystem::path(
            make_handle(mdlConfiguration->get_mdl_path(i))->get_c_str())
            .filename()
            .string();
    if (rootName.empty())
      continue;

    auto marker = "/" + rootName + "/";
    auto pos = path.rfind(marker);
    if (pos == std::string::npos)
      continue;

    auto tail = path.substr(pos + marker.size());
    if (auto n = tail.size(); n > 4 && tail.substr(n - 4) == ".mdl")
      tail = tail.substr(0, n - 4);
    if (tail.empty())
      continue;

    std::string moduleName = "::";
    for (char c : tail)
      moduleName += (c == '/') ? "::"s : std::string(1, c);

    auto executionContext =
        make_handle(m_mdlFactory->clone(m_executionContext.get()));
    if (impexpApi->load_module(
            transaction, moduleName.c_str(), executionContext.get())
        < 0)
      continue;

    auto moduleDbName =
        make_handle(m_mdlFactory->get_db_module_name(moduleName.c_str()));
    if (auto *module = transaction->access<mi::neuraylib::IModule>(
            moduleDbName->get_c_str())) {
      logMessage(mi::base::MESSAGE_SEVERITY_INFO,
          "Recovered '{}' from the MDL search path as '{}'",
          path,
          moduleName);
      return module;
    }
  }

  return {};
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

namespace {

// Resolve `let`-block indirection: a compiled sub-expression may reference a
// temporary slot instead of holding the value; without this, literal-bodied
// emitters would silently classify as non-constant.
mi::base::Handle<const mi::neuraylib::IExpression> derefTemporaries(
    const mi::neuraylib::ICompiled_material *compiledMaterial,
    mi::base::Handle<const mi::neuraylib::IExpression> expr)
{
  using namespace mi::neuraylib;
  using mi::base::make_handle;
  while (expr && expr->get_kind() == IExpression::EK_TEMPORARY) {
    auto temporary =
        make_handle(expr->get_interface<const IExpression_temporary>());
    expr = make_handle(compiledMaterial->get_temporary(temporary->get_index()));
  }
  return expr;
}

// Multiply a constant color/float factor into `scale` componentwise.
bool foldColorFactor(
    const mi::neuraylib::IValue *value, std::array<float, 3> &scale)
{
  using namespace mi::neuraylib;
  using mi::base::make_handle;
  if (!value)
    return false;
  if (value->get_kind() == IValue::VK_COLOR) {
    auto color = make_handle(value->get_interface<const IValue_color>());
    for (int i = 0; i < 3; ++i) {
      auto channel = make_handle(color->get_value(i));
      if (!channel)
        return false;
      auto f = make_handle(channel->get_interface<const IValue_float>());
      if (!f)
        return false;
      scale[i] *= f->get_value();
    }
    return true;
  }
  if (value->get_kind() == IValue::VK_FLOAT) {
    const float f =
        make_handle(value->get_interface<const IValue_float>())->get_value();
    for (auto &c : scale)
      c *= f;
    return true;
  }
  return false;
}

struct DynamicIntensity
{
  std::array<float, 3> scale{1.f, 1.f, 1.f};
  Core::EmissionClassification::DynamicSource source =
      Core::EmissionClassification::DynamicSource::None;
  std::string argumentName;
};

// Multiplicative walk of a non-folding intensity expression: accumulate
// constant factors into `scale` and identify at most ONE dynamic factor — a
// color/float parameter or the `tex` of a tex::lookup_color. Anything outside
// that shape (sums, other calls, two dynamic factors, a multiply chain deeper
// than any sane authoring) fails the walk and the material keeps the
// unit-proxy Pick Power.
constexpr int kMaxIntensityWalkDepth = 16;

bool walkIntensityFactors(
    const mi::neuraylib::ICompiled_material *compiledMaterial,
    mi::base::Handle<const mi::neuraylib::IExpression> expr,
    DynamicIntensity &out,
    int depth = 0)
{
  using namespace mi::neuraylib;
  using mi::base::make_handle;
  using DynamicSource = Core::EmissionClassification::DynamicSource;

  if (depth > kMaxIntensityWalkDepth)
    return false;
  expr = derefTemporaries(compiledMaterial, expr);
  if (!expr)
    return false;

  switch (expr->get_kind()) {
  case IExpression::EK_CONSTANT: {
    auto constant =
        make_handle(expr->get_interface<const IExpression_constant>());
    return foldColorFactor(make_handle(constant->get_value()).get(), out.scale);
  }
  case IExpression::EK_PARAMETER: {
    if (out.source != DynamicSource::None)
      return false;
    auto param =
        make_handle(expr->get_interface<const IExpression_parameter>());
    const char *name = compiledMaterial->get_parameter_name(param->get_index());
    if (!name)
      return false;
    out.source = DynamicSource::Parameter;
    out.argumentName = name;
    return true;
  }
  case IExpression::EK_DIRECT_CALL: {
    auto call =
        make_handle(expr->get_interface<const IExpression_direct_call>());
    const char *definition = call ? call->get_definition() : nullptr;
    if (!definition)
      return false;
    const std::string_view def(definition);
    auto args = make_handle(call->get_arguments());
    if (!args)
      return false;
    // Exact DB-name prefixes, same masquerade guard as the EDF check above.
    if (def.rfind("mdl::operator*(", 0) == 0) {
      if (args->get_size() != 2)
        return false;
      return walkIntensityFactors(compiledMaterial,
                 make_handle(args->get_expression(mi::Size(0))),
                 out,
                 depth + 1)
          && walkIntensityFactors(compiledMaterial,
              make_handle(args->get_expression(mi::Size(1))),
              out,
              depth + 1);
    }
    if (def.rfind("mdl::color(float)", 0) == 0) {
      // Single-float color constructor only: color(r,g,b) factors would
      // wrongly multiply as three scalars.
      if (args->get_size() != 1)
        return false;
      return walkIntensityFactors(compiledMaterial,
          make_handle(args->get_expression(mi::Size(0))),
          out,
          depth + 1);
    }
    // The lookup's coord/crop/wrap arguments are deliberately ignored: the
    // recipe's mean is TEXTURE-domain (the bound sampler's full-image mean),
    // an approximation of the surface mean — variance-only, never bias.
    if (def.rfind("mdl::tex::lookup_color(", 0) == 0) {
      if (out.source != DynamicSource::None)
        return false;
      auto tex = derefTemporaries(
          compiledMaterial, make_handle(args->get_expression("tex")));
      if (!tex || tex->get_kind() != IExpression::EK_PARAMETER)
        return false;
      auto param =
          make_handle(tex->get_interface<const IExpression_parameter>());
      const char *name =
          compiledMaterial->get_parameter_name(param->get_index());
      if (!name)
        return false;
      out.source = DynamicSource::Texture;
      out.argumentName = name;
      return true;
    }
    return false;
  }
  default:
    return false;
  }
}

} // namespace

Core::EmissionClassification Core::classifyEmission(
    const mi::neuraylib::ICompiled_material *compiledMaterial)
{
  using namespace mi::neuraylib;
  using mi::base::make_handle;

  EmissionClassification result;

  // Author-declared emission is a direct call to df::diffuse_edf; the default
  // edf() compiles to a constant invalid-df. Only the diffuse EDF has uniform
  // radiance over the hemisphere, matching the double-sided Geometry Light
  // sampler and the synthetic next-event hit (ADR 0006's fidelity scope).
  auto edf = derefTemporaries(compiledMaterial,
      make_handle(compiledMaterial->lookup_sub_expression(
          "surface.emission.emission")));
  if (!edf || edf->get_kind() != IExpression::EK_DIRECT_CALL)
    return result;
  {
    auto call =
        make_handle(edf->get_interface<const IExpression_direct_call>());
    const char *definition = call ? call->get_definition() : nullptr;
    // Exact prefix match on the DB name of the elemental EDF, so no user
    // module (::somepdf::, ::pkg::df::) can masquerade as it.
    constexpr std::string_view DIFFUSE_EDF_PREFIX = "mdl::df::diffuse_edf(";
    if (!definition
        || std::string_view(definition).rfind(DIFFUSE_EDF_PREFIX, 0) != 0)
      return result;
  }

  // Only the (default) radiant-exitance intensity mode is handled; power mode
  // needs area normalization the host cannot do here.
  constexpr mi::Sint32 INTENSITY_RADIANT_EXITANCE = 0; // ::df::intensity_mode
  auto mode = derefTemporaries(compiledMaterial,
      make_handle(
          compiledMaterial->lookup_sub_expression("surface.emission.mode")));
  if (mode) {
    if (mode->get_kind() != IExpression::EK_CONSTANT)
      return result;
    auto constant =
        make_handle(mode->get_interface<const IExpression_constant>());
    auto value = make_handle(constant->get_value());
    if (value->get_kind() != IValue::VK_ENUM)
      return result;
    if (make_handle(value->get_interface<const IValue_enum>())->get_value()
        != INTENSITY_RADIANT_EXITANCE)
      return result;
  }

  result.isDiffuseEmission = true;

  // Body-literal intensity folds to a constant; anything argument-, texture- or
  // state-driven stays symbolic under class compilation and has no host value.
  // A symbolic single-factor shape still yields a dynamic recipe below, so the
  // Pick Power can track the live argument instead of the unit proxy.
  constexpr float INV_PI = 0.31830988618379067154f;
  auto intensity = derefTemporaries(compiledMaterial,
      make_handle(compiledMaterial->lookup_sub_expression(
          "surface.emission.intensity")));
  if (!intensity)
    return result;
  if (intensity->get_kind() != IExpression::EK_CONSTANT) {
    DynamicIntensity dyn;
    if (walkIntensityFactors(compiledMaterial, intensity, dyn)
        && dyn.source != EmissionClassification::DynamicSource::None) {
      // Radiance domain (= intensity scale / PI). Non-finite OR negative
      // folded factors keep the proxy: clamping a negative scale to zero
      // would zero the Pick Power while the device could still emit
      // (negative scale x negative argument), making the light NEE-dead —
      // exactly the truncation dimming this recipe exists to avoid.
      bool usable = true;
      for (float &c : dyn.scale) {
        if (!std::isfinite(c) || c < 0.0f) {
          usable = false;
          break;
        }
        c *= INV_PI;
      }
      if (usable) {
        result.dynamicSource = dyn.source;
        result.dynamicArgumentName = std::move(dyn.argumentName);
        result.dynamicScale = dyn.scale;
      }
    }
    return result;
  }

  auto constant =
      make_handle(intensity->get_interface<const IExpression_constant>());
  auto value = make_handle(constant->get_value());
  std::array<float, 3> rgb{};
  if (value->get_kind() == IValue::VK_COLOR) {
    auto color = make_handle(value->get_interface<const IValue_color>());
    for (int i = 0; i < 3; ++i) {
      auto channel = make_handle(color->get_value(i));
      auto f = make_handle(channel->get_interface<const IValue_float>());
      if (!f)
        return result; // not a plain float channel: treat as not host-known
      rgb[i] = f->get_value();
    }
  } else if (value->get_kind() == IValue::VK_FLOAT) {
    const float f =
        make_handle(value->get_interface<const IValue_float>())->get_value();
    rgb = {f, f, f};
  } else {
    return result;
  }

  // Emitted radiance = intensity / PI: the device emission callable returns
  // edf * intensity and a diffuse EDF's value is 1/PI. Storing the unfolded
  // intensity would overweight this emitter PI x in the light-pick CDF.
  // A non-finite channel disqualifies the material entirely — clearing the
  // diffuse flag too, or the textured branch would make it sampleable and
  // next-event estimation would spray the NaN/Inf to every receiver the pick
  // selects (one poisoned pick per sample). Negatives are clamped.
  for (float &c : rgb) {
    if (!std::isfinite(c)) {
      result.isDiffuseEmission = false;
      return result;
    }
    c = std::max(c, 0.0f) * INV_PI;
  }
  result.constantRadiance = rgb;
  return result;
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

  ptxBackend->set_option(
      "num_texture_spaces", std::to_string(kNumTextureSpaces).c_str());
  ptxBackend->set_option(
      "num_texture_results", std::to_string(kNumTextureResults).c_str());
  ptxBackend->set_option_binary("llvm_renderer_module", nullptr, 0);
  ptxBackend->set_option("visible_functions", "");

  ptxBackend->set_option("sm_version", "52");
  ptxBackend->set_option("tex_lookup_call_mode", "direct_call");
  ptxBackend->set_option("lambda_return_mode", "value");
  ptxBackend->set_option("texture_runtime_with_derivs", "off");
  ptxBackend->set_option("inline_aggressively", "on");
  ptxBackend->set_option("opt_level", "2");
  ptxBackend->set_option("enable_exceptions", "off");

  // Generate init, surface scattering, surface emission (emission/intensity/mode),
  // volume scattering and cutout opacity.
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

auto Core::resolveResource(std::string_view resourceId,
    std::string_view ownerName,
    std::string_view ownerFilePath) -> std::string
{
  auto mdlConfiguration = make_handle(
      m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
  auto entityResolver = make_handle(mdlConfiguration->get_entity_resolver());
  // Relative resource paths (e.g. "Textures/foo.png") resolve against the
  // owner module. ownerName is the owner's absolute *name*, ownerFilePath its
  // on-disk path; the latter lets resolution work without relying on the MDL
  // search paths. They occupy distinct argument slots and must not be swapped.
  auto resolvedResource = make_handle(
      entityResolver->resolve_resource(std::string(resourceId).c_str(),
          ownerFilePath.empty() ? nullptr : std::string(ownerFilePath).c_str(),
          ownerName.empty() ? nullptr : std::string(ownerName).c_str(),
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
