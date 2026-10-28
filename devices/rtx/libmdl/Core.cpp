// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "Core.h"
#include "MDLBackendConfig.h"

#include <fmt/core.h>
#include <fmt/std.h>

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>
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

#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

using namespace std::string_literals;
using mi::base::make_handle;

namespace visrtx::libmdl {

namespace detail {

struct SharedNeuray;

std::mutex g_sharedNeurayMutex;
std::weak_ptr<SharedNeuray> g_sharedNeuray;
// An instance is "occupied" from the moment it is published until its image is
// unloaded. A replacement that reloaded the image while the old one was still
// unloading would overlap loadLibrary/freeLibrary and can make mi_factory
// refuse for the rest of the process.
//
// The weak_ptr alone cannot express this: it expires at the last strong
// decrement, which is strictly before the destructor runs. A flag set inside
// the destructor therefore leaves a window in which a concurrent acquire sees
// an expired weak_ptr and no announced teardown, and proceeds. The flag is
// instead claimed by the shared_ptr's deleter, which runs under this mutex
// before the object is destroyed, so expiry and occupancy change together as
// far as any lock holder can observe.
bool g_sharedNeurayTearingDown = false;
std::condition_variable g_sharedNeurayIdle;

// Owns the MDL SDK library image and the single INeuray instance
// `mi_factory` is willing to hand out while one is alive. Held through a
// shared_ptr by every self-owned Core; the SDK is shut down and the library
// unloaded when the last one goes away.
struct SharedNeuray
{
  void *dllHandle{};
  mi::base::Handle<mi::neuraylib::INeuray> neuray;
  // The logger the SDK reports its own diagnostics through. The first Core to
  // create the instance installs the one it was given, if any; everyone else
  // reads whatever is installed. See "Logging" in
  // docs/mdl-instance-sharing.md.
  mi::base::Handle<mi::base::ILogger> logger;
  // Set once this object is reachable through g_sharedNeuray, so its teardown
  // has to be serialized against a replacement.
  bool published{};

  ~SharedNeuray()
  {
    if (neuray.is_valid_interface())
      neuray->shutdown();
    neuray = {};
    logger = {};
    if (dllHandle)
      freeLibrary(dllHandle);
  }
};

// Runs when the last shared_ptr goes, which is also when the weak_ptr expires.
// Claiming the teardown flag here -- under the mutex, before `delete` -- means
// no lock holder can ever see an expired weak_ptr without also seeing the
// instance announced as occupied.
struct SharedNeurayDeleter
{
  void operator()(SharedNeuray *instance) const
  {
    const bool announce = instance->published;
    if (announce) {
      std::lock_guard<std::mutex> guard(g_sharedNeurayMutex);
      g_sharedNeurayTearingDown = true;
    }
    delete instance;
    if (announce) {
      {
        std::lock_guard<std::mutex> guard(g_sharedNeurayMutex);
        g_sharedNeurayTearingDown = false;
      }
      g_sharedNeurayIdle.notify_all();
    }
  }
};

namespace {

// The shared instance if this process already created one, else null. Unlike
// acquireSharedNeuray() this never creates one.
std::shared_ptr<SharedNeuray> peekSharedNeuray()
{
  std::lock_guard<std::mutex> guard(g_sharedNeurayMutex);
  return g_sharedNeuray.lock();
}

// One-time setup for a freshly created instance. Must run before
// INeuray::start(): load_plugin_library fails on a running instance (measured
// -1 on MDL SDK 2026.0.2), so deferring any of this until after startup would
// silently leave the instance without its image plugins.
void configureInstance(SharedNeuray &shared, mi::neuraylib::INeuray *neuray)
{
  auto mdlConfiguration = make_handle(
      neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
  mdlConfiguration->add_mdl_system_paths();
  mdlConfiguration->add_mdl_user_paths();

  auto loggingConfig = make_handle(
      neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
  // The instance gets one receiving logger, and every Core sharing it reports
  // through that. Install the creator's sink if it supplied one; otherwise
  // adopt whatever the SDK already has.
  if (shared.logger.is_valid_interface())
    loggingConfig->set_receiving_logger(shared.logger.get());
  else
    shared.logger = make_handle(loggingConfig->get_receiving_logger());

  auto pluginConf = make_handle(
      neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
  if (mi::Sint32 res = pluginConf->load_plugin_library(
          "nv_openimageio" MI_BASE_DLL_FILE_EXT);
      res != 0) {
    shared.logger->message(mi::base::MESSAGE_SEVERITY_WARNING,
        "MDL",
        "Failed to load the nv_openimageio plugin");
  }
  if (mi::Sint32 res =
          pluginConf->load_plugin_library("dds" MI_BASE_DLL_FILE_EXT);
      res != 0) {
    shared.logger->message(mi::base::MESSAGE_SEVERITY_WARNING,
        "MDL",
        "Failed to load the dds plugin");
  }
}

// `preferredLogger`, when set and when this call is the one that creates the
// instance, becomes the instance's receiving logger.
std::shared_ptr<SharedNeuray> acquireSharedNeuray(
    mi::base::ILogger *preferredLogger)
{
  static constexpr const auto filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;

  std::unique_lock<std::mutex> guard(g_sharedNeurayMutex);

  // An instance whose last reference has gone may still be shutting down and
  // unloading its image. Wait for it to finish before considering a
  // replacement; the deleter announces this before the weak_ptr can be
  // observed as expired, so there is no window between the two.
  g_sharedNeurayIdle.wait(guard, [] { return !g_sharedNeurayTearingDown; });

  if (auto existing = g_sharedNeuray.lock())
    return existing;

  // Never std::make_shared: the deleter is what closes the teardown window.
  std::shared_ptr<SharedNeuray> shared(new SharedNeuray, SharedNeurayDeleter{});
  if (preferredLogger)
    shared->logger = mi::base::make_handle_dup(preferredLogger);

  shared->dllHandle = loadLibrary(filename);
  if (shared->dllHandle == nullptr)
    throw std::runtime_error("Failed to load MDL SDK library "s + filename);

  void *symbol = getProcAddress(shared->dllHandle, "mi_factory");
  if (symbol == nullptr)
    throw std::runtime_error("Failed to find MDL SDK mi_factory symbol");

  shared->neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
  if (!shared->neuray.is_valid_interface()) {
    // Check if we have a valid neuray instance, otherwise check why.
    auto version =
        make_handle(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
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

  configureInstance(*shared, shared->neuray.get());
  // A failed startup must not be published: every later Core would adopt a
  // non-started instance and fail far from the cause.
  if (const mi::Sint32 result = shared->neuray->start(); result != 0) {
    throw std::runtime_error(
        "Failed to start the MDL SDK, INeuray::start() returned "s
        + std::to_string(result));
  }

  shared->published = true;
  g_sharedNeuray = shared;
  return shared;
}

} // namespace

// MDL search paths are a property of the INeuray instance, and every Core
// sharing that instance sets them independently -- including Cores that merely
// adopted an application-supplied instance. The SDK offers no way to add a
// path on behalf of one owner, only clear-and-rebuild, so the latest caller
// would silently drop every other Core's roots. Keep each Core's contribution
// and re-apply their union on every change.
//
// Keyed on the instance rather than on a SharedNeuray, so an externally
// supplied INeuray shared by several Cores is handled the same way.
class SearchPathRegistry
{
 public:
  using Paths = std::vector<std::filesystem::path>;

  static SearchPathRegistry &instance()
  {
    static SearchPathRegistry registry;
    return registry;
  }

  void setMdlPaths(
      mi::neuraylib::INeuray *neuray, const void *owner, Paths paths)
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto &entry = m_entries[neuray];
    entry.mdl[owner] = std::move(paths);
    applyMdl(neuray, entry);
  }

  void setResourcePaths(
      mi::neuraylib::INeuray *neuray, const void *owner, Paths paths)
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto &entry = m_entries[neuray];
    entry.resource[owner] = std::move(paths);
    applyResource(neuray, entry);
  }

  // Drop an owner's contribution and re-apply what is left. The instance is
  // passed in because a Core is forgotten while its handle is still valid.
  void forget(mi::neuraylib::INeuray *neuray, const void *owner)
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto it = m_entries.find(neuray);
    if (it == m_entries.end())
      return;
    auto &entry = it->second;
    const bool hadMdl = entry.mdl.erase(owner) != 0;
    const bool hadResource = entry.resource.erase(owner) != 0;
    // Rebuild before forgetting the instance, or the departed owner's roots
    // stay installed on an INeuray that outlives us -- an application-supplied
    // instance, typically.
    if (hadMdl)
      applyMdl(neuray, entry);
    if (hadResource)
      applyResource(neuray, entry);
    if (entry.mdl.empty() && entry.resource.empty()) {
      // Last owner gone: forget the instance too, so a later instance that
      // reuses the address does not inherit stale contributions.
      m_entries.erase(it);
    }
  }

 private:
  // Keyed by owner address, so the order in which contributions are applied
  // is unspecified. Where two devices supply roots containing a module of the
  // same name, which one wins is therefore unspecified too -- documented in
  // docs/mdl-search-paths.md rather than papered over with an arbitrary rule.
  struct Entry
  {
    std::map<const void *, Paths> mdl;
    std::map<const void *, Paths> resource;
  };

  static void applyMdl(mi::neuraylib::INeuray *neuray, const Entry &entry)
  {
    auto mdlConfiguration = make_handle(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mdlConfiguration->clear_mdl_paths();
    std::set<std::string> seen;
    for (const auto &[owner, paths] : entry.mdl) {
      for (const auto &path : paths) {
        auto asString = path.generic_string();
        if (seen.insert(asString).second)
          mdlConfiguration->add_mdl_path(asString.c_str());
      }
    }
    mdlConfiguration->add_mdl_system_paths();
    mdlConfiguration->add_mdl_user_paths();
  }

  static void applyResource(mi::neuraylib::INeuray *neuray, const Entry &entry)
  {
    auto mdlConfiguration = make_handle(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mdlConfiguration->clear_resource_paths();
    std::set<std::string> seen;
    for (const auto &[owner, paths] : entry.resource) {
      for (const auto &path : paths) {
        // generic_string() throughout, so the same root spelled two ways
        // de-duplicates on Windows as well as POSIX.
        auto asString = path.generic_string();
        if (seen.insert(asString).second)
          mdlConfiguration->add_resource_path(asString.c_str());
      }
    }
  }

  std::mutex m_mutex;
  std::map<mi::neuraylib::INeuray *, Entry> m_entries;
};

} // namespace detail

Core::Core() : Core({}, {}) {}

Core::Core(mi::base::ILogger *logger) : Core({}, logger) {}

Core::Core(mi::neuraylib::INeuray *neuray) : Core(neuray, {}) {}

Core::Core(mi::neuraylib::INeuray *neuray, mi::base::ILogger *logger)
{
  if (neuray && logger) {
    throw std::runtime_error(
        "Only one of neuray or logger can be provided to libmdl::Core");
  }

  nonstd::scope_fail handleCleanup([this]() {
    m_executionContext = {};
    m_mdlFactory = {};
    m_globalScope = {};
    m_neuray = {};
    m_logger = {};
    m_shared = {};
  });

  if (neuray) {
    // Externally owned instance: whoever owns it configures and shuts it down,
    // including its logger. Note this Core takes no lease on the started SDK
    // or its loaded image -- holding an INeuray reference is not that -- so
    // the owner must outlive it (docs/mdl-instance-sharing.md).
    m_neuray = mi::base::make_handle_dup(neuray);

    // Report through the instance's logger, so this Core's own messages do
    // not silently fall back to stderr.
    auto loggingConfig = make_handle(
        m_neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
    m_logger = make_handle(loggingConfig->get_receiving_logger());
  } else {
    // A supplied sink is offered to the shared instance as its receiving
    // logger: the first Core in installs it, later ones leave it alone, since
    // the SDK takes only one per instance. That governs where the SDK's own
    // diagnostics go.
    //
    // This Core keeps its own sink regardless, so the messages VisRTX logs
    // itself -- which never pass through the SDK's logger -- still reach the
    // device they belong to. Only Cores given no sink fall back to the
    // instance's.
    if (logger)
      m_logger = mi::base::make_handle_dup(logger);
    m_shared = detail::acquireSharedNeuray(m_logger.get());
    m_neuray = mi::base::make_handle_dup(m_shared->neuray.get());
    if (!m_logger.is_valid_interface())
      m_logger = mi::base::make_handle_dup(m_shared->logger.get());
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
  if (m_neuray.is_valid_interface())
    detail::SearchPathRegistry::instance().forget(m_neuray.get(), this);
  m_logger = {};
  m_neuray = {};
  // Shuts down and unloads the SDK if this was the last live Core.
  m_shared = {};
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
    auto rootName = std::filesystem::path(
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

  ptxBackend->set_option(
      "num_texture_spaces", std::to_string(kNumTextureSpaces).c_str());
  ptxBackend->set_option(
      "num_texture_results", std::to_string(kNumTextureResults).c_str());
  ptxBackend->set_option_binary("llvm_renderer_module", nullptr, 0);
  ptxBackend->set_option("visible_functions", "");

  ptxBackend->set_option("sm_version", VISRTX_MDL_SM_VERSION);
  ptxBackend->set_option("tex_lookup_call_mode", "direct_call");
  ptxBackend->set_option("lambda_return_mode", "value");
  ptxBackend->set_option("texture_runtime_with_derivs", "off");
  ptxBackend->set_option("inline_aggressively", "on");
  ptxBackend->set_option("opt_level", "2");
  ptxBackend->set_option("enable_exceptions", "off");
  // Generate the BSDF auxiliary function (albedo + normal) so the albedo AOV /
  // denoiser guide reads a faithful diffuse+glossy albedo straight from the
  // compiled material, instead of distilling to a `diffuse` proxy.
  ptxBackend->set_option("enable_auxiliary", "on");
  // Keep shading normals valid for the viewer through the renderer's
  // adapt_normal (MDLTexture_ptx.cu), as the native physicallyBased shader
  // does.
  ptxBackend->set_option("use_renderer_adapt_normal", "on");

  // Generate init, surface scattering, surface emission
  // (emission/intensity/mode), volume scattering and cutout opacity.
  // Not static: add_material() writes function_index/return_code back into each
  // entry, so a shared instance would be a data race across parallel compile
  // jobs (ADR 0009). A per-call array keeps each job's writes private.
  mi::neuraylib::Target_function_description materialFunctions[] = {
      {"init", "mdlInit"},
      {"thin_walled", "mdlThinWalled"},

      {"surface.scattering", "mdlBsdf"},
      {"surface.emission.emission", "mdlEmission"},
      {"surface.emission.intensity", "mdlEmissionIntensity"},
      {"surface.emission.mode", "mdlEmissionMode"},

      {"volume.scattering_coefficient", "mdlTransmission"},

      {"geometry.cutout_opacity", "mdlOpacity"},
  };

  // Generate target code for the compiled material
  auto linkUnit = make_handle(
      ptxBackend->create_link_unit(transaction, executionContext.get()));

  // Add main material functions. `surface.scattering` also emits the BSDF
  // auxiliary function (mdlBsdf_auxiliary: albedo + normal) because
  // enable_auxiliary is on.
  linkUnit->add_material(compiledMaterial,
      std::data(materialFunctions),
      std::size(materialFunctions),
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
  detail::SearchPathRegistry::instance().setMdlPaths(
      m_neuray.get(), this, {std::cbegin(paths), std::cend(paths)});
}

auto Core::setMdlResourceSearchPaths(nonstd::span<std::filesystem::path> paths)
    -> void
{
  detail::SearchPathRegistry::instance().setResourcePaths(
      m_neuray.get(), this, {std::cbegin(paths), std::cend(paths)});
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
