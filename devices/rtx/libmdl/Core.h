// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/uuid.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_compiler.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/target_code_types.h>

#include <nonstd/expected.hpp>
#include <nonstd/span.hpp>

#ifndef __CUDACC__
// Explicitly exclude this from device code.
#include <fmt/core.h>
#include <fmt/format.h>
#endif

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace visrtx::libmdl {

class Core
{
 public:
  // Host-side emission classification of a compiled material, per ADR 0006.
  // Computed once at compile time (the compiled material is not retained), so
  // an Emissive Surface can be synthesized into a Geometry Light without
  // recompiling or resolving class-compiled arguments.
  struct EmissionClassification
  {
    // Emitted radiance (= folded intensity / PI, a diffuse EDF's value being
    // 1/PI) when `surface.emission.intensity` folds to a body-literal constant;
    // nullopt when it does not (texture / procedural / parameter-driven — not
    // host-knowable under class compilation).
    std::optional<std::array<float, 3>> constantRadiance;
    // A diffuse radiant-exitance emission EDF is present AND eligible — a
    // non-finite folded constant clears it, disqualifying the material.
    // Invariant: constantRadiance.has_value() implies isDiffuseEmission.
    bool isDiffuseEmission{false};

    // Dynamic mean-radiance recipe: when the intensity does not fold but is a
    // SINGLE class-compilation argument (color/float parameter, or the `tex`
    // of a tex::lookup_color) times folded constants, the host can still
    // compute a live mean radiance at light-build time:
    //   mean = <current argument value | bound sampler mean> * dynamicScale
    // dynamicScale is radiance-domain (the folded constants already carry the
    // diffuse EDF's 1/PI). Without a recipe the Pick Power falls back to the
    // unit proxy — unbiased, but under-picking a bright emitter turns the
    // firefly clamp and the last-depth MIS truncation into visible dimming
    // next to correctly-powered lights.
    enum class DynamicSource
    {
      None,
      Parameter,
      Texture,
    };
    DynamicSource dynamicSource{DynamicSource::None};
    std::string dynamicArgumentName;
    std::array<float, 3> dynamicScale{};
  };

  // The main neuray interface can only be acquired once. Possibly get it
  // as a parameter instead of allocating it internally.
  // Note that we allow overriding the logger only if we own the
  // neuray instance, otherwise we assume logging is already taken care of
  Core();
  Core(mi::base::ILogger *logger);
  Core(mi::neuraylib::INeuray *neuray);

  ~Core();

  // Set MDL search path. It will also add user and system paths.
  void setMdlSearchPaths(nonstd::span<std::filesystem::path> paths);

  // Set MDL resources (textures, light profiles...) search path.
  void setMdlResourceSearchPaths(nonstd::span<std::filesystem::path> paths);

  // Add builtin modules to the global scope
  void addBuiltinModule(
      std::string_view moduleName, std::string_view moduleSource);

  // Load an MDL module from in-memory source into `transaction` and return it.
  // Returns null on failure (diagnostics are logged). Mirrors loadModule.
  const mi::neuraylib::IModule *loadModuleFromString(std::string_view moduleName,
      std::string_view moduleSource,
      mi::neuraylib::ITransaction *transaction);

  // Access an already-loaded module by its MDL name within `transaction`.
  const mi::neuraylib::IModule *accessModule(
      std::string_view moduleName, mi::neuraylib::ITransaction *transaction);

  // The main neuray interface can only be acquired once. Make sure it can be
  // shared if taken from there. The original subsystem keeps the ownership of
  // the returned value.
  mi::neuraylib::INeuray *getINeuray() const;

  mi::neuraylib::IMdl_factory *getMdlFactory() const;

  // Might return null if no logger setup. Use logMessage to have a fallback to
  // stderr.
  mi::base::ILogger *getLogger() const;

#ifdef __CUDACC__
  template <typename... T>
  void logMessage(
      mi::base::Message_severity severity, const char *format, T... fmtargs);
#else
  template <typename... T>
  void logMessage(mi::base::Message_severity severity,
      fmt::format_string<T...> format,
      T &&...fmtargs)
  {
    if (m_logger.is_valid_interface()) {
      m_logger->message(severity,
          "MDL",
          fmt::format(format, std::forward<T>(fmtargs)...).c_str());
    } else {
      fmt::println(stderr, format, std::forward<T>(fmtargs)...);
    }
  }
#endif

  // Database scopes
  mi::neuraylib::IScope *createScope(
      std::string_view scopeName, mi::neuraylib::IScope *parent = {});
  void removeScope(mi::neuraylib::IScope *scope);

  // Transaction
  mi::neuraylib::ITransaction *createTransaction(
      mi::neuraylib::IScope *scope = {});

  // Module and functions
  const mi::neuraylib::IModule *loadModule(std::string_view moduleOrFileName,
      mi::neuraylib::ITransaction *transaction);

  // Fallback for a path-based load that failed (e.g. a scene shipped a .mdl
  // without its ./textures). If the path passes through a directory whose
  // basename matches an MDL search root, derive the canonical module name from
  // the tail and load it by name so the entity resolver finds a complete copy
  // on the search path with resources co-located. Returns null if nothing
  // matches or the named module still fails.
  const mi::neuraylib::IModule *loadModuleByCanonicalName(
      std::string_view filePath, mi::neuraylib::ITransaction *transaction);

  const mi::neuraylib::IFunction_definition *getFunctionDefinition(
      const mi::neuraylib::IModule *module,
      std::string_view functionName,
      mi::neuraylib::ITransaction *transaction);

  mi::neuraylib::ICompiled_material *getCompiledMaterial(
      const mi::neuraylib::IFunction_definition *,
      bool classCompilation = true);

  static EmissionClassification classifyEmission(
      const mi::neuraylib::ICompiled_material *compiledMaterial);

  mi::neuraylib::ICompiled_material *getDistilledToDiffuse(
      const mi::neuraylib::ICompiled_material *compiledMaterial);

  const mi::neuraylib::ITarget_code *getPtxTargetCode(
      const mi::neuraylib::ICompiled_material *compiledMaterial,
      mi::neuraylib::ITransaction *transaction);

  std::string resolveResource(std::string_view resourceId,
      std::string_view ownerName = {},
      std::string_view ownerFilePath = {});
  std::string resolveModule(std::string_view moduleId);

 private:
  Core(mi::neuraylib::INeuray *neuray, mi::base::ILogger *logger);

  // Raw load_module_from_string wrapper. Returns the MDL result code:
  // 0 = loaded, 1 = already present (both success), < 0 = failure (logged).
  mi::Sint32 loadModuleSource(std::string_view moduleName,
      std::string_view moduleSource,
      mi::neuraylib::ITransaction *transaction);

  using DllHandle = void *;
  DllHandle m_dllHandle;
  mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
  mi::base::Handle<mi::neuraylib::IScope> m_globalScope;
  mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdlFactory;
  mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_executionContext;
  mi::base::Handle<mi::base::ILogger> m_logger;

  bool logExecutionContextMessages(
      const mi::neuraylib::IMdl_execution_context *executionContext);
};

} // namespace visrtx::libmdl
