/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Device-free unit tests for the MDL emission classifier. libmdl is a
// standalone static library (no CUDA/OptiX/GPU), so classifyEmission is
// exercised on the host against inline .mdl snippets — the same compile flow
// the device runs in MaterialRegistry::acquireMaterialFromCode, minus the
// device. Only the MDL SDK shared library must be discoverable at runtime; if
// it is not, the test SKIPs (return code 77, wired in CMake) rather than
// failing.
//
// These lock the CURRENT classifyEmission behavior as a baseline before the
// descriptor refactor (ADR 0007). The descriptor-fold vectors are added to this
// same target as the refactor lands.

#include "libmdl/Core.h"
#include "libmdl/EmissionIR.h"
#include "libmdl/source_name_utils.h"

#include <mi/base/handle.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <string_view>

using visrtx::libmdl::Core;
using visrtx::libmdl::makeInlineModuleName;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);              \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

constexpr int kSkipReturnCode = 77;

bool approxEqual(float a, float b, float tol = 1e-4f)
{
  return std::fabs(a - b) <= tol * std::max(1.0f, std::fabs(b));
}

// Compile `source`'s `material emissive(...)` and classify its emission. The
// compiled material must outlive the returned classification's use, so it is
// kept alive for the duration of the caller's asserts via `keepAlive`.
Core::EmissionClassification classify(Core &core,
    mi::neuraylib::ITransaction *txn,
    std::string_view source,
    mi::base::Handle<mi::neuraylib::ICompiled_material> &keepAlive)
{
  using mi::base::make_handle;
  const auto moduleName = makeInlineModuleName(source);
  auto module = make_handle(core.loadModuleFromString(moduleName, source, txn));
  if (!module.is_valid_interface()) {
    std::printf("FAIL could not load inline module\n");
    ++g_failures;
    return {};
  }
  auto fnDef =
      make_handle(core.getFunctionDefinition(module.get(), "emissive", txn));
  if (!fnDef.is_valid_interface()) {
    std::printf("FAIL could not find material 'emissive'\n");
    ++g_failures;
    return {};
  }
  keepAlive = make_handle(core.getCompiledMaterial(fnDef.get()));
  if (!keepAlive.is_valid_interface()) {
    std::printf("FAIL could not compile material\n");
    ++g_failures;
    return {};
  }
  return Core::classifyEmission(keepAlive.get());
}

// intensity = color(2.0) * math::PI folds to a body-literal constant: emitted
// radiance = intensity / PI = 2.0 per channel.
const char *kConstLiteral = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material emissive() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(2.0) * math::PI)));
)mdl";

// A parameter with a default stays symbolic under class compilation, so the
// intensity does not fold; the single-factor walk records a Parameter recipe
// with scale = PI * (1/PI) = 1.
const char *kParamDriven = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material emissive(color value = color(3.0)) = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: value * math::PI)));
)mdl";

// Textured intensity: tex::lookup_color(tex) with tex a symbolic parameter →
// Texture recipe.
const char *kTextured = R"mdl(mdl 1.6;
import ::df::*;
import ::tex::*;
export material emissive(uniform texture_2d tex = texture_2d()) = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: tex::lookup_color(tex: tex, coord: float2(0.0)))));
)mdl";

// No emission: emission defaults to edf() (a constant invalid-df), not a
// df::diffuse_edf direct call.
const char *kNoEmission = R"mdl(mdl 1.6;
import ::df::*;
export material emissive() = material(
    surface: material_surface(
        scattering: df::diffuse_reflection_bsdf()));
)mdl";

// Diffuse EDF but power intensity mode — the classifier only handles
// radiant-exitance, so this is rejected.
const char *kPowerMode = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material emissive() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(2.0) * math::PI,
            mode: intensity_power)));
)mdl";

using DynamicSource = Core::EmissionClassification::DynamicSource;

// A let-shared subexpression: `k` is referenced twice, so class compilation
// stores it as a single temporary. The IR must resolve both references to the
// same node.
const char *kSharedTemporary = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material emissive(color value = color(2.0)) = let {
    color k = value * math::PI;
} in material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: k + k)));
)mdl";

using visrtx::libmdl::buildEmissionIR;
using visrtx::libmdl::ConstantKind;
using visrtx::libmdl::EmissionIR;
using visrtx::libmdl::EmissionNodeKind;
using Semantic = visrtx::libmdl::Semantic;

const visrtx::libmdl::EmissionNode &node(const EmissionIR &ir, int i)
{
  return ir.nodes[std::size_t(i)];
}

bool hasParamNamed(const EmissionIR &ir, const char *name)
{
  for (const auto &n : ir.nodes)
    if (n.parameterName == name)
      return true;
  return false;
}

void runIR(Core &core, mi::neuraylib::ITransaction *txn)
{
  using mi::base::make_handle;
  mi::base::Handle<mi::neuraylib::ICompiled_material> keepAlive;

  auto buildIR = [&](std::string_view src) {
    (void)classify(core, txn, src, keepAlive); // reuse the compile flow
    return buildEmissionIR(keepAlive.get(), txn);
  };

  {
    auto ir = buildIR(kConstLiteral);
    CHECK(!ir.empty());
    CHECK(ir.surface.edfRoot >= 0);
    CHECK(node(ir, ir.surface.edfRoot).kind == EmissionNodeKind::Call);
    CHECK(node(ir, ir.surface.edfRoot).semantic
        == Semantic::DS_INTRINSIC_DF_DIFFUSE_EDF);
    CHECK(ir.surface.intensityRoot >= 0);
    CHECK(ir.emissionDeps.empty()); // body-literal: no argument deps
  }

  {
    auto ir = buildIR(kParamDriven);
    CHECK(node(ir, ir.surface.edfRoot).semantic
        == Semantic::DS_INTRINSIC_DF_DIFFUSE_EDF);
    CHECK(!ir.emissionDeps.empty());
    CHECK(hasParamNamed(ir, "value"));
  }

  {
    auto ir = buildIR(kTextured);
    bool foundTexture = false;
    for (const auto &n : ir.nodes) {
      if (n.kind == EmissionNodeKind::Texture && n.parameterName == "tex")
        foundTexture = true;
    }
    CHECK(foundTexture);
    CHECK(hasParamNamed(ir, "tex"));
    CHECK(!ir.emissionDeps.empty());
  }

  {
    auto ir = buildIR(kNoEmission);
    CHECK(ir.surface.edfRoot >= 0);
    CHECK(node(ir, ir.surface.edfRoot).kind == EmissionNodeKind::Constant);
    CHECK(node(ir, ir.surface.edfRoot).constantKind == ConstantKind::InvalidDf);
  }

  {
    // k + k: the two operands of the addition must resolve to the SAME node
    // index (shared temporary), proving CSE-identity is preserved.
    auto ir = buildIR(kSharedTemporary);
    const auto &intensity = node(ir, ir.surface.intensityRoot);
    CHECK(intensity.kind == EmissionNodeKind::Call);
    CHECK(intensity.operands.size() == 2);
    if (intensity.operands.size() == 2)
      CHECK(intensity.operands[0] == intensity.operands[1]);
  }
}

void run(Core &core, mi::neuraylib::ITransaction *txn)
{
  mi::base::Handle<mi::neuraylib::ICompiled_material> keepAlive;

  {
    auto c = classify(core, txn, kConstLiteral, keepAlive);
    CHECK(c.isDiffuseEmission);
    CHECK(c.constantRadiance.has_value());
    if (c.constantRadiance) {
      CHECK(approxEqual((*c.constantRadiance)[0], 2.0f));
      CHECK(approxEqual((*c.constantRadiance)[1], 2.0f));
      CHECK(approxEqual((*c.constantRadiance)[2], 2.0f));
    }
    CHECK(c.dynamicSource == DynamicSource::None);
  }

  {
    auto c = classify(core, txn, kParamDriven, keepAlive);
    CHECK(c.isDiffuseEmission);
    CHECK(!c.constantRadiance.has_value());
    CHECK(c.dynamicSource == DynamicSource::Parameter);
    CHECK(c.dynamicArgumentName == "value");
    CHECK(approxEqual(c.dynamicScale[0], 1.0f));
    CHECK(approxEqual(c.dynamicScale[1], 1.0f));
    CHECK(approxEqual(c.dynamicScale[2], 1.0f));
  }

  {
    auto c = classify(core, txn, kTextured, keepAlive);
    CHECK(c.isDiffuseEmission);
    CHECK(!c.constantRadiance.has_value());
    CHECK(c.dynamicSource == DynamicSource::Texture);
    CHECK(c.dynamicArgumentName == "tex");
  }

  {
    auto c = classify(core, txn, kNoEmission, keepAlive);
    CHECK(!c.isDiffuseEmission);
    CHECK(!c.constantRadiance.has_value());
    CHECK(c.dynamicSource == DynamicSource::None);
  }

  {
    auto c = classify(core, txn, kPowerMode, keepAlive);
    CHECK(!c.isDiffuseEmission);
    CHECK(!c.constantRadiance.has_value());
  }
}

} // namespace

int main()
{
  using mi::base::make_handle;

  try {
    Core core;
    auto scope = core.createScope("MdlEmissionClassifierTestScope");
    auto txn = make_handle(core.createTransaction(scope));
    run(core, txn.get());
    runIR(core, txn.get());
    txn->commit();
    core.removeScope(scope);
  } catch (const std::exception &e) {
    std::printf("SKIP MDL SDK unavailable: %s\n", e.what());
    return kSkipReturnCode;
  }

  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
