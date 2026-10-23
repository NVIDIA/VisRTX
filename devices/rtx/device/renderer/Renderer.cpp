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

#include "Renderer.h"
// helium
#include <helium/utility/TimeStamp.h>

// specific renderers
#include "Debug.h"
#include "Fast.h"
#include "Interactive.h"
#include "Quality.h"
#include "Test.h"
#include "UnknownRenderer.h"

#include "gpu/gpu_decl.h"
#include "gpu/shadingState.h"
#include "utility/AnariTypeHelpers.h"

// std
#include <optix_types.h>
#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <optional>
#include <string_view>
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

namespace visrtx {

template <typename T>
struct SbtRecord
{
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

template <>
struct SbtRecord<void>
{
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

using RaygenRecord = SbtRecord<void>;
using MissRecord = SbtRecord<void>;
using HitgroupRecord = SbtRecord<void>;
using MaterialRecord = SbtRecord<void>;

// Helper functions ///////////////////////////////////////////////////////////

// Map the fireflyFilterMode string to the enum. Unknown strings (and "none")
// resolve to NONE. The default-when-unset is decided by the caller and passed
// in as the string, so this stays a pure lookup with no special cases.
static FireflyFilterMode parseFireflyFilterMode(const std::string &mode)
{
  if (mode == "tonemap")
    return FireflyFilterMode::TONEMAP;
  if (mode == "clamp")
    return FireflyFilterMode::CLAMP;
  if (mode == "trim")
    return FireflyFilterMode::TRIM;
  return FireflyFilterMode::NONE;
}

static std::string longestBeginningMatch(
    const std::string_view &first, const std::string_view &second)
{
  auto maxMatchLength = std::min(first.size(), second.size());
  auto start1 = first.begin();
  auto start2 = second.begin();
  auto end = first.begin() + maxMatchLength;

  return std::string(start1, std::mismatch(start1, end, start2).first);
}

static bool beginsWith(const std::string_view &inputString,
    const std::string_view &startsWithString)
{
  auto startingMatch = longestBeginningMatch(inputString, startsWithString);
  return startingMatch.size() == startsWithString.size();
}

static Renderer *make_renderer(std::string_view subtype, DeviceGlobalState *d)
{
  auto splitString = [](const std::string &input,
                         const std::string &delim) -> std::vector<std::string> {
    std::vector<std::string> tokens;
    size_t pos = 0;
    while (true) {
      size_t begin = input.find_first_not_of(delim, pos);
      if (begin == input.npos)
        return tokens;
      size_t end = input.find_first_of(delim, begin);
      tokens.push_back(input.substr(
          begin, (end == input.npos) ? input.npos : (end - begin)));
      pos = end;
    }
  };

  if (subtype == "fast")
    return new Fast(d);
  else if (subtype == "quality")
    return new Quality(d);
  else if (subtype == "interactive" || subtype == "default")
    return new Interactive(d);
  else if (subtype == "test")
    return new Test(d);
  else if (beginsWith(subtype, "debug")) {
    auto *retval = new Debug(d);
    auto names = splitString(std::string(subtype), "_");
    if (names.size() > 1)
      retval->setParam("method", ANARI_STRING, names[1].c_str());
    return retval;
  } else
    return new UnknownRenderer(subtype, d);
}

// Renderer definitions ///////////////////////////////////////////////////////

Renderer::Renderer(DeviceGlobalState *s, float defaultAmbientRadiance)
    : Object(ANARI_RENDERER, s),
      m_backgroundImage(this),
      m_defaultAmbientRadiance(defaultAmbientRadiance)
{
  m_ambientIntensity = defaultAmbientRadiance;
  helium::BaseObject::markParameterChanged();
  s->commitBuffer.addObjectToCommit(this);
}

Renderer::~Renderer()
{
  cleanup();
  releasePipeline();
}

void Renderer::commitParameters()
{
  auto *backgroundImage = getParamObject<Array2D>("background");
  // Dropping the old array may free its storage; destroy what refers to it.
  if (backgroundImage != m_backgroundImage.get())
    cleanup();
  m_backgroundImage = backgroundImage;
  m_bgColor = getParam<vec4>("background", vec4(vec3(0.f), 1.f));
  m_spp = getParam<int>("pixelSamples", 1);
  m_ambientColor = getParam<vec3>("ambientColor", vec3(1.f));
  m_ambientIntensity =
      getParam<float>("ambientRadiance", m_defaultAmbientRadiance);
  m_occlusionDistance = getParam<float>("ambientOcclusionDistance", 1e20f);
  m_checkerboard = getParam<bool>("checkerboarding", false);

  m_denoise = getParam<bool>("denoise", false);
  m_denoiseStart = getParam<int>("denoiseStart", 0);
  auto denoiseMode = getParamString("denoiseMode", "color");
  m_denoiseAlbedo =
      (denoiseMode == "colorAlbedo" || denoiseMode == "colorAlbedoNormal");
  m_denoiseNormal = (denoiseMode == "colorAlbedoNormal");

  // Default to tonemap, matching the pre-enum behaviour.
  m_fireflyFilterMode =
      parseFireflyFilterMode(getParamString("fireflyFilterMode", "tonemap"));
  // Default k=8: the cap is mean + k*stddev with σ estimated from clamped
  // samples (outliers excluded), so a moderate k bites without one firefly
  // raising its own threshold. Lower clamps harder (more bias); raise for more
  // energy fidelity at the cost of leaking brighter fireflies.
  m_fireflyFilterSigma =
      std::max(0.f, getParam<float>("fireflyFilterSigma", 8.f));
  m_fireflyFilterWarmup = std::max(1, getParam<int>("fireflyFilterWarmup", 4));
  // TRIM tracks this many of the brightest samples per pixel and trims the ones
  // a base-excluding threshold flags as outliers. Small: per-pixel memory is
  // trim*vec4 and a handful covers the firefly count of any one pixel.
  m_fireflyFilterTrim = std::clamp(getParam<int>("fireflyFilterTrim", 4), 1, 8);
  m_sampleLimit = getParam<int>("sampleLimit", 128);
  m_cullTriangleBF = getParam<bool>("cullTriangleBackfaces", false);
  m_volumeSamplingRate =
      std::clamp(getParam<float>("volumeSamplingRate", 0.125f), 1e-3f, 10.f);
  m_premultiplyBackground = getParam<bool>("premultiplyBackground", false);
  m_cutPlane = getParam<vec4>("cutPlane", vec4(0.f));
  if (m_checkerboard)
    m_spp = 1;
}

// Defaults mirror visrtx_renderer_fog.json, the object-info source of truth;
// the generated schema is not loadable at runtime, so they are restated here.
static const vec3 kDefaultFogColor{1.f};
static constexpr float kDefaultFogStart = 0.f;
static constexpr float kDefaultFogEnd = 1.f;
static constexpr float kDefaultFogDensity = 1.f;

void Renderer::reportInvalidFog(const char *name)
{
  reportMessage(ANARI_SEVERITY_WARNING,
      "invalid %s; disabling fog for this committed configuration",
      name);
}

bool Renderer::hasValidFogType(const char *name, ANARIDataType type)
{
  if (!hasParam(name) || hasParam(name, type))
    return true;
  reportInvalidFog(name);
  return false;
}

std::optional<FogMode> Renderer::parseFogMode()
{
  if (!hasValidFogType("fogMode", ANARI_STRING))
    return std::nullopt;
  const auto mode = getParamString("fogMode", "none");
  if (mode == "none")
    return FogMode::NONE;
  if (mode == "linear")
    return FogMode::LINEAR;
  if (mode == "exp")
    return FogMode::EXP;
  if (mode == "exp2")
    return FogMode::EXP2;
  reportInvalidFog("fogMode");
  return std::nullopt;
}

std::optional<FogDistanceMetric> Renderer::parseFogDistanceMetric()
{
  if (!hasValidFogType("fogDistanceMetric", ANARI_STRING))
    return std::nullopt;
  const auto metric = getParamString("fogDistanceMetric", "viewDepth");
  if (metric == "viewDepth")
    return FogDistanceMetric::VIEW_DEPTH;
  if (metric == "rayDistance")
    return FogDistanceMetric::RAY_DISTANCE;
  reportInvalidFog("fogDistanceMetric");
  return std::nullopt;
}

std::optional<FogColorSource> Renderer::parseFogColorSource()
{
  if (!hasValidFogType("fogColorSource", ANARI_STRING))
    return std::nullopt;
  const auto source = getParamString("fogColorSource", "constant");
  if (source == "constant")
    return FogColorSource::CONSTANT;
  if (source == "background")
    return FogColorSource::BACKGROUND;
  reportInvalidFog("fogColorSource");
  return std::nullopt;
}

std::optional<vec3> Renderer::parseFogColor()
{
  if (!hasValidFogType("fogColor", ANARI_FLOAT32_VEC3))
    return std::nullopt;
  const auto color = getParam<vec3>("fogColor", kDefaultFogColor);
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(color[i]) || color[i] < 0.f) {
      reportInvalidFog("fogColor");
      return std::nullopt;
    }
  }
  return color;
}

bool Renderer::parseFogLinearRange(FogGPUData &fog)
{
  if (!hasValidFogType("fogStart", ANARI_FLOAT32)
      || !hasValidFogType("fogEnd", ANARI_FLOAT32))
    return false;
  const float start = getParam<float>("fogStart", kDefaultFogStart);
  const float end = getParam<float>("fogEnd", kDefaultFogEnd);
  if (!std::isfinite(start) || start < 0.f) {
    reportInvalidFog("fogStart");
    return false;
  }
  if (!std::isfinite(end) || end <= start) {
    reportInvalidFog("fogEnd");
    return false;
  }
  fog.start = start;
  fog.end = end;
  return true;
}

bool Renderer::parseFogDensity(FogGPUData &fog)
{
  if (!hasValidFogType("fogDensity", ANARI_FLOAT32))
    return false;
  const float density = getParam<float>("fogDensity", kDefaultFogDensity);
  if (!std::isfinite(density) || density < 0.f) {
    reportInvalidFog("fogDensity");
    return false;
  }
  fog.density = density;
  return true;
}

// nullopt means "no effective fog this Commit": either mode none, or an
// invalid active setting that was warned about. Inactive settings are not
// validated, and an invalid Commit never keeps a previously effective fog.
std::optional<FogGPUData> Renderer::parseFogParameters()
{
  const auto mode = parseFogMode();
  if (!mode || *mode == FogMode::NONE)
    return std::nullopt;

  FogGPUData fog{};
  fog.mode = *mode;

  const auto metric = parseFogDistanceMetric();
  if (!metric)
    return std::nullopt;
  fog.distanceMetric = *metric;

  const auto source = parseFogColorSource();
  if (!source)
    return std::nullopt;
  fog.colorSource = *source;
  if (fog.colorSource == FogColorSource::CONSTANT) {
    const auto color = parseFogColor();
    if (!color)
      return std::nullopt;
    fog.color = *color;
  }

  const bool rangeOk = fog.mode == FogMode::LINEAR ? parseFogLinearRange(fog)
                                                   : parseFogDensity(fog);
  if (!rangeOk)
    return std::nullopt;
  return fog;
}

void Renderer::finalize()
{
  cleanup();
  if (m_backgroundImage) {
    const ANARIDataType fmt = m_backgroundImage->elementType();
    if (numANARIChannels(fmt) == 0) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "invalid background image element type (%s); ignoring",
          anari::toString(fmt));
      return; // m_backgroundTexture stays null -> COLOR background fallback
    }
    const bool isFp = isFloat(fmt);
    const cudaArray_t cuArray = m_backgroundImage->cudaArray();
    if (!cuArray)
      return;
    m_backgroundTexture = makeCudaTextureObject2D(cuArray,
        !isFp,
        "linear",
        "clampToEdge",
        "clampToEdge",
        vec4(0.f),
        isSrgb8(fmt));
  }
}

Span<HitgroupFunctionNames> Renderer::hitgroupSbtNames() const
{
  return make_Span(&m_defaultHitgroupNames, 1);
}

Span<std::string> Renderer::missSbtNames() const
{
  return make_Span(&m_defaultMissName, 1);
}

void Renderer::populateFrameData(FrameGPUData &fd) const
{
  if (m_backgroundImage && m_backgroundTexture) {
    fd.renderer.backgroundMode = BackgroundMode::IMAGE;
    fd.renderer.background.texobj = m_backgroundTexture;
  } else {
    fd.renderer.backgroundMode = BackgroundMode::COLOR;
    fd.renderer.background.color = m_bgColor;
  }
  fd.renderer.fog = m_fog;
  fd.renderer.ambientColor = m_ambientColor;
  fd.renderer.ambientIntensity = m_ambientIntensity;
  fd.renderer.occlusionDistance = m_occlusionDistance;
  fd.renderer.cullTriangleBF = m_cullTriangleBF;
  fd.renderer.fireflyFilterMode = m_fireflyFilterMode;
  fd.renderer.fireflyFilterSigma = m_fireflyFilterSigma;
  fd.renderer.fireflyFilterWarmup = m_fireflyFilterWarmup;
  fd.renderer.fireflyFilterTrim = m_fireflyFilterTrim;
  fd.renderer.inverseVolumeSamplingRate = 1.f / m_volumeSamplingRate;
  fd.renderer.numIterations = std::max(m_spp, 1);
  fd.renderer.premultiplyBackground = m_premultiplyBackground;
  fd.renderer.cutPlane = m_cutPlane;
}

#ifdef USE_MDL
// The MDL registry is owned by the coordinator thread (ADR 0009); read its
// update timestamp through the coordinator so this render-thread check never
// races a concurrent compile mutating the registry.
static libmdl::TimeStamp mdlLibraryUpdateTime(DeviceGlobalState *state)
{
  auto &mdl = *state->mdl;
  return mdl.coordinator.invokeSerial(
      [&] { return mdl.materialRegistry.getLastUpdateTime(); });
}
#endif

OptixPipeline Renderer::pipeline()
{
#ifndef USE_MDL
  if (!m_pipeline)
#else
  if (!m_pipeline
      || (deviceState()->mdl
          && (mdlLibraryUpdateTime(deviceState())
              > m_lastMDLMaterialLibraryUpdateCheck)))
#endif
    initOptixPipeline();

  return m_pipeline;
}

const OptixShaderBindingTable *Renderer::sbt()
{
  // pipeline() owns version checks. Both accessors are arguments to the same
  // optixLaunch: a registry update during a build must not make this accessor
  // destroy the pipeline handle that the other accessor just returned. The
  // address of m_sbt stays stable if pipeline() is evaluated second instead.
  if (!m_pipeline)
    initOptixPipeline();

  return &m_sbt;
}

int Renderer::spp() const
{
  return m_spp;
}

bool Renderer::checkerboarding() const
{
  return m_checkerboard;
}

bool Renderer::denoise() const
{
  return m_denoise;
}

int Renderer::denoiseStart() const
{
  return m_denoiseStart;
}

bool Renderer::denoiseUsingAlbedo() const
{
  return m_denoiseAlbedo;
}

bool Renderer::denoiseUsingNormal() const
{
  return m_denoiseNormal;
}

int Renderer::sampleLimit() const
{
  return m_sampleLimit;
}

Renderer *Renderer::createInstance(
    std::string_view subtype, DeviceGlobalState *d)
{
  Renderer *retval = nullptr;

  auto *overrideType = getenv("VISRTX_OVERRIDE_RENDERER");

  if (overrideType != nullptr)
    subtype = overrideType;

  retval = make_renderer(subtype, d);

  return retval;
}

void Renderer::initOptixPipeline()
{
  // A rebuild (MDL material-library update) must not strand the previous
  // pipeline, program groups and MDL modules in the driver.
  releasePipeline();

  auto &state = *deviceState();

  auto shadingModule = optixModule();

  char log[2048];
  size_t sizeof_log = sizeof(log);

  // Raygen program //

  {
    m_raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = shadingModule;
    pgDesc.raygen.entryFunctionName = "__raygen__";

    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log,
        &sizeof_log,
        &m_raygenPGs[0]));

    if (sizeof_log > 1)
      reportMessage(ANARI_SEVERITY_DEBUG, "PG Raygen Log:\n%s\n", log);
  }

  // Miss program //

  {
    const auto missNames = missSbtNames();
    m_missPGs.resize(missNames.size());

    int i = 0;
    for (const auto &missName : missNames) {
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      pgDesc.miss.module = shadingModule;
      pgDesc.miss.entryFunctionName = missName.c_str();

      sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
          &pgDesc,
          1,
          &pgOptions,
          log,
          &sizeof_log,
          &m_missPGs[i++]));
    }

    if (sizeof_log > 1)
      reportMessage(ANARI_SEVERITY_DEBUG, "PG Miss Log:\n%s", log);
  }

  // Hit program //

  {
    auto hitgroupNames = hitgroupSbtNames();

    m_hitgroupPGs.resize(
        hitgroupNames.size() * NUM_SBT_PRIMITIVE_INTERSECTOR_ENTRIES);

    int i = 0;
    for (auto &hgn : hitgroupNames) {
      // Triangles
      {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        pgDesc.hitgroup.moduleCH = shadingModule;
        pgDesc.hitgroup.entryFunctionNameCH = hgn.closestHit.c_str();

        if (!hgn.anyHit.empty()) {
          pgDesc.hitgroup.moduleAH = shadingModule;
          pgDesc.hitgroup.entryFunctionNameAH = hgn.anyHit.c_str();
        }

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log,
            &sizeof_log,
            &m_hitgroupPGs[i++]));
        if (sizeof_log > 1) {
          reportMessage(
              ANARI_SEVERITY_DEBUG, "PG Hitgroup Log (Triangles):\n%s", log);
        }
      }

      // Curves
      {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        pgDesc.hitgroup.moduleCH = shadingModule;
        pgDesc.hitgroup.entryFunctionNameCH = hgn.closestHit.c_str();

        if (!hgn.anyHit.empty()) {
          pgDesc.hitgroup.moduleAH = shadingModule;
          pgDesc.hitgroup.entryFunctionNameAH = hgn.anyHit.c_str();
        }

        pgDesc.hitgroup.moduleIS = state.intersectionModules.curveIntersector;
        pgDesc.hitgroup.entryFunctionNameIS = nullptr;

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log,
            &sizeof_log,
            &m_hitgroupPGs[i++]));
        if (sizeof_log > 1) {
          reportMessage(
              ANARI_SEVERITY_DEBUG, "PG Hitgroup Log (Curve):\n%s", log);
        }
      }

      // Custom

      {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = shadingModule;
        pgDesc.hitgroup.entryFunctionNameCH = hgn.closestHit.c_str();

        if (!hgn.anyHit.empty()) {
          pgDesc.hitgroup.moduleAH = shadingModule;
          pgDesc.hitgroup.entryFunctionNameAH = hgn.anyHit.c_str();
        }

        pgDesc.hitgroup.moduleIS = state.intersectionModules.customIntersectors;
        pgDesc.hitgroup.entryFunctionNameIS = "__intersection__";

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log,
            &sizeof_log,
            &m_hitgroupPGs[i++]));

        if (sizeof_log > 1) {
          reportMessage(
              ANARI_SEVERITY_DEBUG, "PG Hitgroup Log (Custom):\n%s", log);
        }
      }

      // Light proxies (ADR 0009)
      //
      // A dedicated slot with its own closest-hit, because a proxy has no
      // Surface, Material or Geometry record and the shading CH dereferences
      // all three. A renderer that does not name a proxy CH gets the inert
      // default, and its rays never set the proxy visibility bits anyway.
      {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = shadingModule;
        pgDesc.hitgroup.entryFunctionNameCH = hgn.lightProxyClosestHit.empty()
            ? hgn.closestHit.c_str()
            : hgn.lightProxyClosestHit.c_str();

        pgDesc.hitgroup.moduleIS = state.intersectionModules.customIntersectors;
        pgDesc.hitgroup.entryFunctionNameIS = "__intersection__lightProxy";

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log,
            &sizeof_log,
            &m_hitgroupPGs[i++]));

        if (sizeof_log > 1) {
          reportMessage(
              ANARI_SEVERITY_DEBUG, "PG Hitgroup Log (Light Proxy):\n%s", log);
        }
      }
    }
  }

  // Callables
  {
    // Reserve space for fixed shaders + samplers before MDL
    constexpr auto FIXED_CALLABLES_COUNT = int(SbtCallableEntryPoints::Last);
    std::vector<OptixProgramGroupDesc> callableDescs(FIXED_CALLABLES_COUNT);

    // Fixed material shaders: Matte, PhysicallyBased
    constexpr auto SBT_CALLABLE_MATTE_OFFSET =
        int(SbtCallableEntryPoints::Matte);
    constexpr auto SBT_CALLABLE_PHYSICALLYBASED_OFFSET =
        int(SbtCallableEntryPoints::PBR);

    OptixProgramGroupDesc callableDesc = {};
    callableDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callableDesc.callables.moduleDC = deviceState()->materialShaders.matte;

    // Matte
    callableDesc.callables.entryFunctionNameDC = "__direct_callable__init";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::Initialize)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC = "__direct_callable__nextRay";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateNextRay)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateTint";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateTint)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateOpacity";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateOpacity)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateEmission";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateEmission)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateTransmission";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateTransmission)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateNormal";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateNormal)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC = "__direct_callable__evalBsdf";

    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvalBsdf)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluatePdf";
    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluatePdf)] = callableDesc;

    // Physically Based
    callableDesc.callables.moduleDC =
        deviceState()->materialShaders.physicallyBased;

    callableDesc.callables.entryFunctionNameDC = "__direct_callable__init";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::Initialize)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC = "__direct_callable__nextRay";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateNextRay)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateTint";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateTint)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateOpacity";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateOpacity)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateEmission";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateEmission)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateTransmission";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateTransmission)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluateNormal";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluateNormal)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC = "__direct_callable__evalBsdf";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvalBsdf)] = callableDesc;

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__evaluatePdf";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::EvaluatePdf)] = callableDesc;

    // Spatial Field Samplers
    OptixProgramGroupDesc samplerDesc = {};
    samplerDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

    // Structured Regular sampler
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_REGULAR_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerRegular);
    samplerDesc.callables.moduleDC = state.fieldSamplers.structuredRegular;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceStructuredRegular";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_REGULAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceStructuredRegular";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_REGULAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeStructuredRegular";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_REGULAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // NanoVDB samplers
    samplerDesc.callables.moduleDC = state.fieldSamplers.nvdb;

    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP4_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp4);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP8_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp8);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP16_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbFp16);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_FPN_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbFpN);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_FLOAT_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbFloat);

    // Fp4
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // Fp8
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // Fp16
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // FpN
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // Float
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // StructuredRectilinear sampler
    samplerDesc.callables.moduleDC = state.fieldSamplers.structuredRectilinear;
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_RECTILINEAR_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerRectilinear);

    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceStructuredRectilinear";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_RECTILINEAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceStructuredRectilinear";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_RECTILINEAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeStructuredRectilinear";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_RECTILINEAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // NanoVDB rectilinear samplers
    samplerDesc.callables.moduleDC = state.fieldSamplers.nvdbRectilinear;

    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP4_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp4);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP8_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp8);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP16_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp16);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FPN_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFpN);
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FLOAT_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFloat);

    // Fp4
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbRectilinearFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbRectilinearFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbRectilinearFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // Fp8
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbRectilinearFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbRectilinearFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbRectilinearFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // Fp16
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbRectilinearFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbRectilinearFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbRectilinearFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // FpN
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbRectilinearFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbRectilinearFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbRectilinearFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // Float
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceNvdbRectilinearFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceNvdbRectilinearFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeNvdbRectilinearFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

    // Custom field sampler (from devices/visrtx)
    // A single callable pair handles all custom field subtypes via type
    // dispatch
    samplerDesc.callables.moduleDC = state.fieldSamplers.customField;

    constexpr auto SBT_CALLABLE_CUSTOM_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerCustom);
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initCustomSampler";
    callableDescs[SBT_CALLABLE_CUSTOM_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleValueCustom";
    callableDescs[SBT_CALLABLE_CUSTOM_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleValue)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNormalCustom";
    callableDescs[SBT_CALLABLE_CUSTOM_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleNormal)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleDistanceCustom";
    callableDescs[SBT_CALLABLE_CUSTOM_OFFSET
        + int(SpatialFieldSamplerEntryPoints::SampleDistance)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__ratioTrackTransmittanceCustom";
    callableDescs[SBT_CALLABLE_CUSTOM_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance)] =
        samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__rayMarchVolumeCustom";
    callableDescs[SBT_CALLABLE_CUSTOM_OFFSET
        + int(SpatialFieldSamplerEntryPoints::RayMarchVolume)] = samplerDesc;

#ifdef USE_MDL
    if (state.mdl) {
      // Copy bytes and their version on the registry's owning thread. Material
      // releases may clear/reuse slots while the modules build below; no span
      // into registry storage may escape this task.
      auto snapshot = state.mdl->coordinator.invokeSerial(
          [&] { return state.mdl->materialRegistry.ptxSnapshot(); });
      auto &ptxBlobs = snapshot.blobs;

      const auto pipelineCompileOptions =
          makeVisRTXOptixPipelineCompileOptions();

      // Phase 1: create each material's OptiX module in parallel on the compile
      // pool -- optixModuleCreate on one context is thread-safe (the built-in
      // renderer modules already build this way). Each task has its own log
      // buffer. The SBT assembly in phase 2 stays serial, so callable order --
      // and thus every material's SBT offset -- is exactly the serial order.
      struct ModuleBuild
      {
        OptixModule module = {};
        OptixResult result = OPTIX_SUCCESS;
        std::string log;
      };
      std::vector<std::future<ModuleBuild>> builds(ptxBlobs.size());
      for (std::size_t i = 0; i < ptxBlobs.size(); ++i) {
        if (ptxBlobs[i].empty())
          continue;
        builds[i] =
            state.mdl->coordinator.submit([optixContext = state.optixContext,
                                              pipelineCompileOptions,
                                              blob = std::move(ptxBlobs[i])] {
              OptixModuleCompileOptions moduleCompileOptions = {};
              moduleCompileOptions.maxRegisterCount =
                  OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
              moduleCompileOptions.optLevel =
                  OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
              moduleCompileOptions.debugLevel =
                  OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
#else
              moduleCompileOptions.optLevel =
                  OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
              moduleCompileOptions.debugLevel =
                  OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
              moduleCompileOptions.numPayloadTypes = 0;
              moduleCompileOptions.payloadTypes = 0;

              ModuleBuild build;
              char log[2048];
              size_t sizeof_log = sizeof(log);
              build.result = optixModuleCreate(optixContext,
                  &moduleCompileOptions,
                  &pipelineCompileOptions,
                  std::data(blob),
                  std::size(blob),
                  log,
                  &sizeof_log,
                  &build.module);
              build.log.assign(log, std::min(sizeof_log, sizeof(log)));
              return build;
            });
      }

      // Phase 2: assemble the SBT in material order.
      for (std::size_t i = 0; i < ptxBlobs.size(); ++i) {
        OptixModule module = {};
        if (builds[i].valid()) {
          auto build = builds[i].get();
          module = build.module;
          if (build.result != OPTIX_SUCCESS)
            reportMessage(ANARI_SEVERITY_ERROR,
                "optixModuleCreate failed for an MDL material: %s",
                build.log.c_str());
        }

        // A null module (empty blob or failed create) falls back to the
        // empty-slot placeholders instead of a dead handle the release path
        // would destroy.
        if (!module) {
          for (auto j = 0; j < int(SurfaceShaderEntryPoints::Count); j++) {
            callableDescs.push_back({});
          }
          continue;
        }
        m_mdlModules.push_back(module);

        OptixProgramGroupDesc callableDesc = {};
        callableDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callableDesc.callables.moduleDC = module;
        const auto mdlBaseOffset = callableDescs.size();

        callableDesc.callables.entryFunctionNameDC = "__direct_callable__init";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__nextRay";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__evaluateTint";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__evaluateOpacity";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__evaluateEmission";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__evaluateTransmission";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__evaluateNormal";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__evalBsdf";
        callableDescs.push_back(callableDesc);

        callableDesc.callables.entryFunctionNameDC =
            "__direct_callable__evaluatePdf";
        callableDescs.push_back(callableDesc);

        // The pushes above are positional: slot N of this block must be the
        // callable SurfaceShaderEntryPoints(N) names. Renaming an entry point
        // without moving its push is otherwise silent -- the wrong callable
        // sits at the right index and the material simply misbehaves.
        assert(callableDescs.size() - mdlBaseOffset
            == size_t(SurfaceShaderEntryPoints::Count));
      }

      // Do not mark changes made during the build as already incorporated.
      m_lastMDLMaterialLibraryUpdateCheck = snapshot.timestamp;
    }
#endif // defined(USE_MDL)

    //
    // Create all program groups (fixed material shaders + samplers + MDL)
    m_materialPGs.resize(size(callableDescs));
    OptixProgramGroupOptions callableOptions = {};
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
        data(callableDescs),
        size(callableDescs),
        &callableOptions,
        log,
        &sizeof_log,
        data(m_materialPGs)));
    if (sizeof_log > 1) {
      reportMessage(ANARI_SEVERITY_DEBUG, "PG Callables Log:\n%s", log);
    }
  }

  // Pipeline //

  {
    auto pipelineCompileOptions = makeVisRTXOptixPipelineCompileOptions();

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 2;

    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : m_raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : m_missPGs)
      programGroups.push_back(pg);
    for (auto pg : m_hitgroupPGs)
      programGroups.push_back(pg);
    for (auto pg : m_materialPGs)
      programGroups.push_back(pg);

    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(state.optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        programGroups.size(),
        log,
        &sizeof_log,
        &m_pipeline));

    if (sizeof_log > 1)
      reportMessage(ANARI_SEVERITY_DEBUG, "Pipeline Create Log:\n%s", log);

    // Handle stack sizes
    OptixStackSizes stackSizes = {};
    for (auto &pg : programGroups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, m_pipeline));
    }
    unsigned int directCallableStackSizeFromTraversal = {};
    unsigned int directCallableStackSizeFromState = {};
    unsigned int continuationStackSize = {};
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
        pipelineLinkOptions
            .maxTraceDepth, // Reuse pipeline configured trace depth.
        0, // We don't rely on continuation, but direct calls
        2, // TBC if 2 is enough
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState,
        &continuationStackSize));
    OPTIX_CHECK(optixPipelineSetStackSize(m_pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        2) // expected for single ias graphs, as per our case.
    );
  }

  // SBT //
  {
    std::vector<RaygenRecord> raygenRecords;
    for (auto &pg : m_raygenPGs) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
      raygenRecords.push_back(rec);
    }
    m_raygenRecordsBuffer.upload(raygenRecords);
    m_sbt.raygenRecord = (CUdeviceptr)m_raygenRecordsBuffer.ptr();

    std::vector<MissRecord> missRecords;
    for (auto &pg : m_missPGs) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
      missRecords.push_back(rec);
    }
    m_missRecordsBuffer.upload(missRecords);
    m_sbt.missRecordBase = (CUdeviceptr)m_missRecordsBuffer.ptr();
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = missRecords.size();

    std::vector<HitgroupRecord> hitgroupRecords;
    for (auto &hpg : m_hitgroupPGs) {
      HitgroupRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(hpg, &rec));
      hitgroupRecords.push_back(rec);
    }
    m_hitgroupRecordsBuffer.upload(hitgroupRecords);
    m_sbt.hitgroupRecordBase = (CUdeviceptr)m_hitgroupRecordsBuffer.ptr();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = hitgroupRecords.size();

    std::vector<MaterialRecord> materialRecords;
    for (auto &mpg : m_materialPGs) {
      MaterialRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(mpg, &rec));
      materialRecords.push_back(rec);
    }

    m_materialRecordsBuffer.upload(materialRecords);
    m_sbt.callablesRecordBase = (CUdeviceptr)m_materialRecordsBuffer.ptr();
    m_sbt.callablesRecordStrideInBytes = sizeof(MaterialRecord);
    m_sbt.callablesRecordCount = materialRecords.size();
  }
}

OptixPipelineCompileOptions makeVisRTXOptixPipelineCompileOptions()
{
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  pipelineCompileOptions.usesPrimitiveTypeFlags =
      OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM
      | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
  pipelineCompileOptions.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipelineCompileOptions.usesMotionBlur = false;
  pipelineCompileOptions.numPayloadValues = PAYLOAD_VALUES;
  pipelineCompileOptions.numAttributeValues = ATTRIBUTE_VALUES;
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "frameData";

  return pipelineCompileOptions;
}

void Renderer::releasePipeline()
{
  // Reverse creation order: pipeline, then the program groups it linked, then
  // the modules the callables were built from.
  if (m_pipeline) {
    // Launches are async on the shared device stream, and a REBUILD can be
    // triggered by one Frame while another Frame's launch still runs this
    // pipeline (each Frame waits only on its own completion event). OptiX
    // forbids destroying in-use objects — drain the stream first; only the
    // rebuild/teardown path pays.
    cudaStreamSynchronize(deviceState()->stream);
    optixPipelineDestroy(m_pipeline);
    m_pipeline = nullptr;
  }
  auto destroyPGs = [](std::vector<OptixProgramGroup> &pgs) {
    for (auto &pg : pgs) {
      if (pg)
        optixProgramGroupDestroy(pg);
    }
    pgs.clear();
  };
  destroyPGs(m_raygenPGs);
  destroyPGs(m_missPGs);
  destroyPGs(m_hitgroupPGs);
  destroyPGs(m_materialPGs);
  for (auto &module : m_mdlModules) {
    if (module)
      optixModuleDestroy(module);
  }
  m_mdlModules.clear();
}

void Renderer::cleanup()
{
  if (m_backgroundTexture) {
    cudaDestroyTextureObject(m_backgroundTexture);
    m_backgroundTexture = {};
  }
}

FireflyFilterMode Renderer::fireflyFilterMode() const
{
  return m_fireflyFilterMode;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Renderer *);
