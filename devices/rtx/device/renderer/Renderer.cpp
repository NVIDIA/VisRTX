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
#include "AmbientOcclusion.h"
#include "Debug.h"
#include "DirectLight.h"
#include "PathTracer.h"
#include "Raycast.h"
#include "Test.h"
#include "UnknownRenderer.h"

#include "gpu/gpu_decl.h"
#include "gpu/shadingState.h"

// std
#include <optix_types.h>
#include <stdlib.h>
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

  if (subtype == "raycast")
    return new Raycast(d);
  else if (subtype == "ao")
    return new AmbientOcclusion(d);
  else if (subtype == "pathTracer" || subtype == "pt")
    return new PathTracer(d);
  else if (subtype == "directLight" || subtype == "default")
    return new DirectLight(d);
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
  optixPipelineDestroy(m_pipeline);
}

void Renderer::commitParameters()
{
  m_backgroundImage = getParamObject<Array2D>("background");
  m_bgColor = getParam<vec4>("background", vec4(vec3(0.f), 1.f));
  m_spp = getParam<int>("pixelSamples", 1);
  m_maxRayDepth = getParam<int>("maxRayDepth", 5);
  m_ambientColor = getParam<vec3>("ambientColor", vec3(1.f));
  m_ambientIntensity =
      getParam<float>("ambientRadiance", m_defaultAmbientRadiance);
  m_occlusionDistance = getParam<float>("ambientOcclusionDistance", 1e20f);
  m_checkerboard = getParam<bool>("checkerboarding", false);

  m_denoise = getParam<bool>("denoise", false);
  auto denoiseMode = getParamString("denoiseMode", "color");
  m_denoiseAlbedo =
      (denoiseMode == "colorAlbedo" || denoiseMode == "colorAlbedoNormal");
  m_denoiseNormal = (denoiseMode == "colorAlbedoNormal");

  m_tonemap = getParam<bool>("tonemap", true);
  m_sampleLimit = getParam<int>("sampleLimit", 128);
  m_cullTriangleBF = getParam<bool>("cullTriangleBackfaces", false);
  m_volumeSamplingRate =
      std::clamp(getParam<float>("volumeSamplingRate", 0.125f), 1e-3f, 10.f);
  m_premultiplyBackground = getParam<bool>("premultiplyBackground", false);
  if (m_checkerboard)
    m_spp = 1;
}

void Renderer::finalize()
{
  cleanup();
  if (m_backgroundImage) {
    auto cuArray = m_backgroundImage->acquireCUDAArrayUint8();
    m_backgroundTexture = makeCudaTextureObject2D(cuArray, true, "linear");
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
  if (m_backgroundImage) {
    fd.renderer.backgroundMode = BackgroundMode::IMAGE;
    fd.renderer.background.texobj = m_backgroundTexture;
  } else {
    fd.renderer.backgroundMode = BackgroundMode::COLOR;
    fd.renderer.background.color = m_bgColor;
  }
  fd.renderer.ambientColor = m_ambientColor;
  fd.renderer.ambientIntensity = m_ambientIntensity;
  fd.renderer.occlusionDistance = m_occlusionDistance;
  fd.renderer.cullTriangleBF = m_cullTriangleBF;
  fd.renderer.tonemap = m_tonemap;
  fd.renderer.inverseVolumeSamplingRate = 1.f / m_volumeSamplingRate;
  fd.renderer.numIterations = std::max(m_spp, 1);
  fd.renderer.maxRayDepth = m_maxRayDepth;
  fd.renderer.premultiplyBackground = m_premultiplyBackground;
}

OptixPipeline Renderer::pipeline()
{
#ifndef USE_MDL
  if (!m_pipeline)
#else
  if (!m_pipeline
      || (deviceState()->mdl
          && (deviceState()->mdl->materialRegistry.getLastUpdateTime()
              > m_lastMDLMaterialLibraryUpdateCheck)))
#endif
    initOptixPipeline();

  return m_pipeline;
}

const OptixShaderBindingTable *Renderer::sbt()
{
#ifndef USE_MDL
  if (!m_pipeline)
#else
  if (!m_pipeline
      || (deviceState()->mdl
          && (deviceState()->mdl->materialRegistry.getLastUpdateTime()
              > m_lastMDLMaterialLibraryUpdateCheck)))
#endif

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

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__shadeSurface";

    callableDescs[SBT_CALLABLE_MATTE_OFFSET
        + int(SurfaceShaderEntryPoints::Shade)] = callableDesc;

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

    callableDesc.callables.entryFunctionNameDC =
        "__direct_callable__shadeSurface";
    callableDescs[SBT_CALLABLE_PHYSICALLYBASED_OFFSET
        + int(SurfaceShaderEntryPoints::Shade)] = callableDesc;

    // Spatial Field Samplers
    OptixProgramGroupDesc samplerDesc = {};
    samplerDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

    // Structured Regular sampler
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_REGULAR_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerRegular);
    samplerDesc.callables.moduleDC = state.fieldSamplers.structuredRegular;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initStructuredRegularSampler";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_REGULAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleStructuredRegular";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_REGULAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

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
        "__direct_callable__initNvdbSamplerFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // Fp8
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbSamplerFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // Fp16
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbSamplerFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // FpN
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbSamplerFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // Float
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbSamplerFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // StructuredRectilinear sampler
    samplerDesc.callables.moduleDC = state.fieldSamplers.structuredRectilinear;
    constexpr auto SBT_CALLABLE_SPATIAL_FIELD_RECTILINEAR_OFFSET =
        int(SbtCallableEntryPoints::SpatialFieldSamplerRectilinear);

    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initStructuredRectilinearSampler";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_RECTILINEAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleStructuredRectilinear";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_RECTILINEAR_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

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
        "__direct_callable__initNvdbRectilinearSamplerFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbRectilinearFp4";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP4_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // Fp8
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbRectilinearSamplerFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbRectilinearFp8";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP8_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // Fp16
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbRectilinearSamplerFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbRectilinearFp16";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FP16_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // FpN
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbRectilinearSamplerFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbRectilinearFpN";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FPN_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

    // Float
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__initNvdbRectilinearSamplerFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Init)] = samplerDesc;
    samplerDesc.callables.entryFunctionNameDC =
        "__direct_callable__sampleNvdbRectilinearFloat";
    callableDescs[SBT_CALLABLE_SPATIAL_FIELD_NVDB_REC_FLOAT_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

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
        "__direct_callable__sampleCustom";
    callableDescs[SBT_CALLABLE_CUSTOM_OFFSET
        + int(SpatialFieldSamplerEntryPoints::Sample)] = samplerDesc;

#ifdef USE_MDL
    if (state.mdl) {
      for (const auto &ptxBlob : state.mdl->materialRegistry.getPtxBlobs()) {
        if (ptxBlob.empty()) {
          for (auto i = 0; i < int(SurfaceShaderEntryPoints::Count); i++) {
            callableDescs.push_back({});
          }
          continue;
        }
        OptixModule module;
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount =
            OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
#else
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel =
            OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; // Could be FULL is -G can be
                                               // enabled at compile time
#endif
        moduleCompileOptions.numPayloadTypes = 0;
        moduleCompileOptions.payloadTypes = 0;

        auto pipelineCompileOptions = makeVisRTXOptixPipelineCompileOptions();

        OPTIX_CHECK(optixModuleCreate(state.optixContext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            std::data(ptxBlob),
            std::size(ptxBlob),
            log,
            &sizeof_log,
            &module));

        OptixProgramGroupDesc callableDesc = {};
        callableDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callableDesc.callables.moduleDC = module;
        auto mdlBaseOffset = callableDescs.size();

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
            "__direct_callable__shadeSurface";
        callableDescs.push_back(callableDesc);
      }

      m_lastMDLMaterialLibraryUpdateCheck =
          deviceState()->mdl->materialRegistry.getLastUpdateTime();
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

void Renderer::cleanup()
{
  if (m_backgroundImage) {
    if (m_backgroundTexture) {
      cudaDestroyTextureObject(m_backgroundTexture);
      m_backgroundImage->releaseCUDAArrayUint8();
    }
  }
}

bool Renderer::tonemap() const
{
  return m_tonemap;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Renderer *);
