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

#include "Renderer.h"
// specific renderers
#include "AmbientOcclusion.h"
#include "Debug.h"
#include "DiffusePathTracer.h"
#include "Raycast.h"
#include "SciVis.h"
#include "Test.h"
#include "UnknownRenderer.h"
// std
#include <stdlib.h>
#include <string_view>
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

namespace visrtx {

struct SBTRecord
{
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

using RaygenRecord = SBTRecord;
using MissRecord = SBTRecord;
using HitgroupRecord = SBTRecord;

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
  else if (subtype == "diffuse_pathtracer" || subtype == "dpt")
    return new DiffusePathTracer(d);
  else if (subtype == "scivis" || subtype == "sv" || subtype == "default")
    return new SciVis(d);
  else if (subtype == "test")
    return new Test(d);
  else if (beginsWith(subtype, "debug")) {
    auto *retval = new Debug(d);
    auto names = splitString(std::string(subtype), "_");
    if (names.size() > 1)
      retval->setParam("method", ANARI_STRING, names[1].c_str());
    return retval;
  } else
    return new UnknownRenderer(d, subtype);
}

// Renderer definitions ///////////////////////////////////////////////////////

static size_t s_numRenderers = 0;

size_t Renderer::objectCount()
{
  return s_numRenderers;
}

Renderer::Renderer(DeviceGlobalState *s) : Object(ANARI_RENDERER, s)
{
  s_numRenderers++;
}

Renderer::~Renderer()
{
  m_backgroundTexture.cleanup();
  optixPipelineDestroy(m_pipeline);
  s_numRenderers--;
}

void Renderer::commit()
{
  m_backgroundTexture.cleanup();
  m_backgroundImage = getParamObject<Array2D>("background");
  if (m_backgroundImage) {
    m_backgroundTexture = makeCudaTextureFloat(
        *m_backgroundImage, m_backgroundImage->size(), "linear");
  }

  m_bgColor = getParam<vec4>("background", vec4(1.f));
  m_spp = getParam<int>("pixelSamples", 1);
  m_ambientColor = getParam<vec3>("ambientColor", vec3(1.f));
  m_ambientIntensity = getParam<float>("ambientIntensity", 1.f);
  m_occlusionDistance = getParam<float>("ambientOcclusionDistance", 1e20f);
}

anari::Span<const HitgroupFunctionNames> Renderer::hitgroupSbtNames() const
{
  return anari::make_Span(&m_defaultHitgroupNames, 1);
}

anari::Span<const std::string> Renderer::missSbtNames() const
{
  return anari::make_Span(&m_defaultMissName, 1);
}

void Renderer::populateFrameData(FrameGPUData &fd) const
{
  if (m_backgroundImage) {
    fd.renderer.backgroundMode = BackgroundMode::IMAGE;
    fd.renderer.background.texobj = m_backgroundTexture.cuObject;
  } else {
    fd.renderer.backgroundMode = BackgroundMode::COLOR;
    fd.renderer.background.color = bgColor();
  }
  fd.renderer.ambientColor = ambientColor();
  fd.renderer.ambientIntensity = ambientIntensity();
  fd.renderer.occlusionDistance = ambientOcclusionDistance();
}

OptixPipeline Renderer::pipeline() const
{
  return m_pipeline;
}

const OptixShaderBindingTable *Renderer::sbt()
{
  if (!m_pipeline)
    initOptixPipeline();

  return &m_sbt;
}

vec4 Renderer::bgColor() const
{
  return m_bgColor;
}

int Renderer::spp() const
{
  return m_spp;
}

vec3 Renderer::ambientColor() const
{
  return m_ambientColor;
}

float Renderer::ambientIntensity() const
{
  return m_ambientIntensity;
}

float Renderer::ambientOcclusionDistance() const
{
  return m_occlusionDistance;
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
    m_missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = shadingModule;
    pgDesc.miss.entryFunctionName = "__miss__";

    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log,
        &sizeof_log,
        &m_missPGs[0]));

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

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Renderer *);
