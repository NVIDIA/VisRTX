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

#pragma once

#include "Object.h"
#include "gpu/gpu_objects.h"
// optix
#include <optix.h>
// std
#include <vector>
// anari
#include "anari/backend/utilities/Span.h"

namespace visrtx {

struct HitgroupFunctionNames
{
  std::string closestHit{"__closesthit__"};
  std::string anyHit;
};

struct Renderer : public Object
{
  static size_t objectCount();

  Renderer(DeviceGlobalState *s);
  ~Renderer() override;

  virtual void commit() override;

  virtual OptixModule optixModule() const = 0;

  virtual anari::Span<const HitgroupFunctionNames> hitgroupSbtNames() const;
  virtual anari::Span<const std::string> missSbtNames() const;

  virtual void populateFrameData(FrameGPUData &fd) const;

  OptixPipeline pipeline() const;
  const OptixShaderBindingTable *sbt();

  vec4 bgColor() const;
  int spp() const;
  vec3 ambientColor() const;
  float ambientIntensity() const;
  float ambientOcclusionDistance() const;

  static Renderer *createInstance(
      std::string_view subtype, DeviceGlobalState *d);

 protected:
  vec4 m_bgColor{1.f};
  int m_spp{1};
  vec3 m_ambientColor{1.f};
  float m_ambientIntensity{1.f};
  float m_occlusionDistance{1e20f};

  // OptiX //

  OptixPipeline m_pipeline{nullptr};

  std::vector<OptixProgramGroup> m_raygenPGs;
  std::vector<OptixProgramGroup> m_missPGs;
  std::vector<OptixProgramGroup> m_hitgroupPGs;
  DeviceBuffer m_raygenRecordsBuffer;
  DeviceBuffer m_missRecordsBuffer;
  DeviceBuffer m_hitgroupRecordsBuffer;
  OptixShaderBindingTable m_sbt{};

 private:
  void initOptixPipeline();

  HitgroupFunctionNames m_defaultHitgroupNames;
  std::string m_defaultMissName{"__miss__"};
};

OptixPipelineCompileOptions makeVisRTXOptixPipelineCompileOptions();

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Renderer *, ANARI_RENDERER);
