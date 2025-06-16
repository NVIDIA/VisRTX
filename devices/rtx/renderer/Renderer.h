/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "array/Array2D.h"
#include "gpu/gpu_objects.h"
#include "utility/CudaImageTexture.h"
// optix
#include <helium/utility/TimeStamp.h>
#include <optix.h>
// std
#include <vector>
// anari
#include "utility/Span.h"

namespace visrtx {

struct HitgroupFunctionNames
{
  std::string closestHit{"__closesthit__"};
  std::string anyHit;
};

struct Renderer : public Object
{
  Renderer(DeviceGlobalState *s, float defaultAmbientRadiance = 0.f);
  ~Renderer() override;

  virtual void commitParameters() override;
  virtual void finalize() override;

  virtual OptixModule optixModule() const = 0;

  virtual Span<HitgroupFunctionNames> hitgroupSbtNames() const;
  virtual Span<std::string> missSbtNames() const;

  virtual void populateFrameData(FrameGPUData &fd) const;

  OptixPipeline pipeline();
  const OptixShaderBindingTable *sbt();

  int spp() const;
  bool checkerboarding() const;
  bool denoise() const;
  int sampleLimit() const;

  static Renderer *createInstance(
      std::string_view subtype, DeviceGlobalState *d);

 protected:
  vec4 m_bgColor{0.f, 0.f, 0.f, 1.f};
  int m_spp{1};
  int m_maxRayDepth{0};
  vec3 m_ambientColor{1.f};
  float m_ambientIntensity{0.f};
  float m_occlusionDistance{1e20f};
  bool m_checkerboard{false};
  bool m_denoise{false};
  int m_sampleLimit{0};
  bool m_cullTriangleBF{false};
  float m_volumeSamplingRate{1.f};

  helium::ChangeObserverPtr<Array2D> m_backgroundImage;
  cudaTextureObject_t m_backgroundTexture{};

  // OptiX //

  OptixPipeline m_pipeline{nullptr};

  std::vector<OptixProgramGroup> m_raygenPGs;
  std::vector<OptixProgramGroup> m_missPGs;
  std::vector<OptixProgramGroup> m_hitgroupPGs;
  std::vector<OptixProgramGroup> m_materialPGs;
  DeviceBuffer m_raygenRecordsBuffer;
  DeviceBuffer m_missRecordsBuffer;
  DeviceBuffer m_hitgroupRecordsBuffer;
  DeviceBuffer m_materialRecordsBuffer;
  OptixShaderBindingTable m_sbt{};

 private:
  void initOptixPipeline();
  void cleanup();

  HitgroupFunctionNames m_defaultHitgroupNames;
  std::string m_defaultMissName{"__miss__"};
  float m_defaultAmbientRadiance{0.f};

#ifdef USE_MDL
  helium::TimeStamp m_lastMDLMaterialLibraryUpdateCheck{};
#endif // defined(USE_MDL)
};

OptixPipelineCompileOptions makeVisRTXOptixPipelineCompileOptions();

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Renderer *, ANARI_RENDERER);
