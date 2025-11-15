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

#include "DirectLight.h"
// ptx
#include "DirectLight_ptx.h"

namespace visrtx {

static const std::vector<HitgroupFunctionNames> g_directLightHitNames = {
    {"__closesthit__shading", "__anyhit__shading"},
    {"__closesthit__shadow", "__anyhit__shadow"},
};

static const std::vector<std::string> g_directLightMissNames = {
    "__miss__shading", "__miss__shadow"
};

DirectLight::DirectLight(DeviceGlobalState *s) : Renderer(s, 0.f) {}

void DirectLight::commitParameters()
{
  Renderer::commitParameters();
  m_lightFalloff = std::clamp(getParam<float>("lightFalloff", 1.f), 0.f, 1.f);
  m_aoSamples = std::clamp(getParam<int>("ambientSamples", 1), 0, 256);
  m_volumeSamplingRateShadows = std::clamp(
      getParam<float>("volumeSamplingRateShadows", 0.0125f), 1e-4f, 10.f);
}

void DirectLight::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  auto &directLight = fd.renderer.params.directLight;
  directLight.lightFalloff = m_lightFalloff;
  directLight.aoSamples = m_aoSamples;
  directLight.aoColor = m_aoColor;
  directLight.aoIntensity = m_aoIntensity;
  directLight.inverseVolumeSamplingRateShadows =
      1.f / m_volumeSamplingRateShadows;
}

OptixModule DirectLight::optixModule() const
{
  return deviceState()->rendererModules.directLight;
}

Span<HitgroupFunctionNames> DirectLight::hitgroupSbtNames() const
{
  return make_Span(g_directLightHitNames.data(), g_directLightHitNames.size());
}

Span<std::string> DirectLight::missSbtNames() const
{
  return make_Span(
      g_directLightMissNames.data(), g_directLightMissNames.size());
}

ptx_blob DirectLight::ptx()
{
  return {DirectLight_ptx, sizeof(DirectLight_ptx)};
}

} // namespace visrtx
