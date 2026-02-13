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

#include "Interactive.h"
// ptx
#include "Interactive_ptx.h"

namespace visrtx {

static const std::vector<HitgroupFunctionNames> g_interactiveHitNames = {
    {"__closesthit__shading", "__anyhit__shading"},
    {"__closesthit__shadow", "__anyhit__shadow"},
};

static const std::vector<std::string> g_interactiveMissNames = {
    "__miss__shading", "__miss__shadow"
};

Interactive::Interactive(DeviceGlobalState *s) : Renderer(s, 0.f) {}

void Interactive::commitParameters()
{
  Renderer::commitParameters();
  m_lightFalloff = std::clamp(getParam<float>("lightFalloff", 1.f), 0.f, 1.f);
  m_aoSamples = std::clamp(getParam<int>("ambientSamples", 1), 0, 256);
  m_volumeSamplingRateShadows = std::clamp(
      getParam<float>("volumeSamplingRateShadows", 0.0125f), 1e-4f, 10.f);
}

void Interactive::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  auto &interactive = fd.renderer.params.interactive;
  interactive.lightFalloff = m_lightFalloff;
  interactive.aoSamples = m_aoSamples;
  interactive.aoColor = m_aoColor;
  interactive.aoIntensity = m_aoIntensity;
  interactive.inverseVolumeSamplingRateShadows =
      1.f / m_volumeSamplingRateShadows;
}

OptixModule Interactive::optixModule() const
{
  return deviceState()->rendererModules.interactive;
}

Span<HitgroupFunctionNames> Interactive::hitgroupSbtNames() const
{
  return make_Span(g_interactiveHitNames.data(), g_interactiveHitNames.size());
}

Span<std::string> Interactive::missSbtNames() const
{
  return make_Span(
      g_interactiveMissNames.data(), g_interactiveMissNames.size());
}

ptx_blob Interactive::ptx()
{
  return {Interactive_ptx, sizeof(Interactive_ptx)};
}

} // namespace visrtx
