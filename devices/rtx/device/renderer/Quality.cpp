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

#include "Quality.h"
// ptx
#include "Quality_ptx.h"
// std
#include <array>

namespace visrtx {

static const std::array<HitgroupFunctionNames, 2> g_qualityHitNames = {
    HitgroupFunctionNames{"__closesthit__shading", "__anyhit__shading"},
    HitgroupFunctionNames{"__closesthit__shadow", "__anyhit__shadow"}};

static const auto g_qualityMissNames =
    std::array<std::string, 2>{"__miss__shading", "__miss__shadow"};

Quality::Quality(DeviceGlobalState *s) : Renderer(s) {}

void Quality::commitParameters()
{
  Renderer::commitParameters();
  m_maxRayDepth = std::max(getParam<int>("maxRayDepth", 5), 1);
  m_maxTransparencyDepth = std::max(getParam<int>("maxTransparencyDepth", 32), 0);
}

void Quality::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  fd.renderer.params.quality.maxRayDepth = m_maxRayDepth;
  fd.renderer.params.quality.maxTransparencyDepth = m_maxTransparencyDepth;
}

OptixModule Quality::optixModule() const
{
  return deviceState()->rendererModules.quality;
}

Span<HitgroupFunctionNames> Quality::hitgroupSbtNames() const
{
  return make_Span(g_qualityHitNames.data(), g_qualityHitNames.size());
}

Span<std::string> Quality::missSbtNames() const
{
  return make_Span(g_qualityMissNames.data(), g_qualityMissNames.size());
}

ptx_blob Quality::ptx()
{
  return {Quality_ptx, sizeof(Quality_ptx)};
}

} // namespace visrtx
