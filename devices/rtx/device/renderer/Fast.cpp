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

#include "Fast.h"
// ptx
#include "Fast_ptx.h"

namespace visrtx {

static const std::vector<HitgroupFunctionNames> g_fastHitNames = {
    {"__closesthit__primary", "__anyhit__primary"},
    {"__closesthit__shadow", "__anyhit__shadow"}};

static const std::vector<std::string> g_fastMissNames = {"__miss__", "__miss__"};

Fast::Fast(DeviceGlobalState *s) : Renderer(s, 1.f) {}

void Fast::commitParameters()
{
  Renderer::commitParameters();
  m_aoSamples = std::clamp(getParam<int>("ambientSamples", 1), 0, 256);
}

void Fast::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  fd.renderer.params.fast.aoSamples = m_aoSamples;
}

OptixModule Fast::optixModule() const
{
  return deviceState()->rendererModules.fast;
}

Span<HitgroupFunctionNames> Fast::hitgroupSbtNames() const
{
  return make_Span(g_fastHitNames.data(), g_fastHitNames.size());
}

Span<std::string> Fast::missSbtNames() const
{
  return make_Span(g_fastMissNames.data(), g_fastMissNames.size());
}

ptx_blob Fast::ptx()
{
  return {Fast_ptx, sizeof(Fast_ptx)};
}

} // namespace visrtx
