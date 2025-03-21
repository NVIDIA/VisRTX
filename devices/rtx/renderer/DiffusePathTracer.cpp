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

#include "DiffusePathTracer.h"
// ptx
#include "DiffusePathTracer_ptx.h"

namespace visrtx {

static const std::vector<HitgroupFunctionNames> g_dptHitNames = {
    {"__closesthit__", "__anyhit__"}};

DiffusePathTracer::DiffusePathTracer(DeviceGlobalState *s) : Renderer(s, 1.f) {}

void DiffusePathTracer::commitParameters()
{
  Renderer::commitParameters();
  m_maxDepth = std::clamp(getParam<int>("maxDepth", 5), 1, 256);
}

void DiffusePathTracer::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  fd.renderer.params.dpt.maxDepth = m_maxDepth;
}

OptixModule DiffusePathTracer::optixModule() const
{
  return deviceState()->rendererModules.diffusePathTracer;
}

Span<HitgroupFunctionNames> DiffusePathTracer::hitgroupSbtNames() const
{
  return make_Span(g_dptHitNames.data(), g_dptHitNames.size());
}

ptx_blob DiffusePathTracer::ptx()
{
  return {DiffusePathTracer_ptx, sizeof(DiffusePathTracer_ptx)};
}

} // namespace visrtx
