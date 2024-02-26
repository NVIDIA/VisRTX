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

#include "Raycast.h"
// ptx
#include "Raycast_ptx.h"

namespace visrtx {

static const std::vector<HitgroupFunctionNames> g_aoHitNames = {
    {"__closesthit__primary", "__anyhit__primary"}};

static const std::vector<std::string> g_aoMissNames = {"__miss__"};

Raycast::Raycast(DeviceGlobalState *s) : Renderer(s) {}

OptixModule Raycast::optixModule() const
{
  return deviceState()->rendererModules.raycast;
}

Span<HitgroupFunctionNames> Raycast::hitgroupSbtNames() const
{
  return make_Span(g_aoHitNames.data(), g_aoHitNames.size());
}

Span<std::string> Raycast::missSbtNames() const
{
  return make_Span(g_aoMissNames.data(), g_aoMissNames.size());
}

ptx_blob Raycast::ptx()
{
  return {Raycast_ptx, sizeof(Raycast_ptx)};
}

} // namespace visrtx
