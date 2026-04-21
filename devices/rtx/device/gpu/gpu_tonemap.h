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

// Tonemap helpers — safe to include from both PTX and regular CUDA sources.
// gpu_util.h includes <optix_device.h> and cannot be used from Frame.cu;
// this header provides the subset needed by the compositing kernel.
#pragma once

#include "gpu_math.h"
// glm
#include <glm/gtc/color_space.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/packing.hpp>

namespace visrtx {
namespace detail {

VISRTX_DEVICE vec3 tonemap(vec3 v)
{
  return v / (1.0f + glm::max(0.0f, compMax(v)));
}

VISRTX_DEVICE vec3 inverseTonemap(vec3 v)
{
  return v / glm::max(1e-12f, 1.f - compMax(v));
}

VISRTX_DEVICE vec4 tonemap(vec4 v)
{
  return vec4(tonemap(vec3(v)), v.w);
}

VISRTX_DEVICE vec4 inverseTonemap(vec4 v)
{
  return vec4(inverseTonemap(vec3(v)), v.w);
}

} // namespace detail
} // namespace visrtx
