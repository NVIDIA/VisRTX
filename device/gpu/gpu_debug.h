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

#ifndef VISRTX_DEBUGGING
#define VISRTX_DEBUGGING 0
#endif

#define VISRTX_DEBUG_PIXEL                                                     \
  (glm::uvec2(                                                                 \
      optixGetLaunchDimensions().x / 2, optixGetLaunchDimensions().y / 2))

#include "gpu/gpu_objects.h"

namespace visrtx {

RT_FUNCTION bool debug()
{
#if VISRTX_DEBUGGING
  glm::uvec2 pixel(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
  return pixel == VISRTX_DEBUG_PIXEL;
#else
  return false;
#endif
}

RT_FUNCTION bool crosshair()
{
  glm::uvec2 pixel(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
  return VISRTX_DEBUGGING
      && (VISRTX_DEBUG_PIXEL.x == pixel.x || VISRTX_DEBUG_PIXEL.y == pixel.y);
}

} // namespace visrtx
