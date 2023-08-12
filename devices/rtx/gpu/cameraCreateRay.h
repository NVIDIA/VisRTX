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

#include "gpu/gpu_objects.h"

namespace visrtx {

RT_FUNCTION Ray cameraCreateRay(const CameraGPUData *c, vec2 screen)
{
  Ray ray;

  screen.x = glm::mix(c->region[0], c->region[2], screen.x);
  screen.y = glm::mix(c->region[1], c->region[3], screen.y);

  switch (c->type) {
  case CameraType::PERSPECTIVE: {
    const auto &p = c->perspective;
    ray.org = c->pos;
    ray.dir = normalize(p.dir_00 + screen.x * p.dir_du + screen.y * p.dir_dv);
    break;
  }
  case CameraType::ORTHOGRAPHIC: {
    const auto &o = c->orthographic;
    ray.dir = c->dir;
    ray.org = o.pos_00 + screen.x * o.pos_du + screen.y * o.pos_dv;
    break;
  }
  default:
    break;
  }

  return ray;
}

RT_FUNCTION Ray makePrimaryRay(ScreenSample &ss)
{
  const vec2 r(curand_uniform(&ss.rs) - 0.5f, curand_uniform(&ss.rs) - 0.5f);
  ss.screen =
      vec2(ss.pixel.x + r.x, ss.pixel.y + r.y) * ss.frameData->fb.invSize;
  return cameraCreateRay(ss.frameData->camera, ss.screen);
}

} // namespace visrtx
