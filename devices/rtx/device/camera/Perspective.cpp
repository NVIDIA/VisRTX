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

#include "Perspective.h"

#include <glm/trigonometric.hpp>

namespace visrtx {

Perspective::Perspective(DeviceGlobalState *s) : Camera(s) {}

void Perspective::commitParameters()
{
  auto &hd = data();
  readBaseParameters(hd);
  hd.type = CameraType::PERSPECTIVE;

  const float fovy = getParam<float>("fovy", glm::radians(60.f));
  const float aspect = getParam<float>("aspect", 1.f);

  vec2 imgPlaneSize;
  imgPlaneSize.y = 2.f * tanf(0.5f * fovy);
  imgPlaneSize.x = imgPlaneSize.y * aspect;

  vec3 dir_du = normalize(cross(hd.dir, hd.up)) * imgPlaneSize.x;
  vec3 dir_dv = normalize(cross(dir_du, hd.dir)) * imgPlaneSize.y;
  vec3 dir_00 = hd.dir - .5f * dir_du - .5f * dir_dv;

  const float focusDistance = getParam<float>("focusDistance", 1.f);
  const float apertureRadius =
      getParam<float>("apertureRadius", 0.f) / (imgPlaneSize.x * focusDistance);
  if (apertureRadius > 0.f) {
    dir_du *= focusDistance;
    dir_dv *= focusDistance;
    dir_00 *= focusDistance;
  }

  auto &p = hd.perspective;
  p.dir_du = dir_du;
  p.dir_dv = dir_dv;
  p.dir_00 = dir_00;
  p.scaledAperture = apertureRadius;
  p.aspect = aspect;
}

} // namespace visrtx
