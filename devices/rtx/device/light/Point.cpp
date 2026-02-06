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

#include "Point.h"

namespace visrtx {

Point::Point(DeviceGlobalState *d) : Light(d) {}

void Point::commitParameters()
{
  Light::commitParameters();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_intensity =
      std::clamp(getParam<float>("intensity", getParam<float>("power", 1.f)),
          0.f,
          std::numeric_limits<float>::max());
  m_radius = std::max(getParam<float>("radius", 1.f), 0.f);
}

LightGPUData Point::gpuData() const
{
  auto retval = Light::gpuData();
  if (m_radius <= 0.0f) {
    retval.type = LightType::POINT;
    retval.point.position = m_position;
    retval.point.intensity = m_intensity;
  } else {
    retval.type = LightType::SPHERE;
    retval.sphere.position = m_position;
    retval.sphere.intensity = m_intensity;
    retval.sphere.radius = m_radius;
    retval.sphere.oneOverArea = 1.0f / (4.0f * float(M_PI) * m_radius * m_radius);
  }
  return retval;
}

} // namespace visrtx
