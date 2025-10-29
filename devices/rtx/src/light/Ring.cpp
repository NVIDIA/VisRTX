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

#include "Ring.h"

namespace visrtx {

Ring::Ring(DeviceGlobalState *d) : Light(d) {}

void Ring::commitParameters()
{
  Light::commitParameters();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_direction = getParam<vec3>("direction", vec3(0.f, 0.f, -1.f));
  m_openingAngle = getParam<float>("openingAngle", M_PI);
  m_falloffAngle = getParam<float>("falloffAngle", 0.1f);
  m_radius = std::max(getParam<float>("radius", 0.f), 0.f);
  m_innerRadius = std::max(getParam<float>("innerRadius", 0.f), 0.f);
  m_intensity = std::clamp(getParam<float>("intensity", 1.f),
      0.f,
      std::numeric_limits<float>::max());

  // Validate parameters
  if (m_innerRadius >= m_radius && m_radius > 0.f) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "innerRadius must be smaller than radius");
    m_innerRadius = std::max(0.f, m_radius - 0.01f); // Ensure valid ring
  }

  float innerAngle = m_openingAngle - 2.f * m_falloffAngle;
  if (innerAngle < 0.f) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "falloffAngle should be smaller than half of openingAngle");
  }
}

LightGPUData Ring::gpuData() const
{
  float innerAngle = m_openingAngle - 2.f * m_falloffAngle;

  auto retval = Light::gpuData();
  retval.type = LightType::RING;
  retval.ring.position = m_position;
  retval.ring.direction = m_direction;
  retval.ring.cosOuterAngle = cosf(m_openingAngle);
  retval.ring.cosInnerAngle = cosf(innerAngle);
  retval.ring.radius = m_radius;
  retval.ring.innerRadius = m_innerRadius;
  retval.ring.intensity = m_intensity;
  retval.ring.oneOverArea = m_radius > m_innerRadius ? 1.0f / (M_PI * (m_radius * m_radius - m_innerRadius * m_innerRadius)) : 1.0f;
  return retval;
}

} // namespace visrtx