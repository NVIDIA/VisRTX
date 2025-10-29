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

#include "Spot.h"

namespace visrtx {

Spot::Spot(DeviceGlobalState *d) : Light(d) {}

void Spot::commitParameters()
{
  Light::commitParameters();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_direction = getParam<vec3>("direction", vec3(0.f, 0.f, -1.f));
  m_openingAngle = getParam<float>("openingAngle", M_PI);
  m_falloffAngle = getParam<float>("falloffAngle", 0.1f);
  m_intensity =
      std::clamp(getParam<float>("intensity", getParam<float>("power", 1.f)),
          0.f,
          std::numeric_limits<float>::max());
}

LightGPUData Spot::gpuData() const
{
  float innerAngle = m_openingAngle - 2.f * m_falloffAngle;
  if (innerAngle < 0.f) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "falloffAngle should be smaller than half of openingAngle");
  }

  auto retval = Light::gpuData();
  retval.type = LightType::SPOT;
  retval.spot.position = m_position;
  retval.spot.direction = m_direction;
  retval.spot.cosOuterAngle = cosf(m_openingAngle);
  retval.spot.cosInnerAngle = cosf(innerAngle);
  retval.spot.intensity = m_intensity;
  return retval;
}

} // namespace visrtx
