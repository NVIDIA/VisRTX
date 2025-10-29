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

#include "Rect.h"

namespace visrtx {

Rect::Rect(DeviceGlobalState *d) : Light(d) {}

void Rect::commitParameters()
{
  Light::commitParameters();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_edge1 = getParam<vec3>("edge1", vec3(1.f, 0.f, 0.f));
  m_edge2 = getParam<vec3>("edge2", vec3(0.f, 1.f, 0.f));
  m_intensity = std::max(
      getParam<float>("intensity", getParam<float>("power", 1.f)), 0.f);

  auto side = getParamString("side", "front");
  if (side == "front")
    m_side = Side::FRONT;
  else if (side == "back")
    m_side = Side::BACK;
  else if (side == "both")
    m_side = Side::BOTH;
  else {
    reportMessage(ANARI_SEVERITY_WARNING, "Invalid 'side' parameter on rect light");
    m_side = Side::FRONT;
  }
}

LightGPUData Rect::gpuData() const
{
  auto retval = Light::gpuData();
  retval.type = LightType::RECT;
  retval.rect.position = m_position;
  retval.rect.edge1 = m_edge1;
  retval.rect.edge2 = m_edge2;
  retval.rect.intensity = m_intensity;
  retval.rect.side.back = (m_side == Side::BACK || m_side == Side::BOTH) ? 1 : 0;
  retval.rect.side.front = (m_side == Side::FRONT || m_side == Side::BOTH) ? 1 : 0;
  auto area = length(cross(m_edge1, m_edge2));
  retval.rect.oneOverArea = area > 0.f ? 1.f / area : 1.f;

  return retval;
}

} // namespace visrtx
