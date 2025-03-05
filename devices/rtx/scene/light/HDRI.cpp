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

#include "HDRI.h"

namespace visrtx {

HDRI::HDRI(DeviceGlobalState *d) : Light(d), m_radiance(this) {}

HDRI::~HDRI()
{
  cleanup();
}

void HDRI::commitParameters()
{
  Light::commitParameters();
  m_radiance = getParamObject<Array2D>("radiance");
}

void HDRI::finalize()
{
  cleanup();

  if (!m_radiance) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'radiance' on HDRI light");
    return;
  }

  m_direction = getParam<vec3>("direction", vec3(1.f, 0.f, 0.f));
  m_up = getParam<vec3>("up", vec3(0.f, 0.f, 1.f));
  m_scale = getParam<float>("scale", 1.f);
  m_visible = getParam<bool>("visible", true);

  cudaArray_t cuArray = {};
  const bool isFp = isFloat(m_radiance->elementType());
  if (isFp)
    cuArray = m_radiance->acquireCUDAArrayFloat();
  else
    cuArray = m_radiance->acquireCUDAArrayUint8();

  m_radianceTex =
      makeCudaTextureObject(cuArray, !isFp, "linear", "repeat", "repeat");

  upload();
}

bool HDRI::isValid() const
{
  return m_radiance;
}

bool HDRI::isHDRI() const
{
  return true;
}

LightGPUData HDRI::gpuData() const
{
  auto retval = Light::gpuData();

  const vec3 forward = glm::normalize(-m_direction);
  const vec3 right = glm::normalize(glm::cross(m_up, forward));
  const vec3 up = glm::normalize(glm::cross(forward, right));

  retval.type = LightType::HDRI;
  retval.hdri.xfm[0] = forward;
  retval.hdri.xfm[1] = right;
  retval.hdri.xfm[2] = up;
  retval.hdri.scale = m_scale;
  retval.hdri.radiance = m_radianceTex;
  retval.hdri.visible = m_visible;

  return retval;
}

void HDRI::cleanup()
{
  if (m_radiance && m_radianceTex) {
    cudaDestroyTextureObject(m_radianceTex);
    if (isFloat(m_radiance->elementType()))
      m_radiance->releaseCUDAArrayFloat();
    else
      m_radiance->releaseCUDAArrayUint8();
  }
}

} // namespace visrtx
