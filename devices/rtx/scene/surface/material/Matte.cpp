/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Matte.h"
#include "gpu/gpu_objects.h"

namespace visrtx {

Matte::Matte(DeviceGlobalState *d) : Material(d) {}

void Matte::commit()
{
  m_opacity = getParam<float>("opacity", 1.f);
  m_opacitySampler = getParamObject<Sampler>("opacity");
  m_opacityAttribute = getParamString("opacity", "");

  m_color = vec4(vec3(0.8f), 1.f);
  getParam("color", ANARI_FLOAT32_VEC4, &m_color);
  getParam("color", ANARI_FLOAT32_VEC3, &m_color);
  m_colorSampler = getParamObject<Sampler>("color");
  m_colorAttribute = getParamString("color", "");

  m_cutoff = getParam<float>("alphaCutoff", 0.5f);
  m_mode = alphaModeFromString(getParamString("alphaMode", "opaque"));

  upload();
}

MaterialGPUData Matte::gpuData() const
{
  MaterialGPUData retval;

  retval.materialType = MaterialType::MATTE;

  populateMaterialParameter(
      retval.matte.color, m_color, m_colorSampler, m_colorAttribute);
  populateMaterialParameter(retval.matte.opacity,
      m_opacity,
      m_opacitySampler,
      m_opacityAttribute);

  retval.matte.cutoff = m_cutoff;
  retval.matte.alphaMode = m_mode;

  return retval;
}

} // namespace visrtx
