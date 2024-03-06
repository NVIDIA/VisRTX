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

#include "PBR.h"

namespace visrtx {

PBR::PBR(DeviceGlobalState *d) : Material(d) {}

void PBR::commit()
{
  m_opacity = getParam<float>("opacity", 1.f);
  m_opacitySampler = getParamObject<Sampler>("opacity");
  m_opacityAttribute = getParamString("opacity", "");

  m_baseColor = vec4(1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &m_baseColor);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &m_baseColor);
  m_baseColorSampler = getParamObject<Sampler>("baseColor");
  m_baseColorAttribute = getParamString("baseColor", "");

  m_metallic = getParam<float>("metallic", 1.f);
  m_metallicSampler = getParamObject<Sampler>("metallic");
  m_metallicAttribute = getParamString("metallic", "");

  m_roughness = getParam<float>("roughness", 1.f);
  m_roughnessSampler = getParamObject<Sampler>("roughness");
  m_roughnessAttribute = getParamString("roughness", "");

  m_specular = getParam<float>("specular", 0.f);
  m_specularSampler = getParamObject<Sampler>("specular");
  m_specularAttribute = getParamString("specular", "");

  m_specularColor = getParam<vec3>("specularColor", vec3(1.f));
  m_specularColorSampler = getParamObject<Sampler>("specularColor");
  m_specularColorAttribute = getParamString("specularColor", "");

  m_emissive = getParam<vec3>("emissive", vec3(0.f));
  m_emissiveSampler = getParamObject<Sampler>("emissive");
  m_emissiveAttribute = getParamString("emissive", "");

  m_ior = getParam<float>("ior", 1.5f);

  m_cutoff = getParam<float>("alphaCutoff", 0.5f);
  m_mode = alphaModeFromString(getParamString("alphaMode", "opaque"));

  upload();
}

MaterialGPUData PBR::gpuData() const
{
  MaterialGPUData retval;

  populateMaterialParameter(retval.values[MV_BASE_COLOR],
      m_baseColor,
      m_baseColorSampler,
      m_baseColorAttribute);
  populateMaterialParameter(retval.values[MV_OPACITY],
      m_opacity,
      m_opacitySampler,
      m_opacityAttribute);
  populateMaterialParameter(retval.values[MV_METALLIC],
      m_metallic,
      m_metallicSampler,
      m_metallicAttribute);
  populateMaterialParameter(retval.values[MV_ROUGHNESS],
      m_roughness,
      m_roughnessSampler,
      m_roughnessAttribute);
  populateMaterialParameter(retval.values[MV_SPECULAR],
      m_specular,
      m_specularSampler,
      m_specularAttribute);
  populateMaterialParameter(retval.values[MV_SPECULAR_COLOR],
      m_specularColor,
      m_specularColorSampler,
      m_specularColorAttribute);
  populateMaterialParameter(retval.values[MV_EMISSIVE],
      m_emissive,
      m_emissiveSampler,
      m_emissiveAttribute);

  retval.ior = m_ior;
  retval.cutoff = m_cutoff;
  retval.mode = m_mode;
  retval.isPBR = true;

  return retval;
}

} // namespace visrtx
