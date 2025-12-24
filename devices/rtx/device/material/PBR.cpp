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

#include "PBR.h"
#include "gpu/gpu_objects.h"
#include "gpu/sbt.h"

namespace visrtx {

PBR::PBR(DeviceGlobalState *d)
    : Material(d),
      m_colorSampler(this),
      m_opacitySampler(this),
      m_metallicSampler(this),
      m_roughnessSampler(this),
      m_normalSampler(this),
      m_emissiveSampler(this),
      m_transmissionSampler(this)
{}

void PBR::commitParameters()
{
  m_opacity = getParam<float>("opacity", 1.f);
  m_opacitySampler = getParamObject<Sampler>("opacity");
  m_opacityAttribute = getParamString("opacity", "");

  m_color = vec4(vec3(0.8f), 1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &m_color);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &m_color);
  m_colorSampler = getParamObject<Sampler>("baseColor");
  m_colorAttribute = getParamString("baseColor", "");

  m_metallic = getParam<float>("metallic", 1.f);
  m_metallicSampler = getParamObject<Sampler>("metallic");
  m_metallicAttribute = getParamString("metallic", "");

  m_roughness = getParam<float>("roughness", 1.f);
  m_roughnessSampler = getParamObject<Sampler>("roughness");
  m_roughnessAttribute = getParamString("roughness", "");

  m_normalSampler = getParamObject<Sampler>("normal");

  m_emissive = vec4(0.f, 0.f, 0.f, 0.f);
  getParam("emissive", ANARI_FLOAT32_VEC4, &m_emissive);
  getParam("emissive", ANARI_FLOAT32_VEC3, &m_emissive);
  m_emissiveSampler = getParamObject<Sampler>("emissive");
  m_emissiveAttribute = getParamString("emissive", "");

  m_transmission = getParam<float>("transmission", 0.f);
  m_transmissionSampler = getParamObject<Sampler>("transmission");
  m_transmissionAttribute = getParamString("transmission", "");

  m_ior = getParam<float>("ior", 1.5f);

  m_cutoff = getParam<float>("alphaCutoff", 0.5f);
  m_mode = alphaModeFromString(getParamString("alphaMode", "opaque"));
}

MaterialGPUData PBR::gpuData() const
{
  MaterialGPUData retval;

  retval.callableBaseIndex = static_cast<uint32_t>(SbtCallableEntryPoints::PBR);

  populateMaterialParameter(retval.materialData.physicallyBased.baseColor,
      m_color,
      m_colorSampler.get(),
      m_colorAttribute);
  populateMaterialParameter(retval.materialData.physicallyBased.opacity,
      m_opacity,
      m_opacitySampler.get(),
      m_opacityAttribute);
  populateMaterialParameter(retval.materialData.physicallyBased.metallic,
      m_metallic,
      m_metallicSampler.get(),
      m_metallicAttribute);
  populateMaterialParameter(retval.materialData.physicallyBased.roughness,
      m_roughness,
      m_roughnessSampler.get(),
      m_roughnessAttribute);
  retval.materialData.physicallyBased.normalSampler =
      m_normalSampler ? m_normalSampler->index() : ~DeviceObjectIndex{0};
  populateMaterialParameter(retval.materialData.physicallyBased.emissive,
      m_emissive,
      m_emissiveSampler.get(),
      m_emissiveAttribute);
  populateMaterialParameter(retval.materialData.physicallyBased.transmission,
      m_transmission,
      m_transmissionSampler.get(),
      m_transmissionAttribute);

  retval.materialData.physicallyBased.ior = m_ior;
  retval.materialData.physicallyBased.cutoff = m_cutoff;
  retval.materialData.physicallyBased.alphaMode = m_mode;

  return retval;
}

} // namespace visrtx
