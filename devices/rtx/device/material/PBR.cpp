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
      m_occlusionSampler(this),
      m_specularSampler(this),
      m_specularColorSampler(this),
      m_clearcoatSampler(this),
      m_clearcoatRoughnessSampler(this),
      m_clearcoatNormalSampler(this),
      m_transmissionSampler(this),
      m_thicknessSampler(this),
      m_sheenColorSampler(this),
      m_sheenRoughnessSampler(this),
      m_iridescenceSampler(this),
      m_iridescenceThicknessSampler(this)
{}

void PBR::commitParameters()
{
  m_opacity = getParam<float>("opacity", 1.f);
  m_opacitySampler = getParamObject<Sampler>("opacity");
  m_opacityAttribute = getParamString("opacity", "");

  m_color = vec4(1.f);
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

  m_emissive = vec4(0.f);
  getParam("emissive", ANARI_FLOAT32_VEC4, &m_emissive);
  getParam("emissive", ANARI_FLOAT32_VEC3, &m_emissive);
  m_emissiveSampler = getParamObject<Sampler>("emissive");
  m_emissiveAttribute = getParamString("emissive", "");

  m_occlusionSampler = getParamObject<Sampler>("occlusion");

  m_useSpecular = getParamDirect("specular").valid()
      || getParamDirect("specularColor").valid();
  m_specular = getParam<float>("specular", m_useSpecular ? 1.f : 0.f);
  m_specularSampler = getParamObject<Sampler>("specular");
  m_specularAttribute = getParamString("specular", "");

  m_specularColor = vec3(1.f);
  getParam("specularColor", ANARI_FLOAT32_VEC3, &m_specularColor);
  m_specularColorSampler = getParamObject<Sampler>("specularColor");
  m_specularColorAttribute = getParamString("specularColor", "");

  m_clearcoat = getParam<float>("clearcoat", 0.f);
  m_clearcoatSampler = getParamObject<Sampler>("clearcoat");
  m_clearcoatAttribute = getParamString("clearcoat", "");

  m_clearcoatRoughness = getParam<float>("clearcoatRoughness", 0.f);
  m_clearcoatRoughnessSampler = getParamObject<Sampler>("clearcoatRoughness");
  m_clearcoatRoughnessAttribute = getParamString("clearcoatRoughness", "");

  m_clearcoatNormalSampler = getParamObject<Sampler>("clearcoatNormal");

  m_transmission = getParam<float>("transmission", 0.f);
  m_transmissionSampler = getParamObject<Sampler>("transmission");
  m_transmissionAttribute = getParamString("transmission", "");

  m_ior = getParam<float>("ior", 1.5f);

  m_thickness = getParam<float>("thickness", 0.f);
  m_thicknessSampler = getParamObject<Sampler>("thickness");
  m_thicknessAttribute = getParamString("thickness", "");

  m_attenuationDistance = getParam<float>(
      "attenuationDistance", std::numeric_limits<float>::infinity());
  m_attenuationColor = vec3(1.f);
  getParam("attenuationColor", ANARI_FLOAT32_VEC3, &m_attenuationColor);

  m_sheenColor = vec3(0.f);
  getParam("sheenColor", ANARI_FLOAT32_VEC3, &m_sheenColor);
  m_sheenColorSampler = getParamObject<Sampler>("sheenColor");
  m_sheenColorAttribute = getParamString("sheenColor", "");

  m_sheenRoughness = getParam<float>("sheenRoughness", 0.f);
  m_sheenRoughnessSampler = getParamObject<Sampler>("sheenRoughness");
  m_sheenRoughnessAttribute = getParamString("sheenRoughness", "");

  m_iridescence = getParam<float>("iridescence", 0.f);
  m_iridescenceSampler = getParamObject<Sampler>("iridescence");
  m_iridescenceAttribute = getParamString("iridescence", "");

  m_iridescenceIor = getParam<float>("iridescenceIor", 1.3f);

  m_iridescenceThickness = getParam<float>("iridescenceThickness", 0.f);
  m_iridescenceThicknessSampler =
      getParamObject<Sampler>("iridescenceThickness");
  m_iridescenceThicknessAttribute = getParamString("iridescenceThickness", "");

  m_cutoff = getParam<float>("alphaCutoff", 0.5f);
  m_mode = alphaModeFromString(getParamString("alphaMode", "opaque"));
}

MaterialGPUData PBR::gpuData() const
{
  MaterialGPUData retval;
  auto &pb = retval.materialData.physicallyBased;

  retval.callableBaseIndex = static_cast<uint32_t>(SbtCallableEntryPoints::PBR);

  populateMaterialParameter(
      pb.baseColor, m_color, m_colorSampler.get(), m_colorAttribute);
  populateMaterialParameter(
      pb.opacity, m_opacity, m_opacitySampler.get(), m_opacityAttribute);
  populateMaterialParameter(
      pb.metallic, m_metallic, m_metallicSampler.get(), m_metallicAttribute);
  populateMaterialParameter(pb.roughness,
      m_roughness,
      m_roughnessSampler.get(),
      m_roughnessAttribute);
  pb.normalSampler =
      m_normalSampler ? m_normalSampler->index() : ~DeviceObjectIndex{0};
  populateMaterialParameter(
      pb.emissive, m_emissive, m_emissiveSampler.get(), m_emissiveAttribute);
  populateMaterialParameter(pb.transmission,
      m_transmission,
      m_transmissionSampler.get(),
      m_transmissionAttribute);

  pb.ior = m_ior;
  pb.cutoff = m_cutoff;
  pb.alphaMode = m_mode;

  // OPAQUE mode ignores alpha at shading time (see adjustedMaterialOpacity),
  // so it alone implies alpha-opaque; otherwise both opacity and baseColor.alpha
  // must be statically 1. Transmission is orthogonal — non-zero values let
  // light through even at OPAQUE, opacity=1, alpha=1 — so it must also be zero.
  retval.isFullyOpaque = (m_mode == AlphaMode::OPAQUE
                             || (isStaticOne(pb.opacity)
                                 && isStaticOne(pb.baseColor, 3)))
      && isStaticZero(pb.transmission);

  pb.occlusionSampler =
      m_occlusionSampler ? m_occlusionSampler->index() : ~DeviceObjectIndex{0};

  populateMaterialParameter(
      pb.specular, m_specular, m_specularSampler.get(), m_specularAttribute);
  populateMaterialParameter(pb.specularColor,
      vec4(m_specularColor, 1.f),
      m_specularColorSampler.get(),
      m_specularColorAttribute);
  pb.useSpecular = m_useSpecular ? 1u : 0u;

  populateMaterialParameter(pb.clearcoat,
      m_clearcoat,
      m_clearcoatSampler.get(),
      m_clearcoatAttribute);
  populateMaterialParameter(pb.clearcoatRoughness,
      m_clearcoatRoughness,
      m_clearcoatRoughnessSampler.get(),
      m_clearcoatRoughnessAttribute);
  pb.clearcoatNormalSampler = m_clearcoatNormalSampler
      ? m_clearcoatNormalSampler->index()
      : ~DeviceObjectIndex{0};

  populateMaterialParameter(pb.thickness,
      m_thickness,
      m_thicknessSampler.get(),
      m_thicknessAttribute);
  pb.attenuationDistance = m_attenuationDistance;
  pb.attenuationColor = m_attenuationColor;

  populateMaterialParameter(pb.sheenColor,
      vec4(m_sheenColor, 0.f),
      m_sheenColorSampler.get(),
      m_sheenColorAttribute);
  populateMaterialParameter(pb.sheenRoughness,
      m_sheenRoughness,
      m_sheenRoughnessSampler.get(),
      m_sheenRoughnessAttribute);

  populateMaterialParameter(pb.iridescence,
      m_iridescence,
      m_iridescenceSampler.get(),
      m_iridescenceAttribute);
  pb.iridescenceIor = m_iridescenceIor;
  populateMaterialParameter(pb.iridescenceThickness,
      m_iridescenceThickness,
      m_iridescenceThicknessSampler.get(),
      m_iridescenceThicknessAttribute);

  return retval;
}

} // namespace visrtx
