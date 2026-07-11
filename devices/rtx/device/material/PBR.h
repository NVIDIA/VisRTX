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

#pragma once

#include "Material.h"
#include "sampler/Sampler.h"

#include <limits>

namespace visrtx {

struct PBR : public Material
{
  PBR(DeviceGlobalState *d);

  void commitParameters() override;

  bool emissionIsConstant() const override;
  bool emissionIsSampleable() const override;
  vec3 emissionAverage() const override;

 private:
  MaterialGPUData gpuData() const override;

  float m_cutoff{0.5f};
  AlphaMode m_mode{AlphaMode::OPAQUE};

  vec4 m_color{1.f, 1.f, 1.f, 1.f};
  helium::ChangeObserverPtr<Sampler> m_colorSampler;
  std::string m_colorAttribute;

  float m_opacity{1.f};
  helium::ChangeObserverPtr<Sampler> m_opacitySampler;
  std::string m_opacityAttribute;

  float m_metallic{1.f};
  helium::ChangeObserverPtr<Sampler> m_metallicSampler;
  std::string m_metallicAttribute;

  float m_roughness{1.f};
  helium::ChangeObserverPtr<Sampler> m_roughnessSampler;
  std::string m_roughnessAttribute;

  helium::ChangeObserverPtr<Sampler> m_normalSampler;

  vec4 m_emissive{0.f, 0.f, 0.f, 0.f};
  helium::ChangeObserverPtr<Sampler> m_emissiveSampler;
  std::string m_emissiveAttribute;

  helium::ChangeObserverPtr<Sampler> m_occlusionSampler;

  float m_specular{0.f};
  helium::ChangeObserverPtr<Sampler> m_specularSampler;
  std::string m_specularAttribute;

  vec3 m_specularColor{1.f};
  helium::ChangeObserverPtr<Sampler> m_specularColorSampler;
  std::string m_specularColorAttribute;

  float m_clearcoat{0.f};
  helium::ChangeObserverPtr<Sampler> m_clearcoatSampler;
  std::string m_clearcoatAttribute;

  float m_clearcoatRoughness{0.f};
  helium::ChangeObserverPtr<Sampler> m_clearcoatRoughnessSampler;
  std::string m_clearcoatRoughnessAttribute;

  helium::ChangeObserverPtr<Sampler> m_clearcoatNormalSampler;

  float m_transmission{0.f};
  helium::ChangeObserverPtr<Sampler> m_transmissionSampler;
  std::string m_transmissionAttribute;

  float m_ior{1.5f};

  float m_thickness{0.f};
  helium::ChangeObserverPtr<Sampler> m_thicknessSampler;
  std::string m_thicknessAttribute;

  float m_attenuationDistance{std::numeric_limits<float>::infinity()};
  vec3 m_attenuationColor{1.f};

  vec3 m_sheenColor{0.f};
  helium::ChangeObserverPtr<Sampler> m_sheenColorSampler;
  std::string m_sheenColorAttribute;

  float m_sheenRoughness{0.f};
  helium::ChangeObserverPtr<Sampler> m_sheenRoughnessSampler;
  std::string m_sheenRoughnessAttribute;

  float m_iridescence{0.f};
  helium::ChangeObserverPtr<Sampler> m_iridescenceSampler;
  std::string m_iridescenceAttribute;

  float m_iridescenceIor{1.3f};

  float m_iridescenceThickness{0.f};
  helium::ChangeObserverPtr<Sampler> m_iridescenceThicknessSampler;
  std::string m_iridescenceThicknessAttribute;
};

} // namespace visrtx
