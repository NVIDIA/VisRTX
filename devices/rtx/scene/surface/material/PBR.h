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

#pragma once

#include "sampler/Sampler.h"
#include "Material.h"

namespace visrtx {

struct PBR : public Material
{
  PBR(DeviceGlobalState *d);

  void commit() override;

 private:
  MaterialGPUData gpuData() const override;

  bool m_separateOpacity{false};

  vec3 m_color{1.f};
  helium::IntrusivePtr<Sampler> m_colorSampler;
  std::string m_colorAttribute;

  float m_opacity{1.f};
  helium::IntrusivePtr<Sampler> m_opacitySampler;
  std::string m_opacityAttribute;

  float m_metalness{0.f};
  helium::IntrusivePtr<Sampler> m_metalnessSampler;
  std::string m_metalnessAttribute;

  vec3 m_emissive{0.f};
  helium::IntrusivePtr<Sampler> m_emissiveSampler;
  std::string m_emissiveAttribute;

  float m_transmissiveness{0.f};
  helium::IntrusivePtr<Sampler> m_transmissivenessSampler;
  std::string m_transmissivenessAttribute;

  float m_roughness{1.f};
  helium::IntrusivePtr<Sampler> m_roughnessSampler;
  std::string m_roughnessAttribute;
};

} // namespace visrtx