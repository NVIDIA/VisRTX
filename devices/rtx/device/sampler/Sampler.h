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

#include "RegisteredObject.h"
#if defined(USE_MDL)
#include "libmdl/ResourceStats.h"
#endif

namespace visrtx {

struct Sampler : public RegisteredObject<SamplerGPUData>
{
  Sampler(DeviceGlobalState *d);

  virtual void commitParameters() override;

  virtual int numChannels() const = 0;

  // Average sampled value, used only to size a textured emitter's Pick Power
  // (variance, never bias). The default assumes a fully-lit unit value so an
  // un-averaged sampler is still picked; Image2D overrides with the mean texel.
  virtual vec4 averageValue() const;

#if defined(USE_MDL)
  // Per-channel texel reduction consumed by the MDL emission classifier's value
  // source (maxAbs for the zero proof, meanPositive for the magnitude proxy,
  // minValue for the sign proof). The default is Unknown (valid but unproven):
  // a sampler that cannot reduce its texels neither proves zero nor proves a
  // non-negative sign, so its emission stays register-eligible only under the
  // policy's Unknown path. Image2D overrides with a real scan. MDL-only: the
  // classifier is the sole consumer and lives behind USE_MDL.
  virtual libmdl::ResourceStats emissionStats() const;
#endif

  static Sampler *createInstance(
      std::string_view subtype, DeviceGlobalState *d);

 protected:
  virtual SamplerGPUData gpuData() const override = 0;

  std::string m_inAttribute;
  mat4 m_inTransform;
  vec4 m_inOffset;
  mat4 m_outTransform;
  vec4 m_outOffset;
  vec4 m_borderColor;
};

MaterialAttribute attributeFromString(const std::string &str);

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Sampler *, ANARI_SAMPLER);
