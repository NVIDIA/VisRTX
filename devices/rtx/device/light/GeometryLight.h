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

#include "Light.h"

namespace visrtx {

// A Geometry Light: an internal Light synthesized from an Emissive Surface, not
// reachable through Light::createInstance. Owned by its Surface, it occupies a
// registry.lights slot like any authored light and is instanced and sampled the
// same way. "Synthesize" names the act of creating one.
struct GeometryLight : public Light
{
  GeometryLight(DeviceGlobalState *d);

  // Configure from the owning surface's geometry + material and mean radiance,
  // then publish to the registry. `radiance` is the emission average (== the
  // constant for a constant emitter); the sampler evaluates the material at the
  // sampled point when its emission is not constant. area is the geometry's
  // object-space total.
  void configure(DeviceObjectIndex geometry,
      DeviceObjectIndex material,
      const vec3 &radiance,
      float area);

  // Unlike an authored light, a Geometry Light is (re)configured by the World
  // DURING the light-set rebuild, so it must NOT bump lastLightSetChange — that
  // would leave the world perpetually dirty and rebuild every frame. Its
  // invalidation rides the owning material/geometry commit instead.
  void markFinalized() override;

 private:
  LightGPUData gpuData() const override;

  DeviceObjectIndex m_geometry{-1};
  DeviceObjectIndex m_material{-1};
  vec3 m_radiance{0.f};
  float m_area{0.f};
};

} // namespace visrtx
