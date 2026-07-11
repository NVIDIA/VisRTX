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

#include "PhysicallyBasedMDL.h"

#include <anari/anari_cpp/Traits.h>
#include <anari/frontend/anari_enums.h>

#include <string_view>

using namespace std::string_view_literals;

namespace visrtx {

PhysicallyBasedMDL::PhysicallyBasedMDL(DeviceGlobalState *d) : MDL(d)
{
  setParam("source",
      ANARI_STRING,
      "::visrtx::physically_based::physically_based_material");
}

void PhysicallyBasedMDL::translateAndRemoveParameter(std::string_view name)
{
  std::string nameS(name);
  auto asAny = getParamDirect(nameS);
  switch (asAny.type()) {
  case ANARI_FLOAT32:
  case ANARI_FLOAT32_VEC2:
  case ANARI_FLOAT32_VEC3:
  case ANARI_FLOAT32_VEC4: {
    setParamDirect(nameS + ".value", asAny);
    removeParam(nameS + ".texture");
    break;
  }
  case ANARI_SAMPLER:
    setParamDirect(nameS + ".texture", asAny);
    removeParam(nameS + ".value");
    break;
  default:
    // Do nothing...
    return;
  }

  removeParam(nameS);
}

void PhysicallyBasedMDL::commitParameters()
{
  // Limitation: attribute-bound inputs are not supported by the MDL wrapper;
  // only .value / .texture bindings are translated below.

  // Translate all supported parameters to their matching .value or .texture if they are
  // variant inputs.
  translateAndRemoveParameter("opacity"sv);
  translateAndRemoveParameter("baseColor"sv);
  translateAndRemoveParameter("metallic"sv);
  translateAndRemoveParameter("roughness"sv);
  translateAndRemoveParameter("normal"sv);
  translateAndRemoveParameter("emissive"sv);
  translateAndRemoveParameter("occlusion"sv);
  translateAndRemoveParameter("specular"sv);
  translateAndRemoveParameter("specularColor"sv);
  translateAndRemoveParameter("clearcoat"sv);
  translateAndRemoveParameter("clearcoatRoughness"sv);
  translateAndRemoveParameter("clearcoatNormal"sv);
  translateAndRemoveParameter("transmission"sv);
  translateAndRemoveParameter("thickness"sv);
  translateAndRemoveParameter("sheenColor"sv);
  translateAndRemoveParameter("sheenRoughness"sv);
  translateAndRemoveParameter("iridescence"sv);
  translateAndRemoveParameter("iridescenceThickness"sv);

  // Capture the emissive value AFTER translation, from the persistent
  // post-translate keys: the translation consumes the pre-translate `emissive`
  // key at its first commit, so reading that key here would zero the capture —
  // and silently drop the Geometry Light — on any later commit that doesn't
  // re-set `emissive`. The keys are mutually exclusive post-translate; the
  // sampler gate mirrors the material's own texture-over-value precedence.
  // Constant iff a nonzero inline color is bound (a sampler stays per-hit only).
  // Note: unsetting `emissive` leaves the last translated key in place — a
  // pre-existing translation-lifecycle trait shared by the MDL argument block,
  // so the light stays consistent with the still-glowing surface.
  {
    vec3 radiance(0.f);
    if (getParamDirect("emissive.texture").type() != ANARI_SAMPLER) {
      vec4 v(0.f);
      getParam("emissive.value", ANARI_FLOAT32_VEC4, &v);
      getParam("emissive.value", ANARI_FLOAT32_VEC3, &v);
      radiance = vec3(v);
    }
    m_emissionRadiance = radiance;
    m_emissionIsConstant = glm::any(glm::greaterThan(radiance, vec3(0.f)));
  }

  // Purposefully exclude the following from the above translation, as they are
  // raw parameters and are already set correctly.
  // alphaMode: enum
  // alphaCutoff: float
  // ior: float
  // attenuationDistance: float
  // attenuationColor: color
  // iridescenceIor: float

  // Translate alphaMode to its matching integer value
  if (auto alphaModeAny = getParamDirect("alphaMode");
      alphaModeAny.type() == ANARI_STRING) {
    auto alphaMode = alphaModeFromString(alphaModeAny.getString());
    setParam("alphaMode", static_cast<int>(alphaMode));
  }

  MDL::commitParameters();
  // The light-set refresh runs from MDL::finalize (after the emission
  // classification is resolved), not here.
}

bool PhysicallyBasedMDL::emissionIsConstant() const
{
  return m_emissionIsConstant;
}

bool PhysicallyBasedMDL::emissionIsSampleable() const
{
  // Deliberate scoping per ADR 0006: only a nonzero constant `emissive` is a
  // Geometry Light on the wrapper; a bound emissive sampler stays per-hit for
  // now (it still deposits via the compiled EDF, just without next-event
  // sampling). The synthetic-hit evaluation itself works — raw `mdl` textured
  // emission uses it — bringing the wrapper's sampler case into next-event is
  // a follow-up.
  return m_emissionIsConstant;
}

vec3 PhysicallyBasedMDL::emissionAverage() const
{
  return m_emissionRadiance;
}

} // namespace visrtx
