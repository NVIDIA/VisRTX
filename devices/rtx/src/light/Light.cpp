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

#include "Light.h"
// specific types
#include "Directional.h"
#include "HDRI.h"
#include "Point.h"
#include "Rect.h"
#include "Ring.h"
#include "Spot.h"
#include "UnknownLight.h"

namespace visrtx {

Light::Light(DeviceGlobalState *s)
    : RegisteredObject<LightGPUData>(ANARI_LIGHT, s)
{
  setRegistry(s->registry.lights);
}

void Light::commitParameters()
{
  m_color = getParam<vec3>("color", vec3(1.f));
}

LightGPUData Light::gpuData() const
{
  LightGPUData retval;
  retval.color = m_color;
  return retval;
}

Light *Light::createInstance(std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "directional")
    return new Directional(d);
  else if (subtype == "hdri")
    return new HDRI(d);
  else if (subtype == "point")
    return new Point(d);
  else if (subtype == "quad")
    return new Rect(d);
  else if (subtype == "ring")
    return new Ring(d);
  else if (subtype == "spot")
    return new Spot(d);
  else
    return new UnknownLight(subtype, d);
}

bool Light::isHDRI() const
{
  return false;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Light *);
