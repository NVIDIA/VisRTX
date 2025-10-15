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

#include "Sampler.h"
// specific types
#include "CompressedImage2D.h"
#include "Image1D.h"
#include "Image2D.h"
#include "Image3D.h"
#include "PrimitiveSampler.h"
#include "TransformSampler.h"
#include "UnknownSampler.h"

namespace visrtx {

Sampler::Sampler(DeviceGlobalState *s)
    : RegisteredObject<SamplerGPUData>(ANARI_SAMPLER, s)
{
  setRegistry(s->registry.samplers);
}

Sampler *Sampler::createInstance(std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "compressedImage2D")
    return new CompressedImage2D(d);
  else if (subtype == "image1D")
    return new Image1D(d);
  else if (subtype == "image2D")
    return new Image2D(d);
  else if (subtype == "image3D")
    return new Image3D(d);
  else if (subtype == "primitive")
    return new PrimitiveSampler(d);
  else if (subtype == "transform")
    return new TransformSampler(d);
  else
    return new UnknownSampler(subtype, d);
}

void Sampler::commitParameters()
{
  m_inAttribute = getParamString("inAttribute", "attribute0");
  m_inTransform = getParam<mat4>("inTransform", mat4(1.f));
  m_inOffset = getParam<vec4>("inOffset", vec4(0.f));
  m_outTransform = getParam<mat4>("outTransform", mat4(1.f));
  m_outOffset = getParam<vec4>("outOffset", vec4(0.f));
  m_borderColor = getParam<vec4>("borderColor", vec4(0.f));
}

SamplerGPUData Sampler::gpuData() const
{
  SamplerGPUData retval;
  retval.attribute = attributeFromString(m_inAttribute);
  retval.inTransform = m_inTransform;
  retval.inOffset = m_inOffset;
  retval.outTransform = m_outTransform;
  retval.outOffset = m_outOffset;
  return retval;
}

MaterialAttribute attributeFromString(const std::string &str)
{
  if (str == "attribute0")
    return MaterialAttribute::ATTRIB_0;
  else if (str == "attribute1")
    return MaterialAttribute::ATTRIB_1;
  else if (str == "attribute2")
    return MaterialAttribute::ATTRIB_2;
  else if (str == "attribute3")
    return MaterialAttribute::ATTRIB_3;
  else if (str == "color")
    return MaterialAttribute::COLOR;
  else if (str == "objectNormal")
    return MaterialAttribute::OBJECT_NORMAL;
  else if (str == "objectPosition")
    return MaterialAttribute::OBJECT_POSITION;
  else if (str == "worldNormal")
    return MaterialAttribute::WORLD_NORMAL;
  else if (str == "worldPosition")
    return MaterialAttribute::WORLD_POSITION;
  else
    return MaterialAttribute::UNKNOWN;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Sampler *);
