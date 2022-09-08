/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ColorMap.h"
#include "Image1D.h"
#include "Image2D.h"
#include "PrimitiveSampler.h"
#include "UnknownSampler.h"

namespace visrtx {

Sampler *Sampler::createInstance(std::string_view subtype, DeviceGlobalState *d)
{
  Sampler *retval = nullptr;

  if (subtype == "image1D")
    retval = new Image1D();
  else if (subtype == "image2D")
    retval = new Image2D();
  else if (subtype == "primitive")
    retval = new PrimitiveSampler();
  else if (subtype == "colorMap")
    retval = new ColorMap();
  else
    retval = new UnknownSampler;

  retval->setDeviceState(d);
  retval->setRegistry(d->registry.samplers);
  return retval;
}

void Sampler::commit()
{
  m_inAttribute = getParam<std::string>("inAttribute", "attribute0");
}

SamplerGPUData Sampler::gpuData() const
{
  SamplerGPUData retval;
  if (m_inAttribute == "attribute0")
    retval.attribute = 0;
  else if (m_inAttribute == "attribute1")
    retval.attribute = 1;
  else if (m_inAttribute == "attribute2")
    retval.attribute = 2;
  else if (m_inAttribute == "attribute3")
    retval.attribute = 3;
  else if (m_inAttribute == "color")
    retval.attribute = 4;
  return retval;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Sampler *);
