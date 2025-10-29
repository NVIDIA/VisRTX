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

#pragma once

#include "RegisteredObject.h"
#include "sampler/Sampler.h"

namespace visrtx {

struct Material : public RegisteredObject<MaterialGPUData>
{
  Material(DeviceGlobalState *d);
  ~Material() = default;

  static Material *createInstance(
      std::string_view subtype, DeviceGlobalState *d);
};

// Inlined helper functions ///////////////////////////////////////////////////

template <typename T>
inline void populateMaterialParameter(MaterialParameter &mp,
    T value,
    helium::IntrusivePtr<Sampler> sampler,
    const std::string &attrib)
{
  if (sampler && sampler->isValid()) {
    mp.type = MaterialParameterType::SAMPLER;
    mp.sampler = sampler->index();
  } else if (!attrib.empty()) {
    mp.type = MaterialParameterType::ATTRIBUTE;
    mp.attribute = attributeFromString(attrib);
  } else {
    mp.type = MaterialParameterType::VALUE;
    mp.value = vec4(value);
  }
}

inline AlphaMode alphaModeFromString(const std::string &s)
{
  if (s == "blend")
    return AlphaMode::BLEND;
  else if (s == "mask")
    return AlphaMode::MASK;
  else
    return AlphaMode::OPAQUE;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Material *, ANARI_MATERIAL);
