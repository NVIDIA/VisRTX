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

#include "Material.h"
// specific types

#include "Matte.h"
#include "PBR.h"
#include "UnknownMaterial.h"
#ifdef USE_MDL
#include "MDL.h"
#include "PhysicallyBasedMDL.h"
#endif // defined(USE_MDL)

#include <string>

using namespace std::string_literals;

namespace visrtx {

Material::Material(DeviceGlobalState *s)
    : RegisteredObject<MaterialGPUData>(ANARI_MATERIAL, s)
{
  setRegistry(s->registry.materials);
}

Material *Material::createInstance(
    std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "matte" || subtype == "transparentMatte")
    return new Matte(d);
  else if (subtype == "pbr" || subtype == "physicallyBased")
#if defined(USE_MDL) && defined(USE_MDL_FOR_PHYSICALLY_BASED)
    if (d->mdl)
      return new PhysicallyBasedMDL(d);
    else
      return new PBR(d);
#else
    return new PBR(d);
#endif
#ifdef USE_MDL
  else if (subtype == "mdl" && d->mdl)
    return new MDL(d);
#endif // defined(USE_MDL)
  else
    return new UnknownMaterial(subtype, d);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Material *);
