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

#include "SpatialField.h"
#include "SpatialFieldRegistry.h"
// specific types
#include "NvdbRectilinearField.h"
#include "NvdbRegularField.h"
#include "StructuredRectilinearField.h"
#include "StructuredRegularField.h"
#include "UnknownSpatialField.h"

namespace visrtx {

SpatialField::SpatialField(DeviceGlobalState *s)
    : RegisteredObject<SpatialFieldGPUData>(ANARI_SPATIAL_FIELD, s)
{
  setRegistry(s->registry.fields);
}

void SpatialField::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastBLASChange = helium::newTimeStamp();
}

SpatialField *SpatialField::createInstance(
    std::string_view subtype, DeviceGlobalState *d)
{
  // Try built-in types first
  if (subtype == "structuredRegular")
    return new StructuredRegularField(d);
  else if (subtype == "structuredRectilinear")
    return new StructuredRectilinearField(d);
  else if (subtype == "nanovdb")
    return new NvdbRegularField(d);
  else if (subtype == "nanovdbRectilinear")
    return new NvdbRectilinearField(d);
  
  // Try registry for custom field types (registered at static init time)
  std::string subtypeStr(subtype);
  if (auto* customField = SpatialFieldRegistry::instance().create(d, subtypeStr)) {
    return customField;
  }
  
  // Unknown type
  return new UnknownSpatialField(subtype, d);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::SpatialField *);
