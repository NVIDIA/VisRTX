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

#include "Geometry.h"
// specific types
#include "Cones.h"
#include "Cylinders.h"
#include "Quads.h"
#include "Spheres.h"
#include "Triangles.h"
#include "UnknownGeometry.h"
// std
#include <string_view>

namespace visrtx {

// Geometry definitions ///////////////////////////////////////////////////////

Geometry *Geometry::createInstance(
    std::string_view subtype, DeviceGlobalState *d)
{
  Geometry *retval = nullptr;

  if (subtype == "triangle")
    retval = new Triangles;
  else if (subtype == "quad")
    retval = new Quads;
  else if (subtype == "sphere")
    retval = new Spheres;
  else if (subtype == "cylinder")
    retval = new Cylinders;
  else if (subtype == "cone")
    retval = new Cones;
  else
    retval = new UnknownGeometry;

  retval->setDeviceState(d);
  retval->setRegistry(d->registry.geometries);
  return retval;
}

void Geometry::commit()
{
  m_colors = getParamObject<Array1D>("primitive.color");

  m_attribute0 = getParamObject<Array1D>("primitive.attribute0");
  m_attribute1 = getParamObject<Array1D>("primitive.attribute1");
  m_attribute2 = getParamObject<Array1D>("primitive.attribute2");
  m_attribute3 = getParamObject<Array1D>("primitive.attribute3");

  m_primID = getParamObject<Array1D>("primitive.primID");
}

void Geometry::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLASChange = newTimeStamp();
}

GeometryGPUData Geometry::gpuData() const
{
  GeometryGPUData retval{};

  populateAttributePtr(m_attribute0, retval.attr[0]);
  populateAttributePtr(m_attribute1, retval.attr[1]);
  populateAttributePtr(m_attribute2, retval.attr[2]);
  populateAttributePtr(m_attribute3, retval.attr[3]);
  populateAttributePtr(m_colors, retval.attr[4]);

  retval.primID =
      m_primID ? m_primID->beginAs<uint32_t>(AddressSpace::GPU) : nullptr;

  return retval;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Geometry *);
