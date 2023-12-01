/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "Cone.h"
#include "Curve.h"
#include "Cylinder.h"
#include "Quad.h"
#include "Sphere.h"
#include "Triangle.h"
#include "UnknownGeometry.h"
// std
#include <cstring>
#include <string_view>
// helium
#include <helium/helium_math.h>

#include "utility/AnariTypeHelpers.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

static void populateAttributeData(helium::IntrusivePtr<Array1D> array,
    AttributeData &attr,
    const std::optional<vec4> &uniformValue)
{
  attr.type = ANARI_UNKNOWN;
  attr.numChannels = 0;
  attr.data = nullptr;
  std::memcpy(&attr.uniformValue,
      &helium::DEFAULT_ATTRIBUTE_VALUE,
      sizeof(attr.uniformValue));

  if (!array && !uniformValue)
    return;

  auto type = array ? array->elementType() : ANARI_FLOAT32_VEC4;

  if (!isColor(type) && !uniformValue)
    return;

  attr.type = isColor(type) ? type : ANARI_FLOAT32_VEC4;
  attr.numChannels = numANARIChannels(attr.type);
  attr.data = array ? array->dataGPU() : nullptr;
  attr.uniformValue = *uniformValue;
}

// Geometry definitions ///////////////////////////////////////////////////////

Geometry::Geometry(DeviceGlobalState *s)
    : RegisteredObject<GeometryGPUData>(ANARI_GEOMETRY, s)
{
  setRegistry(s->registry.geometries);
}

Geometry *Geometry::createInstance(
    std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "triangle")
    return new Triangle(d);
  else if (subtype == "quad")
    return new Quad(d);
  else if (subtype == "sphere")
    return new Sphere(d);
  else if (subtype == "cylinder")
    return new Cylinder(d);
  else if (subtype == "cone")
    return new Cone(d);
  else if (subtype == "curve")
    return new Curve(d);
  else
    return new UnknownGeometry(subtype, d);
}

void Geometry::commit()
{
  commitAttributes("primitive.", m_primitiveAttributes);
}

void Geometry::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLASChange = helium::newTimeStamp();
}

GeometryGPUData Geometry::gpuData() const
{
  GeometryGPUData retval{};
  populateAttributeDataSet(m_primitiveAttributes, retval.attr);
  return retval;
}

void Geometry::commitAttributes(const char *_prefix, GeometryAttributes &attrs)
{
  std::string prefix = _prefix;

  attrs.attribute0 = getParamObject<Array1D>(prefix + "attribute0");
  attrs.attribute1 = getParamObject<Array1D>(prefix + "attribute1");
  attrs.attribute2 = getParamObject<Array1D>(prefix + "attribute2");
  attrs.attribute3 = getParamObject<Array1D>(prefix + "attribute3");
  attrs.color = getParamObject<Array1D>(prefix + "color");

  auto getUniformAttribute =
      [&](const std::string &pName) -> std::optional<vec4> {
    vec4 v(0.f, 0.f, 0.f, 1.f);
    if (getParam(pName, ANARI_FLOAT32_VEC4, &v))
      return v;
    else
      return {};
  };

  attrs.uniformAttribute0 = getUniformAttribute(prefix + "attribute0");
  attrs.uniformAttribute1 = getUniformAttribute(prefix + "attribute1");
  attrs.uniformAttribute2 = getUniformAttribute(prefix + "attribute2");
  attrs.uniformAttribute3 = getUniformAttribute(prefix + "attribute3");
  attrs.uniformColor = getUniformAttribute(prefix + "color");
}

void Geometry::populateAttributeDataSet(
    const GeometryAttributes &hostAttrs, AttributeDataSet &gpuAttrs) const
{
  populateAttributeData(
      hostAttrs.attribute0, gpuAttrs[0], hostAttrs.uniformAttribute0);
  populateAttributeData(
      hostAttrs.attribute1, gpuAttrs[1], hostAttrs.uniformAttribute1);
  populateAttributeData(
      hostAttrs.attribute2, gpuAttrs[2], hostAttrs.uniformAttribute2);
  populateAttributeData(
      hostAttrs.attribute3, gpuAttrs[3], hostAttrs.uniformAttribute3);
  populateAttributeData(hostAttrs.color, gpuAttrs[4], hostAttrs.uniformColor);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Geometry *);
