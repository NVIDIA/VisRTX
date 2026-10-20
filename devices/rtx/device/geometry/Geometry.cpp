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

#include "Geometry.h"
// specific types
#include "Cone.h"
#include "Curve.h"
#include "Cylinder.h"
#include "Isosurface.h"
#include "Quad.h"
#include "Sphere.h"
#include "Triangle.h"

#ifdef VISRTX_USE_NEURAL
#include "Neural.h"
#endif
#include "SDF.h"
#include "UnknownGeometry.h"
// std
#include <cstring>
#include <string_view>
// helium
#include <helium/helium_math.h>

#include "utility/AnariTypeHelpers.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

static void populateAttributeData(
    helium::IntrusivePtr<Array1D> array, AttributeData &attr)
{
  attr.type = ANARI_UNKNOWN;
  attr.numChannels = 0;
  attr.data = nullptr;

  if (!array)
    return;

  auto type = array->elementType();

  if (!isColor(type))
    return;

  attr.type = type;
  attr.numChannels = numANARIChannels(attr.type);
  attr.data = array->dataGPU();
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
  else if (subtype == "sdf")
    return new SDF(d);
  else if (subtype == "isosurface")
    return new Isosurface(d);
#ifdef VISRTX_USE_NEURAL
  else if (subtype == "neural")
    return new Neural(d);
#endif
  else
    return new UnknownGeometry(subtype, d);
}

void Geometry::commitParameters()
{
  // Array observers can re-finalize without a parameter commit, but parameter
  // replacement can release the last reference to an aliased device array.
  // Complete earlier consumers before subclasses replace those references.
  m_inputRetirementFailed = !checkPreparation(this,
      cudaStreamSynchronize(deviceState()->stream),
      "retiring previous geometry inputs");
  commitAttributes("primitive.", m_primitiveAttributes);

  auto getUniformAttribute =
      [&](const std::string &pName) -> std::optional<vec4> {
    vec4 v(0.f, 0.f, 0.f, 1.f);
    if (getParam(pName, ANARI_FLOAT32_VEC4, &v))
      return v;
    else
      return {};
  };

  m_uniformAttributes.attribute0 = getUniformAttribute("attribute0");
  m_uniformAttributes.attribute1 = getUniformAttribute("attribute1");
  m_uniformAttributes.attribute2 = getUniformAttribute("attribute2");
  m_uniformAttributes.attribute3 = getUniformAttribute("attribute3");
  m_uniformAttributes.color = getUniformAttribute("color");
  m_primitiveId = getParamObject<Array1D>("primitive.id");
}

void Geometry::markFinalized()
{
  Object::markFinalized();
  if (!m_preparationPreserved)
    deviceState()->objectUpdates.lastSurfaceBLASChange = helium::newTimeStamp();
}

bool Geometry::isAreaSamplingSupported() const
{
  return false;
}

bool Geometry::PreparationParameter::operator==(
    const PreparationParameter &other) const
{
  return parameterIndex == other.parameterIndex && value == other.value
      && updated == other.updated && finalized == other.finalized
      && modified == other.modified && devicePointer == other.devicePointer;
}

void Geometry::upload()
{
  RegisteredObject<GeometryGPUData>::upload();
  m_hasPreparedData = !preparationFailed() && isValid();
}

bool Geometry::beginPreparation(std::initializer_list<const char *> parameters)
{
  m_preparationPreserved = false;
  if (m_inputRetirementFailed)
    return false;

  std::vector<PreparationParameter> current;
  if (parameters.size() != 0) {
    current.reserve(m_preparedParameters.size());
    size_t parameterIndex = 0;
    auto append = [&](const char *name) {
      PreparationParameter parameter;
      parameter.parameterIndex = parameterIndex++;
      parameter.value = getParamDirect(name); // committed snapshot, not staging
      if (!parameter.value.valid())
        return; // Keep only present values, retaining their parameter identity.
      if (anari::isObject(parameter.value.type())) {
        auto *object = parameter.value.getObject<helium::BaseObject>();
        if (object) {
          parameter.updated = object->lastUpdated();
          parameter.finalized = object->lastFinalized();
          if (auto *array = dynamic_cast<Array *>(object)) {
            parameter.modified = array->lastDataModified();
            // Privatization can change backing pointers without changing values
            // or the array timestamp. Never reuse cached aliases in that case.
            parameter.devicePointer = array->dataGPU();
          }
        }
      }
      current.push_back(std::move(parameter));
    };
    for (const char *name : {"primitive.attribute0",
             "primitive.attribute1",
             "primitive.attribute2",
             "primitive.attribute3",
             "primitive.color",
             "primitive.id",
             "vertex.attribute0",
             "vertex.attribute1",
             "vertex.attribute2",
             "vertex.attribute3",
             "vertex.color"})
      append(name);
    for (const char *name : parameters)
      append(name);

    // Observer-only updates always take the preparation path. A geometry-wide
    // attribute edit coalesced with an array change cannot hide that change;
    // the versions/pointers above still invalidate preparation reuse.
    if (m_hasPreparedData && !preparationFailed() && isValid()
        && lastCommitted() > lastFinalized()
        && current == m_preparedParameters) {
      m_preparationPreserved = true;
      upload();
      return false;
    }
  }
  m_preparedParameters = std::move(current);
  m_hasPreparedData = false;
  m_preparationFailed = false;
  return true;
}

bool Geometry::preparationFailed() const
{
  return m_preparationFailed;
}

void Geometry::setPreparationFailed(bool failed)
{
  m_preparationFailed = failed;
}

void Geometry::ensureAreaData() {}

float Geometry::totalArea() const
{
  return 0.f;
}

GeometryGPUData Geometry::gpuData() const
{
  GeometryGPUData retval{};

  const vec4 defaultAttr(0.f, 0.f, 0.f, 1.f);
  retval.attrUniform[0] = m_uniformAttributes.attribute0.value_or(defaultAttr);
  retval.attrUniform[1] = m_uniformAttributes.attribute1.value_or(defaultAttr);
  retval.attrUniform[2] = m_uniformAttributes.attribute2.value_or(defaultAttr);
  retval.attrUniform[3] = m_uniformAttributes.attribute3.value_or(defaultAttr);
  retval.attrUniform[4] = m_uniformAttributes.color.value_or(defaultAttr);
  retval.attrUniformPresent[0] = m_uniformAttributes.attribute0.has_value();
  retval.attrUniformPresent[1] = m_uniformAttributes.attribute1.has_value();
  retval.attrUniformPresent[2] = m_uniformAttributes.attribute2.has_value();
  retval.attrUniformPresent[3] = m_uniformAttributes.attribute3.has_value();
  retval.attrUniformPresent[4] = m_uniformAttributes.color.has_value();
  populateAttributeDataSet(m_primitiveAttributes, retval.attr);
  retval.primitiveId =
      (const uint32_t *)(m_primitiveId ? m_primitiveId->dataGPU() : nullptr);
  retval.epsilonScale = m_epsilonScale;

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
}

void Geometry::populateAttributeDataSet(
    const GeometryAttributes &hostAttrs, AttributeDataSet &gpuAttrs) const
{
  populateAttributeData(hostAttrs.attribute0, gpuAttrs[0]);
  populateAttributeData(hostAttrs.attribute1, gpuAttrs[1]);
  populateAttributeData(hostAttrs.attribute2, gpuAttrs[2]);
  populateAttributeData(hostAttrs.attribute3, gpuAttrs[3]);
  populateAttributeData(hostAttrs.color, gpuAttrs[4]);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Geometry *);
