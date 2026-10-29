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

#include "Cylinder.h"

#include "gpu/intersectPrimitives.h"

namespace visrtx {

// ANARI caps string -> per-endpoint bitmask (none/first/second/both).
static uint8_t parseCapFlags(const std::string &s)
{
  if (s == "both")
    return CAP_FIRST | CAP_SECOND;
  if (s == "first")
    return CAP_FIRST;
  if (s == "second")
    return CAP_SECOND;
  return 0; // "none"
}

Cylinder::Cylinder(DeviceGlobalState *d)
    : Geometry(d),
      m_index(this),
      m_radius(this),
      m_vertex(this),
      m_vertexCaps(this)
{
  declarePreparationParameters({"primitive.index",
      "vertex.position",
      "primitive.radius",
      "vertex.cap",
      "caps",
      "radius"});
}

Cylinder::~Cylinder() = default;

void Cylinder::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("primitive.radius");
  m_defaultCapFlags = parseCapFlags(getParamString("caps", "none"));
  m_vertexCaps = getParamObject<Array1D>("vertex.cap");
  m_vertex = getParamObject<Array1D>("vertex.position");
  m_globalRadius = getParam<float>("radius", 1.f);
  commitAttributes("vertex.", m_vertexAttributes);
}

PrimitiveInput Cylinder::primitiveInput() const
{
  PrimitiveInput input;
  input.type = GeometryType::CYLINDER;
  input.count = m_index ? m_index->size() : m_vertex->size() / 2;
  input.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  input.indices =
      m_index ? m_index->beginAs<uvec2>(AddressSpace::GPU) : nullptr;
  input.radii =
      m_radius ? m_radius->beginAs<float>(AddressSpace::GPU) : nullptr;
  input.caps = m_vertexCaps ? m_vertexCaps->beginAs<uint8_t>(AddressSpace::GPU)
                            : nullptr;
  input.radius = m_globalRadius;
  input.capFlags = m_defaultCapFlags;
  return input;
}

void Cylinder::finalize()
{
  if (!beginPreparation())
    return;
  if (!m_vertex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on cylinders geometry");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing %s cylinder geometry",
      m_index ? "indexed" : "soup");

  if (!m_preprocessing.bounds(this, primitiveInput(), m_aabbs, m_epsilonScale))
    return;
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.ptr();

  // Vertices/radii/caps may have changed. Rebuild the Geometry Light CDF if a
  // surface has ever requested it (order-independent of the surface commit),
  // else drop stale data. Mirrors Triangle/Sphere.
  m_areaDataValid = false;
  if (m_areaDataWanted)
    buildAreaData();
  else {
    m_totalArea = 0.f;
    m_primAreaCdf.reset();
  }

  if (!preparationFailed())
    upload();
}

void Cylinder::buildAreaData()
{
  m_areaDataValid =
      m_preprocessing.areas(this, primitiveInput(), m_primAreaCdf, m_totalArea);
}

bool Cylinder::isAreaSamplingSupported() const
{
  return true;
}

void Cylinder::ensureAreaData()
{
  m_areaDataWanted = true;
  if (m_areaDataValid)
    return;
  buildAreaData();
  if (m_areaDataValid)
    upload();
}

float Cylinder::totalArea() const
{
  return m_totalArea;
}

bool Cylinder::isValid() const
{
  return m_vertex;
}

void Cylinder::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives =
      m_index ? m_index->size() : m_vertex->size() / 2;

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;
}

GeometryGPUData Cylinder::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::CYLINDER;

  auto &cylinder = retval.cylinder;
  cylinder.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  cylinder.indices =
      m_index ? m_index->beginAs<uvec2>(AddressSpace::GPU) : nullptr;
  cylinder.radii =
      m_radius ? m_radius->beginAs<float>(AddressSpace::GPU) : nullptr;
  cylinder.radius = m_globalRadius;
  cylinder.defaultCapFlags = m_defaultCapFlags;
  cylinder.vertexCaps = m_vertexCaps
      ? m_vertexCaps->beginAs<uint8_t>(AddressSpace::GPU)
      : nullptr;
  populateAttributeDataSet(m_vertexAttributes, cylinder.vertexAttr);

  // Geometry Light sampling data; null/zero until ensureAreaData() runs.
  cylinder.primAreaCdf =
      m_totalArea > 0.f ? m_primAreaCdf.ptrAs<float>() : nullptr;
  cylinder.numPrimitives =
      m_totalArea > 0.f ? uint32_t(primitiveInput().count) : 0;
  cylinder.totalArea = m_totalArea;

  return retval;
}

int Cylinder::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

} // namespace visrtx
