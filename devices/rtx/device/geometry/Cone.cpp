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

#include "Cone.h"

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

Cone::Cone(DeviceGlobalState *d)
    : Geometry(d),
      m_index(this),
      m_radius(this),
      m_vertex(this),
      m_vertexCaps(this)
{}

Cone::~Cone() = default;

void Cone::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("vertex.radius");
  m_defaultCapFlags = parseCapFlags(getParamString("caps", "none"));
  m_vertexCaps = getParamObject<Array1D>("vertex.cap");
  m_vertex = getParamObject<Array1D>("vertex.position");
  commitAttributes("vertex.", m_vertexAttributes);
}

PrimitiveInput Cone::primitiveInput() const
{
  PrimitiveInput input;
  input.type = GeometryType::CONE;
  input.count = m_index ? m_index->size() : m_vertex->size() / 2;
  input.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  input.indices =
      m_index ? m_index->beginAs<uvec2>(AddressSpace::GPU) : nullptr;
  input.radii =
      m_radius ? m_radius->beginAs<float>(AddressSpace::GPU) : nullptr;
  input.caps = m_vertexCaps ? m_vertexCaps->beginAs<uint8_t>(AddressSpace::GPU)
                            : nullptr;
  input.radius = 0.f;
  input.capFlags = m_defaultCapFlags;
  return input;
}

void Cone::finalize()
{
  if (!beginPreparation())
    return;
  if (!m_vertex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on cone geometry");
    return;
  }

  if (!m_radius) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.radius' on cone geometry");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing %s cone geometry",
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

void Cone::buildAreaData()
{
  m_areaDataValid =
      m_preprocessing.areas(this, primitiveInput(), m_primAreaCdf, m_totalArea);
}

bool Cone::isAreaSamplingSupported() const
{
  return true;
}

void Cone::ensureAreaData()
{
  m_areaDataWanted = true;
  if (m_areaDataValid)
    return;
  buildAreaData();
  if (m_areaDataValid)
    upload();
}

float Cone::totalArea() const
{
  return m_totalArea;
}

bool Cone::isValid() const
{
  return m_vertex && m_radius;
}

void Cone::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives =
      m_index ? m_index->size() : m_vertex->size() / 2;

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;
}

int Cone::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

GeometryGPUData Cone::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::CONE;

  auto &cone = retval.cone;
  cone.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  cone.indices = m_index ? m_index->beginAs<uvec2>(AddressSpace::GPU) : nullptr;
  cone.radii = m_radius->beginAs<float>(AddressSpace::GPU);
  cone.defaultCapFlags = m_defaultCapFlags;
  cone.vertexCaps = m_vertexCaps
      ? m_vertexCaps->beginAs<uint8_t>(AddressSpace::GPU)
      : nullptr;
  populateAttributeDataSet(m_vertexAttributes, cone.vertexAttr);

  // Geometry Light sampling data; null/zero until ensureAreaData() runs.
  cone.primAreaCdf = m_totalArea > 0.f ? m_primAreaCdf.ptrAs<float>() : nullptr;
  cone.numPrimitives = m_totalArea > 0.f ? uint32_t(primitiveInput().count) : 0;
  cone.totalArea = m_totalArea;

  return retval;
}

} // namespace visrtx
