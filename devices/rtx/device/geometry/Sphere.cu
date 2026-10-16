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

#include "Sphere.h"

namespace visrtx {

Sphere::Sphere(DeviceGlobalState *d)
    : Geometry(d), m_index(this), m_vertex(this), m_vertexRadius(this)
{}

Sphere::~Sphere() = default;

void Sphere::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertex = getParamObject<Array1D>("vertex.position");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");
  m_globalRadius = getParam<float>("radius", 0.01f);
  commitAttributes("vertex.", m_vertexAttributes);
}

void Sphere::finalize()
{
  if (!beginPreparation())
    return;
  if (!m_vertex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on sphere geometry");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing %s sphere geometry",
      m_index ? "indexed" : "soup");

  // Calculate bounds //

  m_numSpheres = m_index ? m_index->size() : m_vertex->size();
  PrimitiveInput input;
  input.type = GeometryType::SPHERE;
  input.count = m_numSpheres;
  input.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  input.indices =
      m_index ? m_index->beginAs<uint32_t>(AddressSpace::GPU) : nullptr;
  input.radii = m_vertexRadius
      ? m_vertexRadius->beginAs<float>(AddressSpace::GPU)
      : nullptr;
  input.radius = m_globalRadius;
  if (!m_preprocessing.bounds(this, input, m_aabbs, m_epsilonScale))
    return;
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.ptr();

  // Radii may have changed. Rebuild the Geometry Light CDF if a surface has
  // ever requested it (order-independent of the surface commit), else drop
  // stale data.
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

void Sphere::buildAreaData()
{
  PrimitiveInput input;
  input.type = GeometryType::SPHERE;
  input.count = m_numSpheres;
  input.radii = m_vertexRadius
      ? m_vertexRadius->beginAs<float>(AddressSpace::GPU)
      : nullptr;
  input.indices =
      m_index ? m_index->beginAs<uint32_t>(AddressSpace::GPU) : nullptr;
  input.radius = m_globalRadius;
  m_areaDataValid =
      m_preprocessing.areas(this, input, m_primAreaCdf, m_totalArea);
}

bool Sphere::isAreaSamplingSupported() const
{
  return true;
}

void Sphere::ensureAreaData()
{
  m_areaDataWanted = true;
  if (m_areaDataValid)
    return;
  buildAreaData();
  if (m_areaDataValid)
    upload();
}

float Sphere::totalArea() const
{
  return m_totalArea;
}

bool Sphere::isValid() const
{
  return m_vertex;
}

void Sphere::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives = m_numSpheres;

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;
}

GeometryGPUData Sphere::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::SPHERE;

  auto &sphere = retval.sphere;
  sphere.centers = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  sphere.indices = nullptr;
  if (m_index)
    sphere.indices = m_index->beginAs<uint32_t>(AddressSpace::GPU);
  sphere.radii = nullptr;
  if (m_vertexRadius)
    sphere.radii = m_vertexRadius->beginAs<float>(AddressSpace::GPU);
  sphere.radius = m_globalRadius;
  populateAttributeDataSet(m_vertexAttributes, sphere.vertexAttr);

  // Geometry Light sampling data; null/zero until ensureAreaData() runs.
  sphere.primAreaCdf =
      m_totalArea > 0.f ? m_primAreaCdf.ptrAs<float>() : nullptr;
  sphere.numPrimitives = m_totalArea > 0.f ? uint32_t(m_numSpheres) : 0;
  sphere.totalArea = m_totalArea;

  return retval;
}

int Sphere::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

} // namespace visrtx
