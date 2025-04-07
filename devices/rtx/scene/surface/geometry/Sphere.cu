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

#include "Sphere.h"
// thrust
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

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
  if (!m_vertex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on sphere geometry");
    return;
  }

  m_numSpheres = m_vertex->size();

  if (m_index) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "indexed spheres not yet implemented -- interpreting as sphere soup.");
  }

  m_vertexBufferPtr = (CUdeviceptr)m_vertex->beginAs<vec3>(AddressSpace::GPU);

  if (m_vertexRadius) {
    m_radiiBufferPtr =
        (CUdeviceptr)m_vertexRadius->beginAs<float>(AddressSpace::GPU);
    m_radii.reset();
  } else {
    m_radii.reserve(m_numSpheres * sizeof(float));
    auto *begin = m_radii.ptrAs<float>();
    auto *end = begin + m_numSpheres;
    thrust::fill(thrust::cuda::par.on(deviceState()->stream),
        begin,
        end,
        m_globalRadius);
    m_radiiBufferPtr = (CUdeviceptr)begin;
  }

  upload();
}

bool Sphere::isValid() const
{
  return m_vertex;
}

void Sphere::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

  auto &sphereArray = buildInput.sphereArray;

  sphereArray.vertexBuffers = &m_vertexBufferPtr;
  sphereArray.vertexStrideInBytes = 0;
  sphereArray.numVertices = m_numSpheres;
  sphereArray.radiusBuffers = &m_radiiBufferPtr;
  sphereArray.radiusStrideInBytes = 0;
  sphereArray.singleRadius = 0;

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  sphereArray.flags = buildInputFlags;
  sphereArray.numSbtRecords = 1;
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

  return retval;
}

int Sphere::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_SPHERES;
}

} // namespace visrtx
