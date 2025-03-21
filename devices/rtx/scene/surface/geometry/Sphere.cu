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
#include <thrust/sequence.h>
#include <thrust/transform.h>

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

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing %s sphere geometry",
      m_index ? "indexed" : "soup");

  // Calculate bounds //

  m_numSpheres = m_index ? m_index->size() : m_vertex->size();
  m_aabbs.reserve(m_numSpheres * sizeof(box3));

  const float globalRadius = m_globalRadius;
  const float *radii = nullptr;
  if (m_vertexRadius)
    radii = m_vertexRadius->beginAs<float>(AddressSpace::GPU);

  const auto *vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);

  auto &state = *deviceState();

  if (m_index) {
    auto *begin = m_index->beginAs<uint32_t>(AddressSpace::GPU);
    auto *end = m_index->endAs<uint32_t>(AddressSpace::GPU);
    thrust::transform(thrust::cuda::par.on(state.stream),
        begin,
        end,
        thrust::device_pointer_cast<box3>((box3 *)m_aabbs.ptr()),
        [=] __device__(uint32_t i) {
          const auto &v = vertices[i];
          const float r = radii ? radii[i] : globalRadius;
          return box3(v - r, v + r);
        });
  } else {
    DeviceBuffer index;
    index.reserve(m_numSpheres * sizeof(uint32_t));
    auto idx_begin =
        thrust::device_pointer_cast<uint32_t>((uint32_t *)index.ptr());
    auto idx_end = idx_begin + m_numSpheres;
    thrust::sequence(thrust::cuda::par.on(state.stream), idx_begin, idx_end);
    thrust::transform(thrust::cuda::par.on(state.stream),
        idx_begin,
        idx_end,
        thrust::device_pointer_cast<box3>((box3 *)m_aabbs.ptr()),
        [=] __device__(uint32_t i) {
          const auto &v = vertices[i];
          const float r = radii ? radii[i] : globalRadius;
          return box3(v - r, v + r);
        });
  }

  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.ptr();

  upload();
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

  return retval;
}

int Sphere::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

} // namespace visrtx
