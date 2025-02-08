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

#include "Cone.h"

namespace visrtx {

Cone::Cone(DeviceGlobalState *d)
    : Geometry(d), m_index(this), m_radius(this), m_vertex(this)
{}

Cone::~Cone() = default;

void Cone::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("vertex.radius");
  m_caps = getParamString("caps", "none") != "none";
  m_vertex = getParamObject<Array1D>("vertex.position");
  commitAttributes("vertex.", m_vertexAttributes);
}

void Cone::finalize()
{
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

  std::vector<uvec2> implicitIndices;
  Span<uvec2> indices;

  if (!m_index) {
    implicitIndices.resize(m_vertex->size() / 2);
    uvec2 idx(0, 1);
    std::for_each(
        implicitIndices.begin(), implicitIndices.end(), [&](uvec2 &i) {
          i = idx;
          idx += 2;
        });
    indices = make_Span(implicitIndices.data(), implicitIndices.size());
  } else {
    indices = make_Span(m_index->beginAs<uvec2>(), m_index->size());
  }

  const float *radius = m_radius->beginAs<float>();

  m_aabbs.resize(indices.size());

  const auto *posBegin = m_vertex->beginAs<vec3>();
  std::transform(
      indices.begin(), indices.end(), m_aabbs.begin(), [&](const uvec2 &c) {
        const vec3 &v0 = posBegin[c.x];
        const vec3 &v1 = posBegin[c.y];
        const float &r0 = radius[c.x];
        const float &r1 = radius[c.y];
        return box3(glm::min(v0, v1) - glm::max(r0, r1),
            glm::max(v0, v1) + glm::max(r0, r1));
      });

  m_aabbs.upload();
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.dataDevice();

  upload();
}

bool Cone::isValid() const
{
  return m_vertex && m_radius;
}

void Cone::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives = m_aabbs.size();

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
  populateAttributeDataSet(m_vertexAttributes, cone.vertexAttr);

  return retval;
}

} // namespace visrtx
