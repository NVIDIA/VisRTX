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

#include "Cones.h"
// glm
#include <glm/gtx/rotate_vector.hpp>

namespace visrtx {

static mat4 orientConeMatrix(vec3 dir)
{
  const vec3 v = glm::vec3(0, -dir.z, dir.y);
  const float angle = acos(dir.x / glm::length(dir));
  return glm::rotate(angle, v);
}

static void appendCone(const vec3 &vtx0,
    const vec3 &vtx1,
    float r0,
    float r1,
    bool addCaps,
    std::vector<vec3> &vertices,
    std::vector<uvec3> &indices)
{
  unsigned int baseOffset = vertices.size();

  auto dir = vtx1 - vtx0;
  auto m = orientConeMatrix(dir);

  auto v00 = vec4(0.f, r0, 0.f, 1.f);
  auto v01 = vec4(0.f, 0.f, -r0, 1.f);
  auto v02 = vec4(0.f, -r0, 0.f, 1.f);
  auto v03 = vec4(0.f, 0.f, r0, 1.f);

  auto v10 = vec4(0.f, r1, 0.f, 1.f);
  auto v11 = vec4(0.f, 0.f, -r1, 1.f);
  auto v12 = vec4(0.f, -r1, 0.f, 1.f);
  auto v13 = vec4(0.f, 0.f, r1, 1.f);

  vertices.push_back(vec3(m * v00) + vtx0); // 0
  vertices.push_back(vec3(m * v01) + vtx0); // 1
  vertices.push_back(vec3(m * v02) + vtx0); // 2
  vertices.push_back(vec3(m * v03) + vtx0); // 3

  vertices.push_back(vec3(m * v10) + vtx1); // 4
  vertices.push_back(vec3(m * v11) + vtx1); // 5
  vertices.push_back(vec3(m * v12) + vtx1); // 6
  vertices.push_back(vec3(m * v13) + vtx1); // 7

  indices.push_back(baseOffset + uvec3(5, 1, 4));
  indices.push_back(baseOffset + uvec3(0, 4, 1));

  indices.push_back(baseOffset + uvec3(1, 5, 2));
  indices.push_back(baseOffset + uvec3(6, 2, 5));

  indices.push_back(baseOffset + uvec3(2, 6, 3));
  indices.push_back(baseOffset + uvec3(7, 3, 6));

  indices.push_back(baseOffset + uvec3(3, 7, 0));
  indices.push_back(baseOffset + uvec3(4, 0, 7));

  if (addCaps) {
    indices.push_back(baseOffset + uvec3(0, 1, 2));
    indices.push_back(baseOffset + uvec3(2, 3, 0));

    indices.push_back(baseOffset + uvec3(4, 5, 6));
    indices.push_back(baseOffset + uvec3(6, 7, 4));
  }
}

Cones::~Cones()
{
  cleanup();
}

void Cones::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("vertex.radius");
  m_caps = getParam<std::string>("caps", "none") != "none";

  m_vertex = getParamObject<Array1D>("vertex.position");

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

  if (m_index)
    m_index->addCommitObserver(this);
  m_vertex->addCommitObserver(this);
  m_radius->addCommitObserver(this);

  generateCones();
}

void Cones::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  buildInput.triangleArray.vertexStrideInBytes = sizeof(vec3);
  buildInput.triangleArray.numVertices = m_cones.vertices.size();
  buildInput.triangleArray.vertexBuffers = &m_cones.vertexBufferPtr;

  buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  buildInput.triangleArray.indexStrideInBytes = sizeof(uvec3);
  buildInput.triangleArray.numIndexTriplets = m_cones.indices.size();
  buildInput.triangleArray.indexBuffer = (CUdeviceptr)m_cones.indexBuffer.ptr();

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  buildInput.triangleArray.flags = buildInputFlags;
  buildInput.triangleArray.numSbtRecords = 1;
}

int Cones::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
}

GeometryGPUData Cones::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::CONE;

  auto &cone = retval.cone;

  cone.vertices = (const vec3 *)m_cones.vertexBuffer.ptr();
  cone.indices = (const uvec3 *)m_cones.indexBuffer.ptr();
  cone.trianglesPerCone = m_caps ? 12 : 8;

  return retval;
}

void Cones::generateCones()
{
  std::vector<uvec2> implicitIndices;
  anari::Span<uvec2> indices;

  if (!m_index) {
    implicitIndices.resize(m_vertex->size() / 2);
    uvec2 idx(0, 1);
    std::for_each(
        implicitIndices.begin(), implicitIndices.end(), [&](uvec2 &i) {
          i = idx;
          idx += 2;
        });
    indices = anari::make_Span(implicitIndices.data(), implicitIndices.size());
  } else {
    indices = anari::make_Span(m_index->beginAs<uvec2>(), m_index->size());
  }

  m_cones.vertices.clear();
  m_cones.indices.clear();

  {
    auto *begin = indices.begin();
    auto *end = indices.end();

    auto *radius = m_radius->beginAs<float>();
    auto *vertex = m_vertex->beginAs<vec3>();

    size_t coneID = 0;
    std::for_each(begin, end, [&](const uvec2 &i) {
      const auto v0 = vertex[i.x];
      const auto v1 = vertex[i.y];
      const auto r0 = radius[i.x];
      const auto r1 = radius[i.y];
      appendCone(v0, v1, r0, r1, m_caps, m_cones.vertices, m_cones.indices);
    });
  }

  m_cones.vertexBuffer.upload(m_cones.vertices);
  m_cones.indexBuffer.upload(m_cones.indices);
  m_cones.vertexBufferPtr = (CUdeviceptr)m_cones.vertexBuffer.ptr();
}

void Cones::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertex)
    m_vertex->removeCommitObserver(this);
  if (m_radius)
    m_radius->removeCommitObserver(this);
}

} // namespace visrtx
