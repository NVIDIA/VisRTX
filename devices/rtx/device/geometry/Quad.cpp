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

#include "Quad.h"

namespace visrtx {

Quad::Quad(DeviceGlobalState *d) : Geometry(d), m_index(this), m_vertex(this) {}

Quad::~Quad() = default;

void Quad::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertex = getParamObject<Array1D>("vertex.position");
  m_vertexNormal = getParamObject<Array1D>("vertex.normal");
  m_cullBackfaces = getParam<bool>("cullBackfaces", false);
  commitAttributes("vertex.", m_vertexAttributes);
}

void Quad::finalize()
{
  if (!m_vertex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on quad geometry");
    return;
  }

  if (!m_index && m_vertex->size() % 4 != 0) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "'vertex.position' on quad geometry is a non-multiple of 4"
        " without 'primitive.index' present");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing %s quad geometry",
      m_index ? "indexed" : "soup");

  generateIndices();
  m_vertexBufferPtr = (CUdeviceptr)m_vertex->beginAs<vec3>(AddressSpace::GPU);

  upload();
}

bool Quad::isValid() const
{
  return m_vertex;
}

void Quad::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  buildInput.triangleArray.vertexStrideInBytes = sizeof(vec3);
  buildInput.triangleArray.numVertices = m_vertex->size();
  buildInput.triangleArray.vertexBuffers = &m_vertexBufferPtr;

  buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  buildInput.triangleArray.indexStrideInBytes = sizeof(uvec3);
  buildInput.triangleArray.numIndexTriplets = m_indices.size();
  buildInput.triangleArray.indexBuffer = (CUdeviceptr)m_indices.dataDevice();

  static uint32_t buildInputFlags[1] = {0};

  buildInput.triangleArray.flags = buildInputFlags;
  buildInput.triangleArray.numSbtRecords = 1;
}

int Quad::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
}

GeometryGPUData Quad::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::QUAD;

  auto &quad = retval.quad;
  quad.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  quad.indices = m_indices.dataDevice();
  quad.vertexNormals = m_vertexNormal
      ? m_vertexNormal->beginAs<vec3>(AddressSpace::GPU)
      : nullptr;
  quad.cullBackfaces = m_cullBackfaces;
  populateAttributeDataSet(m_vertexAttributes, quad.vertexAttr);

  return retval;
}

void Quad::generateIndices()
{
  if (m_index) {
    size_t numIndices = 2 * m_index->size();
    m_indices.resize(numIndices);
    auto *indicesIn = (const uvec4 *)m_index->dataAs<uvec4>(AddressSpace::HOST);
    auto *indicesOut = m_indices.dataHost();
    for (size_t i = 0; i < m_index->size(); i++) {
      auto idx = indicesIn[i];
      indicesOut[2 * i + 0] = uvec3(idx.x, idx.y, idx.w);
      indicesOut[2 * i + 1] = uvec3(idx.z, idx.w, idx.y);
    }
  } else {
    size_t numQuad = m_vertex->size() / 4;
    m_indices.resize(2 * numQuad);
    auto *indicesOut = m_indices.dataHost();
    for (size_t i = 0; i < numQuad; i++) {
      indicesOut[2 * i + 0] = uvec3(4 * i) + uvec3(0, 1, 3);
      indicesOut[2 * i + 1] = uvec3(4 * i) + uvec3(2, 3, 1);
    }
  }

  m_indices.upload();
}

} // namespace visrtx
