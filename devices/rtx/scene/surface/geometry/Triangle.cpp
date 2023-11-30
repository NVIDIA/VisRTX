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

#include "Triangle.h"

namespace visrtx {

Triangle::Triangle(DeviceGlobalState *d) : Geometry(d) {}

Triangle::~Triangle()
{
  cleanup();
}

void Triangle::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");

  m_vertex = getParamObject<Array1D>("vertex.position");
  m_vertexColor = getParamObject<Array1D>("vertex.color");
  m_vertexNormal = getParamObject<Array1D>("vertex.normal");
  m_vertexAttribute0 = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttribute1 = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttribute2 = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttribute3 = getParamObject<Array1D>("vertex.attribute3");

  m_vertexNormalFV = getParamObject<Array1D>("faceVarying.normal");
  m_vertexAttribute0FV = getParamObject<Array1D>("faceVarying.attribute0");
  m_vertexAttribute1FV = getParamObject<Array1D>("faceVarying.attribute1");
  m_vertexAttribute2FV = getParamObject<Array1D>("faceVarying.attribute2");
  m_vertexAttribute3FV = getParamObject<Array1D>("faceVarying.attribute3");
  m_vertexColorFV = getParamObject<Array1D>("faceVarying.color");

  if (!m_vertex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on triangle geometry");
    return;
  }

  if (!m_index && m_vertex->size() % 3 != 0) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "'vertex.position' on triangle geometry is a non-multiple of 3"
        " without 'primitive.index' present");
    return;
  }

  if (m_vertexNormal && m_vertex->size() != m_vertexNormal->size()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.normal' on triangle geometry not the same size as "
        "'vertex.position' (%zu) vs. (%zu)",
        m_vertexNormal->size(),
        m_vertex->size());
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "committing %s triangle geometry",
      m_index ? "indexed" : "soup");

  if (m_index)
    m_index->addCommitObserver(this);
  m_vertex->addCommitObserver(this);

  m_vertexBufferPtr = (CUdeviceptr)m_vertex->beginAs<vec3>(AddressSpace::GPU);

  upload();
}

void Triangle::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  buildInput.triangleArray.vertexStrideInBytes = sizeof(vec3);
  buildInput.triangleArray.numVertices = m_vertex->size();
  buildInput.triangleArray.vertexBuffers = &m_vertexBufferPtr;

  if (m_index) {
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uvec3);
    buildInput.triangleArray.numIndexTriplets = m_index->size();
    buildInput.triangleArray.indexBuffer =
        (CUdeviceptr)m_index->beginAs<uvec3>(AddressSpace::GPU);
  } else {
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    buildInput.triangleArray.indexStrideInBytes = 0;
    buildInput.triangleArray.numIndexTriplets = 0;
    buildInput.triangleArray.indexBuffer = 0;
  }

  static uint32_t buildInputFlags[1] = {0};

  buildInput.triangleArray.flags = buildInputFlags;
  buildInput.triangleArray.numSbtRecords = 1;
}

int Triangle::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
}

bool Triangle::isValid() const
{
  return m_vertex;
}

GeometryGPUData Triangle::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::TRIANGLE;

  auto &tri = retval.tri;

  tri.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  tri.indices = m_index ? m_index->beginAs<uvec3>(AddressSpace::GPU) : nullptr;

  tri.vertexNormals = m_vertexNormal
      ? m_vertexNormal->beginAs<vec3>(AddressSpace::GPU)
      : nullptr;

  populateAttributePtr(m_vertexAttribute0, tri.vertexAttr[0]);
  populateAttributePtr(m_vertexAttribute1, tri.vertexAttr[1]);
  populateAttributePtr(m_vertexAttribute2, tri.vertexAttr[2]);
  populateAttributePtr(m_vertexAttribute3, tri.vertexAttr[3]);
  populateAttributePtr(m_vertexColor, tri.vertexAttr[4]);

  tri.vertexNormalsFV = m_vertexNormalFV
      ? m_vertexNormalFV->beginAs<vec3>(AddressSpace::GPU)
      : nullptr;

  populateAttributePtr(m_vertexAttribute0FV, tri.vertexAttrFV[0]);
  populateAttributePtr(m_vertexAttribute1FV, tri.vertexAttrFV[1]);
  populateAttributePtr(m_vertexAttribute2FV, tri.vertexAttrFV[2]);
  populateAttributePtr(m_vertexAttribute3FV, tri.vertexAttrFV[3]);
  populateAttributePtr(m_vertexColorFV, tri.vertexAttrFV[4]);

  return retval;
}

void Triangle::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertex)
    m_vertex->removeCommitObserver(this);
}

} // namespace visrtx
