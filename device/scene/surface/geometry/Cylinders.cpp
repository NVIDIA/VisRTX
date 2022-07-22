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

#include "Cylinders.h"

namespace visrtx {

Cylinders::~Cylinders()
{
  cleanup();
}

void Cylinders::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("primitive.radius");
  m_caps = getParam<std::string>("caps", "none") != "none";

  m_vertex = getParamObject<Array1D>("vertex.position");
  m_vertexColor = getParamObject<Array1D>("vertex.color");
  m_vertexAttribute0 = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttribute1 = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttribute2 = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttribute3 = getParamObject<Array1D>("vertex.attribute3");

  if (!m_vertex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on cylinders geometry");
    return;
  }

  if (m_index)
    m_index->addCommitObserver(this);
  m_vertex->addCommitObserver(this);

  if (hasParam("radius"))
    m_globalRadius = getParam<float>("radius", 1.f);
  else
    m_globalRadius.reset();

  float globalRadius = m_globalRadius.value_or(1.f);

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

  float *radius = nullptr;
  if (m_radius)
    radius = m_radius->beginAs<float>();

  m_aabbs.resize(indices.size());

  const auto *posBegin = m_vertex->beginAs<vec3>();
  size_t cylinderID = 0;
  std::transform(
      indices.begin(), indices.end(), m_aabbs.begin(), [&](const uvec2 &v) {
        const float r = radius ? radius[cylinderID++] : globalRadius;
        const vec3 &v1 = *(posBegin + v.x);
        const vec3 &v2 = *(posBegin + v.y);
        box3 bounds = box3(v1 - r, v1 + r);
        bounds.extend(box3(v2 - r, v2 + r));
        return bounds;
      });

  m_aabbs.upload();
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.dataDevice();
}

void Cylinders::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives = m_aabbs.size();

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;
}

GeometryGPUData Cylinders::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::CYLINDER;

  auto &cylinder = retval.cylinder;

  cylinder.vertices = m_vertex->beginAs<vec3>(AddressSpace::GPU);
  cylinder.indices =
      m_index ? m_index->beginAs<uvec2>(AddressSpace::GPU) : nullptr;
  cylinder.radii =
      m_radius ? m_radius->beginAs<float>(AddressSpace::GPU) : nullptr;
  cylinder.radius = m_globalRadius.value_or(1.f);
  cylinder.caps = m_caps;

  populateAttributePtr(m_vertexAttribute0, cylinder.vertexAttr[0]);
  populateAttributePtr(m_vertexAttribute1, cylinder.vertexAttr[1]);
  populateAttributePtr(m_vertexAttribute2, cylinder.vertexAttr[2]);
  populateAttributePtr(m_vertexAttribute3, cylinder.vertexAttr[3]);
  populateAttributePtr(m_vertexColor, cylinder.vertexAttr[4]);

  return retval;
}

int Cylinders::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

void Cylinders::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertex)
    m_vertex->removeCommitObserver(this);
}

} // namespace visrtx
