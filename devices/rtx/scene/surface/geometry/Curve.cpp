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

#include "Curve.h"
// std
#include <numeric>

namespace visrtx {

Curve::Curve(DeviceGlobalState *d)
    : Geometry(d), m_index(this), m_vertexPosition(this), m_vertexRadius(this)
{}

Curve::~Curve() = default;

void Curve::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");
  m_globalRadius = getParam<float>("radius", 1.f);
  commitAttributes("vertex.", m_vertexAttributes);
}

void Curve::finalize()
{
  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on curve geometry");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing %s curve geometry",
      m_index ? "indexed" : "soup");

  computeIndices();
  computeRadii();

  m_vertexBufferPtr = (CUdeviceptr)m_vertexPosition->begin(AddressSpace::GPU);
  m_radiusBufferPtr = (CUdeviceptr)m_generatedRadii.dataDevice();

  upload();
}

void Curve::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

  auto &curveArray = buildInput.curveArray;
  curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
  curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;

  curveArray.vertexStrideInBytes = sizeof(vec3);
  curveArray.numVertices = m_vertexPosition->size();
  curveArray.vertexBuffers = &m_vertexBufferPtr;

  curveArray.widthStrideInBytes = sizeof(float);
  curveArray.widthBuffers = &m_radiusBufferPtr;

  curveArray.indexStrideInBytes = sizeof(uint32_t);
  curveArray.numPrimitives = m_generatedIndices.size();
  curveArray.indexBuffer = (CUdeviceptr)m_generatedIndices.dataDevice();

  curveArray.normalBuffers = 0;
  curveArray.normalStrideInBytes = 0;
}

int Curve::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CURVES;
}

bool Curve::isValid() const
{
  return m_vertexPosition;
}

void Curve::computeIndices()
{
  if (m_index) {
    m_generatedIndices.resize(m_index->size());
    std::copy(m_index->beginAs<uint32_t>(),
        m_index->endAs<uint32_t>(),
        m_generatedIndices.begin());
  } else {
    m_generatedIndices.resize(m_vertexPosition->size() / 2);
    std::iota(m_generatedIndices.begin(), m_generatedIndices.end(), 0);
  }
  m_generatedIndices.upload();
}

void Curve::computeRadii()
{
  if (m_vertexRadius) {
    m_generatedRadii.resize(m_vertexRadius->totalCapacity());
    std::copy(m_vertexRadius->dataAs<float>(),
        m_vertexRadius->dataAs<float>() + m_vertexRadius->totalCapacity(),
        m_generatedRadii.begin());
  } else {
    m_generatedRadii.resize(m_vertexPosition->size());
    std::fill(m_generatedRadii.begin(), m_generatedRadii.end(), m_globalRadius);
  }
  m_generatedRadii.upload();
}

GeometryGPUData Curve::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::CURVE;

  auto &curve = retval.curve;
  curve.vertices = m_vertexPosition->beginAs<vec3>(AddressSpace::GPU);
  curve.indices = m_generatedIndices.dataDevice();
  curve.radii = m_generatedRadii.dataDevice();
  populateAttributeDataSet(m_vertexAttributes, curve.vertexAttr);

  return retval;
}

} // namespace visrtx
