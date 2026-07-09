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
        const float r = glm::max(std::abs(radius[c.x]), std::abs(radius[c.y]));
        return box3(glm::min(v0, v1) - r, glm::max(v0, v1) + r);
      });

  m_aabbs.upload();
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.dataDevice();

  // Coordinate scale of the intersector's arithmetic; floors hit.epsilon so
  // secondary rays clear the solve's fp noise band (see
  // GeometryGPUData::epsilonScale).
  m_epsilonScale = 0.f;
  for (auto it = m_aabbs.begin(); it != m_aabbs.end(); ++it) {
    const vec3 m = glm::max(glm::abs(it->lower), glm::abs(it->upper));
    m_epsilonScale =
        std::max(m_epsilonScale, std::max(m.x, std::max(m.y, m.z)));
  }

  // Vertices/radii/caps may have changed. Rebuild the Geometry Light CDF if a
  // surface has ever requested it (order-independent of the surface commit),
  // else drop stale data. Mirrors Triangle/Sphere.
  m_areaDataValid = false;
  if (m_areaDataWanted)
    buildAreaData();
  else {
    m_totalArea = 0.f;
    m_primAreaCdf.clear();
  }

  upload();
}

void Cone::buildAreaData()
{
  std::vector<uvec2> implicitIndices;
  Span<uvec2> indices;
  if (!m_index) {
    implicitIndices.resize(m_vertex->size() / 2);
    uvec2 idx(0, 1);
    for (auto &i : implicitIndices) {
      i = idx;
      idx += 2;
    }
    indices = make_Span(implicitIndices.data(), implicitIndices.size());
  } else
    indices = make_Span(m_index->beginAs<uvec2>(), m_index->size());

  const float *radius = m_radius->beginAs<float>();
  const vec3 *pos = m_vertex->beginAs<vec3>();
  const uint8_t *vcaps = m_vertexCaps ? m_vertexCaps->beginAs<uint8_t>() : nullptr;

  const size_t n = indices.size();
  m_primAreaCdf.resize(n);
  auto *cdf = m_primAreaCdf.dataHost();

  double cumulative = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const uvec2 c = indices[i];
    const double r0 = double(std::abs(radius[c.x]));
    const double r1 = double(std::abs(radius[c.y]));
    const double len = double(glm::length(pos[c.y] - pos[c.x]));
    const bool cap0 = vcaps ? (vcaps[c.x] != 0) : (m_defaultCapFlags & CAP_FIRST);
    const bool cap1 = vcaps ? (vcaps[c.y] != 0) : (m_defaultCapFlags & CAP_SECOND);
    // A degenerate (zero-length) cone gets no pick mass; the sampler
    // early-returns on it, so any mass here would be a wasted sample.
    double area = 0.0;
    if (len > 0.0) {
      const double slant = std::sqrt(len * len + (r0 - r1) * (r0 - r1));
      area = double(kPi) * (r0 + r1) * slant; // lateral (frustum)
      if (cap0)
        area += double(kPi) * r0 * r0;
      if (cap1)
        area += double(kPi) * r1 * r1;
    }
    cumulative += area;
    cdf[i] = float(cumulative);
  }
  m_totalArea = float(cumulative);

  if (m_totalArea > 0.f) {
    for (size_t i = 0; i < n; ++i)
      cdf[i] /= m_totalArea;
  }

  m_primAreaCdf.upload();
  m_areaDataValid = true;
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
  upload(); // republish gpuData() so the CDF pointers reach the device
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
  cone.defaultCapFlags = m_defaultCapFlags;
  cone.vertexCaps = m_vertexCaps
      ? m_vertexCaps->beginAs<uint8_t>(AddressSpace::GPU)
      : nullptr;
  populateAttributeDataSet(m_vertexAttributes, cone.vertexAttr);

  // Geometry Light sampling data; null/zero until ensureAreaData() runs.
  cone.primAreaCdf = m_primAreaCdf.dataDevice();
  cone.numPrimitives = uint32_t(m_primAreaCdf.size());
  cone.totalArea = m_totalArea;

  return retval;
}

} // namespace visrtx
