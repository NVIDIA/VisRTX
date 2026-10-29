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

#include "SDF.h"
// glm
#include <glm/glm.hpp>
// std
#include <algorithm>

static_assert(
    sizeof(visrtx::SDFPrimitive) == 72, "SDFPrimitive layout must be 72 bytes");
static_assert(offsetof(visrtx::SDFPrimitive, neighboursIndex) == 56,
    "SDFPrimitive::neighboursIndex must be at offset 56");

namespace visrtx {

SDF::SDF(DeviceGlobalState *d) : Geometry(d), m_sdf(this), m_neighbour(this) {}

SDF::~SDF() = default;

void SDF::commitParameters()
{
  Geometry::commitParameters();
  m_sdf = getParamObject<Array1D>("primitive.sdf");
  m_neighbour = getParamObject<Array1D>("primitive.neighbor");
  m_epsilon = getParam<float>("epsilon", 1e-5f);
  m_nbMarchIterations = static_cast<uint32_t>(
      glm::max(getParam<int32_t>("nbMarchIterations", 16), 1));
  m_blendFactor = getParam<float>("blendFactor", 1.f);
  m_blendLerpFactor = getParam<float>("blendLerpFactor", 0.5f);
  m_omega = glm::clamp(getParam<float>("omega", 1.f), 0.f, 1.f);
  m_distanceFromCamera = getParam<float>("distanceFromCamera", 100.f);
  m_noiseFactor = glm::clamp(getParam<float>("noiseFactor", 0.f), 0.f, 1.f);
}

void SDF::finalize()
{
  if (!beginPreparation({"primitive.sdf",
          "primitive.neighbor",
          "epsilon",
          "nbMarchIterations",
          "blendFactor",
          "blendLerpFactor",
          "omega",
          "distanceFromCamera",
          "noiseFactor"}))
    return;
  if (!m_sdf) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'primitive.sdf' on sdf geometry");
    return;
  }

  // primitive.sdf must be an ANARI_UINT8 array of N * sizeof(SDFPrimitive)
  // bytes.
  m_numPrimitives = m_sdf->size() / sizeof(SDFPrimitive);

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing sdf geometry with %zu primitives",
      m_numPrimitives);

  PrimitiveInput input;
  input.type = GeometryType::SDF;
  input.count = m_numPrimitives;
  const void *source = m_sdf->begin(AddressSpace::GPU);
  if (reinterpret_cast<uintptr_t>(source) % alignof(SDFPrimitive) != 0) {
    const size_t bytes = m_numPrimitives * sizeof(SDFPrimitive);
    if (!reservePreparation(this, m_alignedPrimitives, bytes)
        || !checkPreparation(this,
            cudaMemcpyAsync(m_alignedPrimitives.ptr(),
                source,
                bytes,
                cudaMemcpyDeviceToDevice,
                deviceState()->preparation.stream()),
            "aligning SDF input"))
      return;
    m_primitives = m_alignedPrimitives.ptrAs<SDFPrimitive>();
  } else
    m_primitives = static_cast<const SDFPrimitive *>(source);
  input.sdf = m_primitives;
  input.blendFactor = m_blendFactor;
  input.blendLerpFactor = m_blendLerpFactor;
  input.noiseFactor = m_noiseFactor;
  if (!m_preprocessing.bounds(this, input, m_aabbs, m_epsilonScale))
    return;
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.ptr();

  upload();
}

bool SDF::isValid() const
{
  return bool(m_sdf);
}

void SDF::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives = m_numPrimitives;

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;
}

int SDF::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

GeometryGPUData SDF::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::SDF;

  auto &sdf = retval.sdf;
  sdf.geometries = m_primitives;
  sdf.neighbours =
      m_neighbour ? m_neighbour->beginAs<uint64_t>(AddressSpace::GPU) : nullptr;
  sdf.numGeometries = static_cast<uint32_t>(m_numPrimitives);
  sdf.epsilon = m_epsilon;
  sdf.nbMarchIterations = m_nbMarchIterations;
  sdf.blendFactor = m_blendFactor;
  sdf.blendLerpFactor = m_blendLerpFactor;
  sdf.omega = m_omega;
  sdf.distanceFromCamera = m_distanceFromCamera;
  sdf.noiseFactor = m_noiseFactor;

  return retval;
}

} // namespace visrtx
