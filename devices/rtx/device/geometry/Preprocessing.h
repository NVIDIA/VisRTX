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

#pragma once

#include "gpu/gpu_objects.h"
#include "utility/DeviceBuffer.h"

namespace visrtx {
struct Geometry;

// GPU views use logical array beginnings, never application allocation bases.
// Geometry::beginPreparation() completes authored host-array uploads before
// these views are consumed; kernels need no legacy default-stream dependency.
struct PrimitiveInput
{
  GeometryType type{GeometryType::TRIANGLE};
  size_t count{0};
  const vec3 *vertices{nullptr};
  const void *indices{nullptr};
  const float *radii{nullptr};
  const uint8_t *caps{nullptr};
  float radius{0.f};
  uint8_t capFlags{0};
  const SDFPrimitive *sdf{nullptr};
  float blendFactor{1.f};
  float blendLerpFactor{0.5f};
  float noiseFactor{0.f};
};

// Per-geometry reusable workspace. Only the compact result crosses to the host.
struct Preprocessing
{
  Preprocessing() = default;
  Preprocessing(const Preprocessing &) = delete;
  Preprocessing &operator=(const Preprocessing &) = delete;
  Preprocessing(Preprocessing &&) = delete;
  Preprocessing &operator=(Preprocessing &&) = delete;

  bool areas(Geometry *geometry,
      const PrimitiveInput &input,
      DeviceBuffer &cdf,
      float &total);
  bool bounds(Geometry *geometry,
      const PrimitiveInput &input,
      DeviceBuffer &aabbs,
      float &scale);

 private:
  DeviceBuffer m_prefix;
  DeviceBuffer m_scratch;
  DeviceBuffer m_result;
};

bool checkPreparation(
    Geometry *geometry, cudaError_t error, const char *operation);
bool reservePreparation(Geometry *geometry, DeviceBuffer &buffer, size_t bytes);
bool finishPreparation(Geometry *geometry);
bool generateQuadIndices(Geometry *geometry,
    const uvec4 *indices,
    size_t count,
    DeviceBuffer &output);
bool generateCurveIndices(
    Geometry *geometry, size_t count, DeviceBuffer &output);
bool generateCurveRadii(
    Geometry *geometry, size_t count, float radius, DeviceBuffer &output);

} // namespace visrtx
