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

#include "Geometry.h"
#include "array/Array1D.h"
#include "spatial_field/SpatialField.h"
#include "utility/DeviceBuffer.h"

namespace visrtx {

struct Isosurface : public Geometry
{
  Isosurface(DeviceGlobalState *d);
  ~Isosurface() override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  void populateBuildInput(OptixBuildInput &) const override;
  int optixGeometryType() const override;

 private:
  GeometryGPUData gpuData() const override;

  // Drops any per-primitive array shorter than m_numIsovalues (warns), so the
  // GPU never indexes a primitive attribute out of bounds.
  void dropUndersizedPrimitiveArrays();

  helium::ChangeObserverPtr<SpatialField> m_field;
  helium::ChangeObserverPtr<Array1D> m_isovalueArray;
  float m_isovalueScalar{0.f};
  bool m_hasScalarIsovalue{false};

  DeviceBuffer m_scalarIsovalueBuffer; // holds the single scalar on device
  DeviceBuffer m_aabbs; // coarse active-region brick boxes (object space)
  CUdeviceptr m_aabbsBufferPtr{};
  size_t m_numBricks{0};

  const float *m_isovaluesDev{nullptr};
  uint32_t m_numIsovalues{0};
};

} // namespace visrtx
