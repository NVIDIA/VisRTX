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
#include "utility/DeviceBuffer.h"

namespace visrtx {

// Signed Distance Field geometry.
//
// ANARI subtype: "sdf"
//
// Parameters:
//   primitive.sdf      Array1D<SDFPrimitive>  (required) - SDF primitive data
//   primitive.neighbor Array1D<uint64_t>      (optional) - flat neighbour index
//   buffer epsilon            float   (default 1e-5) - sphere-tracing
//   convergence threshold nbMarchIterations  uint32  (default 128)  - maximum
//   ray-march iterations blendFactor        float   (default 1.0)  - smooth-min
//   blend strength blendLerpFactor    float   (default 0.5)  - radius lerp for
//   blend factor omega              float   (default 1.0)  - sphere-tracing
//   step scale (<=1) distanceFromCamera float   (default 100)  - max camera
//   distance for displacement noiseFactor             float   (default 0.0)   -
//   organic noise [0=off, 1=max]
struct SDF : public Geometry
{
  SDF(DeviceGlobalState *d);
  ~SDF() override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  void populateBuildInput(OptixBuildInput &) const override;
  int optixGeometryType() const override;

 private:
  GeometryGPUData gpuData() const override;

  helium::ChangeObserverPtr<Array1D> m_sdf;
  helium::ChangeObserverPtr<Array1D> m_neighbour;

  float m_epsilon{1e-5f};
  uint32_t m_nbMarchIterations{16};
  float m_blendFactor{1.f};
  float m_blendLerpFactor{0.5f};
  float m_omega{1.f};
  float m_distanceFromCamera{100.f};
  float m_noiseFactor{0.f};

  DeviceBuffer m_alignedPrimitives;
  const SDFPrimitive *m_primitives{nullptr};
  DeviceBuffer m_aabbs;
  CUdeviceptr m_aabbsBufferPtr{};
  size_t m_numPrimitives{0};
};

} // namespace visrtx
