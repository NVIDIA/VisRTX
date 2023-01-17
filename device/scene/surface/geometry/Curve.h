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

#pragma once

#include "array/Array.h"
#include "Geometry.h"
#include "utility/HostDeviceArray.h"

namespace visrtx {

struct Curve : public Geometry
{
  Curve(DeviceGlobalState *d);
  ~Curve() override;

  void commit() override;

  void populateBuildInput(OptixBuildInput &) const override;

  int optixGeometryType() const override;

  bool isValid() const override;

 private:
  void computeIndices();
  void computeRadii();
  GeometryGPUData gpuData() const override;
  void cleanup();

  helium::IntrusivePtr<Array1D> m_index;

  helium::IntrusivePtr<Array1D> m_vertexPosition;
  helium::IntrusivePtr<Array1D> m_vertexRadius;
  helium::IntrusivePtr<Array1D> m_vertexAttribute0;
  helium::IntrusivePtr<Array1D> m_vertexAttribute1;
  helium::IntrusivePtr<Array1D> m_vertexAttribute2;
  helium::IntrusivePtr<Array1D> m_vertexAttribute3;
  helium::IntrusivePtr<Array1D> m_vertexColor;

  float m_globalRadius;

  HostDeviceArray<uint32_t> m_generatedIndices;
  HostDeviceArray<float> m_generatedRadii;

  CUdeviceptr m_vertexBufferPtr{};
  CUdeviceptr m_radiusBufferPtr{};
};

} // namespace visrtx
