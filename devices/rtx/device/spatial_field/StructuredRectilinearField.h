/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "array/Array1D.h"
#include "array/Array3D.h"
#include "spatial_field/SpatialField.h"

namespace visrtx {

struct StructuredRectilinearField : public SpatialField
{
  StructuredRectilinearField(DeviceGlobalState *d);
  ~StructuredRectilinearField();

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  box3 bounds() const override;
  float stepSize() const override;

 private:
  SpatialFieldGPUData gpuData() const override;
  void cleanup();

  void buildGrid();

  std::string m_filter;
  bool m_cellCentered{false};
  box3 m_roi{box3(vec3(std::numeric_limits<float>::lowest()),
      vec3(std::numeric_limits<float>::max()))};
  helium::ChangeObserverPtr<Array3D> m_data;

  // Required rectilinear axis arrays (all 3 required)
  box3 m_bounds;
  helium::ChangeObserverPtr<Array1D> m_axisArrayX;
  helium::ChangeObserverPtr<Array1D> m_axisArrayY;
  helium::ChangeObserverPtr<Array1D> m_axisArrayZ;
  cudaArray_t m_axisLutArrays[3]{};
  cudaTextureObject_t m_axisLutTextures[3]{};

  cudaArray_t m_cudaArray{};
  cudaTextureObject_t m_textureObject{};
};

} // namespace visrtx
