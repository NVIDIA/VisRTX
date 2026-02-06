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

#include "array/Array1D.h"
#include "volume/Volume.h"
#include "spatial_field/SpatialField.h"

namespace visrtx {

struct TransferFunction1D : public Volume
{
  TransferFunction1D(DeviceGlobalState *d);
  ~TransferFunction1D();

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

 private:
  VolumeGPUData gpuData() const override;
  void discritizeTFData();
  void createTFTexture();
  void cleanup();

  helium::ChangeObserverPtr<Array1D> m_color;
  helium::ChangeObserverPtr<Array1D> m_opacity;

  box1 m_valueRange{0.f, 1.f};
  float m_unitDistance{1.f};
  vec4 m_uniformColor{1.f};
  float m_uniformOpacity{1.f};

  helium::ChangeObserverPtr<SpatialField> m_field;

  std::vector<vec4> m_tf;
  int m_tfDim{256};

  cudaArray_t m_cudaArray{};
  cudaTextureObject_t m_textureObject{};
};

} // namespace visrtx
