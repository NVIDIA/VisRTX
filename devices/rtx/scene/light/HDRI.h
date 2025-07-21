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

#pragma once

#include "Light.h"
#include "array/Array.h"
#include "array/Array2D.h"
#include "utility/CudaImageTexture.h"

namespace visrtx {

struct HDRI : public Light
{
  HDRI(DeviceGlobalState *d);
  ~HDRI() override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;
  bool isHDRI() const override;

 private:
  LightGPUData gpuData() const override;
  void cleanup();

  // Data //

  vec3 m_up{0.f, 0.f, 1.f};
  vec3 m_direction{1.f, 0.f, 0.f};
  float m_scale{1.f};
  bool m_visible{true};
  uvec2 m_size{0, 0};
  helium::ChangeObserverPtr<Array2D> m_radiance;
  cudaTextureObject_t m_radianceTex{};
  DeviceBuffer m_marginalCDF;
  DeviceBuffer m_conditionalCDF;
  float m_pdfWeight{0.f};
#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  uint32_t *m_samples{nullptr};
#endif

  // Importance sampling support
  float generateCDFTables(const glm::vec3 *envMap,
      int width,
      int height,
      DeviceBuffer *marginalCdf,
      DeviceBuffer *conditionalCdf);
};

} // namespace visrtx
