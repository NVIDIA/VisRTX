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

#include "Sampler.h"
#include "array/Array2D.h"
#include "utility/CudaImageTexture.h"

namespace visrtx {

struct Image2D : public Sampler
{
  Image2D(DeviceGlobalState *d);
  ~Image2D();

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  int numChannels() const override;
  vec4 averageValue() const override;

  cudaTextureObject_t textureObject() const;

 private:
  SamplerGPUData gpuData() const override;

  // Mean linear texel for emissive Pick Power. computeAverageValue() is the host
  // scan; computeAverageValueGPU() is the intended device-side reduction over the
  // already-resident texels (stubbed — currently delegates to the host scan).
  vec4 computeAverageValue() const;
  vec4 computeAverageValueGPU() const;

  void cleanupImageCudaArray();
  void cleanupImageTextureObjects();

  std::string m_filter;
  std::string m_wrap1;
  std::string m_wrap2;
  helium::ChangeObserverPtr<Array2D> m_image;

  cudaTextureObject_t m_texture{};
  cudaTextureObject_t m_texels{};

  // Mean linear texel, consumed only by the emissive Pick-Power path. Computed
  // lazily on the first averageValue() query and memoized against the image's
  // lastDataModified stamp: non-emissive samplers (base color, normal, roughness,
  // ...) never query it and so never scan, and a no-op recommit (filter/wrap
  // change, scene churn) does not rescan. mutable: filled from the const query.
  mutable vec4 m_averageValue{1.f};
  mutable helium::TimeStamp m_averageValueStamp{0};
};

} // namespace visrtx
