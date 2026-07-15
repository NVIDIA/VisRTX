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

#include <array>

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
#if defined(USE_MDL)
  libmdl::ResourceStats emissionStats() const override;
#endif

  cudaTextureObject_t textureObject() const;

 private:
  SamplerGPUData gpuData() const override;

  // Single-pass texel reduction feeding both the emissive Pick Power and the
  // MDL emission classifier from one scan (replacing the former separate
  // averageValue and emissionStats scans). Pick Power reads meanPositive — the
  // same non-negative magnitude proxy the classifier uses — so a signed texel
  // never inflates or cancels an emitter's picked power.
  struct TextureReduction
  {
    // Per channel; unit default so an un-reduced emitter is still picked.
    std::array<float, 3> maxAbs{{1.f, 1.f, 1.f}}; // maxAbs==0 ⇒ exact zero
    std::array<float, 3> meanPositive{{1.f, 1.f, 1.f}}; // magnitude / Pick Power
    std::array<float, 3> minValue{{-1.f, -1.f, -1.f}}; // minValue>=0 ⇒ nonneg
    bool transferPreservesZero{false}; // T(0)==0 unless a nonzero border color
    bool finite{true};
    bool valid{false}; // false ⇒ unbound/unsupported ⇒ classifier Unknown
  };

  // Lazy, memoized against the image data stamp: computed on the first query
  // and reused until the bound image's texels actually change. Non-emissive
  // samplers never query it and so never scan.
  const TextureReduction &textureReduction() const;
  TextureReduction computeTextureReduction() const;

  void cleanupImageCudaArray();
  void cleanupImageTextureObjects();

  std::string m_filter;
  std::string m_wrap1;
  std::string m_wrap2;
  helium::ChangeObserverPtr<Array2D> m_image;

  cudaTextureObject_t m_texture{};
  cudaTextureObject_t m_texels{};

  // Memoized reduction, filled lazily from the const query and guarded on the
  // image's lastDataModified stamp: a no-op recommit (filter/wrap change, scene
  // churn) does not rescan. mutable: filled from the const query.
  mutable TextureReduction m_reduction;
  mutable helium::TimeStamp m_reductionStamp{0};
};

} // namespace visrtx
