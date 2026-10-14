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
#include "array/NumericArray.h"

namespace visrtx {

/* Shared image binding, validation, texture lifetime, and registry
 * publication. The bound array owns the CUDA storage; this sampler keeps the
 * array alive and destroys its texture objects before releasing it.
 * Dimensional samplers only bind their typed image and create texture objects.
 */
struct ImageSampler : public Sampler
{
  ImageSampler(DeviceGlobalState *d, const char *subtype);
  ~ImageSampler() override;
  ImageSampler(const ImageSampler &) = delete;
  ImageSampler &operator=(const ImageSampler &) = delete;
  ImageSampler(ImageSampler &&) = delete;
  ImageSampler &operator=(ImageSampler &&) = delete;

  bool isValid() const override final;
  int numChannels() const override final;
  void finalize() override final;

 protected:
  void setImage(NumericArray *image);
  NumericArray *imageArray() const;
  virtual void makeTextureObjects(cudaArray_t array, ANARIDataType format) = 0;

  cudaTextureObject_t m_texture{};
  cudaTextureObject_t m_texels{};

 private:
  void cleanupTextureObjects();

  const char *m_subtype;
  helium::ChangeObserverPtr<NumericArray> m_image;
};

} // namespace visrtx
