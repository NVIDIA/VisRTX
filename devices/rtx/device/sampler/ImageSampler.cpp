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

#include "ImageSampler.h"
#include "utility/AnariTypeHelpers.h"

namespace visrtx {

ImageSampler::ImageSampler(DeviceGlobalState *d, const char *subtype)
    : Sampler(d), m_subtype(subtype), m_image(this)
{}

ImageSampler::~ImageSampler()
{
  // Runs before m_image is released, which may free the array's storage.
  cleanupTextureObjects();
}

bool ImageSampler::isValid() const
{
  return m_texture && m_texels;
}

int ImageSampler::numChannels() const
{
  return m_image ? numANARIChannels(m_image->elementType()) : 0;
}

void ImageSampler::setImage(NumericArray *image)
{
  // Dropping the old array may free its storage; destroy what refers to it.
  if (image != m_image.get())
    cleanupTextureObjects();
  m_image = image;
}

NumericArray *ImageSampler::imageArray() const
{
  return m_image.get();
}

void ImageSampler::finalize()
{
  cleanupTextureObjects();
  if (!m_image) {
    upload();
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on %s sampler",
        m_subtype);
    return;
  }

  const auto format = m_image->elementType();
  if (numANARIChannels(format) == 0) {
    upload();
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in %s sampler (%s)",
        m_subtype,
        anari::toString(format));
    return;
  }

  if (auto cuArray = m_image->cudaArray())
    makeTextureObjects(cuArray, format);
  // Failed uploads publish an invalid sampler rather than stale handles.
  upload();
}

void ImageSampler::cleanupTextureObjects()
{
  if (m_texels)
    cudaDestroyTextureObject(m_texels);
  if (m_texture)
    cudaDestroyTextureObject(m_texture);
  m_texels = {};
  m_texture = {};
}

} // namespace visrtx
