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

#include "Image2D.h"

#include "utility/AnariTypeHelpers.h"

namespace visrtx {

static float srgbToLinear(float v)
{
  return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}

Image2D::Image2D(DeviceGlobalState *d) : Sampler(d), m_image(this) {}

Image2D::~Image2D()
{
  cleanupImageTextureObjects();
  cleanupImageCudaArray();
}

void Image2D::commitParameters()
{
  Sampler::commitParameters();
  m_filter = getParamString("filter", "linear");
  m_wrap1 = getParamString("wrapMode1", "clampToEdge");
  m_wrap2 = getParamString("wrapMode2", "clampToEdge");
  auto *oldImage = m_image.get();
  auto *newImage = getParamObject<Array2D>("image");
  if (oldImage != newImage)
    cleanupImageCudaArray();
  m_image = newImage;
}

void Image2D::finalize()
{
  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image2D sampler");
    return;
  }

  const ANARIDataType format = m_image->elementType();
  auto nc = numANARIChannels(format);
  if (nc == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid texture type encountered in image2D sampler (%s)",
        anari::toString(format));
    return;
  }

  const bool isFp = isFloat(m_image->elementType());
  cudaArray_t cuArray = m_image->acquireCUDAArray();

  cleanupImageTextureObjects();

  // sRGB data is kept as raw bytes; the sampler does sRGB->linear in hardware.
  const bool sRGB = isSrgb8(m_image->elementType());
  m_texture = makeCudaTextureObject2D(
      cuArray, !isFp, m_filter, m_wrap1, m_wrap2, m_borderColor, sRGB);
  m_texels = makeCudaTexelObject2D(
      cuArray, !isFp, "nearest", m_wrap1, m_wrap2, m_borderColor);

  // The mean texel is NOT computed here: it is only needed by the emissive
  // Pick-Power path, so averageValue() scans lazily on first query and memoizes
  // against the image stamp (see m_averageValue).

  upload();
}

bool Image2D::isValid() const
{
  return m_image;
}

vec4 Image2D::averageValue() const
{
  // Lazy + guarded: recompute only when the bound image's data actually changed.
  // A fresh (0) stamp forces the first compute; a filter/wrap recommit leaves the
  // image stamp untouched and returns the cache. Non-emissive samplers never
  // reach here at all.
  const helium::TimeStamp stamp =
      m_image ? m_image->lastDataModified() : helium::TimeStamp{0};
  if (stamp != m_averageValueStamp) {
    m_averageValue = computeAverageValueGPU();
    m_averageValueStamp = stamp;
  }
  return m_averageValue;
}

// Mean linear texel, used only to size a textured emitter's Pick Power
// (variance, never bias). Reads the retained host pixels; sRGB byte data is
// linearized to match the hardware sampler. Unsupported element types fall back
// to the fully-lit default so the emitter is still picked. Computed lazily and
// memoized; see averageValue() / m_averageValue.
vec4 Image2D::computeAverageValue() const
{
  if (!m_image)
    return Sampler::averageValue();

  const ANARIDataType t = m_image->elementType();
  const int nc = numANARIChannels(t);
  const size_t count = size_t(m_image->size().x) * m_image->size().y;
  const void *host = m_image->data(AddressSpace::HOST);
  if (nc == 0 || count == 0 || !host)
    return Sampler::averageValue();

  // sRGB 8-bit formats carry a linear alpha in the LAST channel (present for the
  // RGBA/RA variants, i.e. even channel counts); only the color channels are
  // gamma-encoded.
  const bool srgb = isSrgb8(t);
  const int colorChannels = (srgb && (nc == 2 || nc == 4)) ? nc - 1 : nc;

  glm::dvec4 sum(0.0);
  if (isFloat32(t)) {
    const auto *p = static_cast<const float *>(host);
    for (size_t i = 0; i < count; ++i)
      for (int c = 0; c < nc; ++c)
        sum[c] += double(p[i * nc + c]);
  } else if (isFixed8(t) || srgb) {
    const auto *p = static_cast<const uint8_t *>(host);
    for (size_t i = 0; i < count; ++i)
      for (int c = 0; c < nc; ++c) {
        const float v = p[i * nc + c] / 255.0f;
        sum[c] += double((srgb && c < colorChannels) ? srgbToLinear(v) : v);
      }
  } else {
    return Sampler::averageValue(); // uncommon type for emission; coarse fallback
  }

  glm::dvec4 avg = sum / double(count);
  // Broadcast 1/2-channel textures to RGB so a grayscale emissive texture still
  // yields a sensible average color.
  if (nc == 1)
    return vec4(float(avg.x), float(avg.x), float(avg.x), 1.f);
  if (nc == 2)
    return vec4(float(avg.x), float(avg.x), float(avg.x), float(avg.y));
  if (nc == 3)
    return vec4(float(avg.x), float(avg.y), float(avg.z), 1.f);
  return vec4(avg);
}

// TODO(perf): reduce over the resident texels on the device instead of the host
// scan above — the image is already uploaded as a cudaArray for sampling
// (m_texels), so a device reduction (cf. the thrust::reduce-over-image in
// light/sampling/CDF.cu) avoids the host readback entirely for large emissive
// textures. The kernel must reproduce computeAverageValue()'s per-channel
// sRGB->linear (color channels only) and the 1/2-channel broadcast, then read
// back a single vec4. Stubbed: delegates to the host scan for now.
vec4 Image2D::computeAverageValueGPU() const
{
  return computeAverageValue();
}

int Image2D::numChannels() const
{
  ANARIDataType format = m_image->elementType();
  return numANARIChannels(format);
}

cudaTextureObject_t Image2D::textureObject() const
{
  return m_texture;
}

SamplerGPUData Image2D::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::TEXTURE2D;
  retval.image2D.texobj = m_texture;
  retval.image2D.texelTexobj = m_texels;
  retval.image2D.size = glm::uvec2(m_image->size().x, m_image->size().y);
  retval.image2D.invSize =
      glm::vec2(1.0f / m_image->size().x, 1.0f / m_image->size().y);

  return retval;
}

void Image2D::cleanupImageCudaArray()
{
  if (!m_image)
    return;

  m_image->releaseCUDAArray();
}

void Image2D::cleanupImageTextureObjects()
{
  cudaDestroyTextureObject(m_texels);
  cudaDestroyTextureObject(m_texture);
  m_texels = {};
  m_texture = {};
}

} // namespace visrtx
