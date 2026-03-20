// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "OutputTransformPass.h"
// std
#include <cmath>

#include "detail/parallel_for.h"

namespace tsd::rendering {

namespace {

DEVICE_FCN_INLINE tsd::math::float3 linearToGamma(
    tsd::math::float3 c, float invGamma)
{
  c.x = std::pow(std::clamp(c.x, 0.f, 1.f), invGamma);
  c.y = std::pow(std::clamp(c.y, 0.f, 1.f), invGamma);
  c.z = std::pow(std::clamp(c.z, 0.f, 1.f), invGamma);
  return c;
}

} // namespace

OutputTransformPass::OutputTransformPass() = default;

OutputTransformPass::~OutputTransformPass() = default;

void OutputTransformPass::setColorFormat(anari::DataType format)
{
  m_colorFormat = format;
}

void OutputTransformPass::setGamma(float gamma)
{
  m_gamma = gamma;
}

void applyOutputTransform(ImageBuffers &b,
    uint32_t totalPixels,
    float invGamma,
    anari::DataType colorFormat)
{
  detail::parallel_for(b.stream, 0u, totalPixels, [=] DEVICE_FCN(uint32_t i) {
    tsd::math::float4 c(0.f);

    if (colorFormat == ANARI_FLOAT32_VEC4) {
      const uint32_t idx = i * 4;
      c.x = b.hdrColor[idx + 0];
      c.y = b.hdrColor[idx + 1];
      c.z = b.hdrColor[idx + 2];
      c.w = b.hdrColor[idx + 3];
    } else {
      c = helium::cvt_color_to_float4(b.color[i]);
    }

    const auto encoded =
        linearToGamma(tsd::math::float3(c.x, c.y, c.z), invGamma);
    b.color[i] =
        helium::cvt_color_to_uint32({encoded.x, encoded.y, encoded.z, c.w});
  });
}

void OutputTransformPass::render(ImageBuffers &b, int stageId)
{
  if (stageId == 0 || m_colorFormat == ANARI_UFIXED8_RGBA_SRGB)
    return;

  const auto size = getDimensions();
  const uint32_t totalPixels = size.x * size.y;
  if (totalPixels == 0 || !b.color)
    return;

  applyOutputTransform(b, totalPixels, 1.f / m_gamma, m_colorFormat);
}

} // namespace tsd::rendering
