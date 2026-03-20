// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AutoExposurePass.h"
// std
#include <algorithm>
#include <cmath>

#include "detail/parallel_reduce.h"

namespace tsd::rendering {

namespace {

constexpr uint32_t SAMPLE_COUNT = 16384;
constexpr float MIN_LUMINANCE = 1e-4f;
constexpr float MIN_EXPOSURE = -20.f;
constexpr float MAX_EXPOSURE = 20.f;
constexpr float MID_GRAY = 0.18f;

float computeSumLogLuminance(
    const ImageBuffers &b, uint32_t numSamples, uint32_t stride)
{
  // very approximate and suboptimal, going only through a handleful of samples.
  // Good enough for now.
  // Might have to be revisited to be exhautive and still high performance.
  // https://gpuopen.com/fidelityfx-spd/ ?
  const float *hdrColor = b.hdrColor;
  return detail::parallel_reduce(
      b.stream,
      0u,
      numSamples,
      0.f,
      [=] DEVICE_FCN(uint32_t j) {
        const uint32_t idx = j * stride * 4;
        const float lum = std::max(0.2126f * hdrColor[idx + 0]
                + 0.7152f * hdrColor[idx + 1] + 0.0722f * hdrColor[idx + 2],
            MIN_LUMINANCE);
        return std::log2(lum);
      },
      [] DEVICE_FCN(float a, float b) { return a + b; });
}

} // namespace

AutoExposurePass::AutoExposurePass() = default;

AutoExposurePass::~AutoExposurePass() = default;

void AutoExposurePass::setHDREnabled(bool enabled)
{
  if (enabled && !m_hdrEnabled)
    m_hasExposure = false;
  m_hdrEnabled = enabled;
}

float AutoExposurePass::currentExposure() const
{
  return m_currentExposure;
}

void AutoExposurePass::render(ImageBuffers &b, int stageId)
{
  if (stageId == 0 || !m_hdrEnabled)
    return;

  const auto size = getDimensions();
  const uint32_t totalPixels = size.x * size.y;
  if (totalPixels == 0 || !b.hdrColor)
    return;

  const uint32_t stride = std::max(1u, totalPixels / SAMPLE_COUNT);
  const uint32_t numSamples = (totalPixels + stride - 1) / stride;
  if (numSamples == 0)
    return;

  const float sumLogLum = computeSumLogLuminance(b, numSamples, stride);
  const float avgLum = std::exp2(sumLogLum / float(numSamples));
  const float targetExposure =
      std::clamp(std::log2(MID_GRAY / avgLum), MIN_EXPOSURE, MAX_EXPOSURE);

  if (!m_hasExposure) {
    m_currentExposure = targetExposure;
    m_hasExposure = true;
  } else {
    m_currentExposure += (targetExposure - m_currentExposure) * m_response;
  }

  b.exposure = m_currentExposure;
}

} // namespace tsd::rendering
