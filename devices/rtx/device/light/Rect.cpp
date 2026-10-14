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

#include "Rect.h"
#include <cuda_runtime.h>
#include "sampler/TextureStats.h"
#include "utility/CudaImageTexture.h"

#include <algorithm>
#include <cmath>

namespace visrtx {

Rect::Rect(DeviceGlobalState *d) : Light(d), m_colorMap(this) {}

Rect::~Rect()
{
  cleanupColorMap();
}

void Rect::cleanupColorMap()
{
  if (m_colorTexture) {
    cudaDestroyTextureObject(m_colorTexture);
    m_colorTexture = {};
  }
  m_colorMapMean = vec3(1.f);
}

void Rect::finalize()
{
  if (m_colorMap && m_colorMap->elementType() != ANARI_FLOAT32_VEC3) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "quad colorMap requires a FLOAT32_VEC3 image; using untextured emission");
    cleanupColorMap();
    m_colorMap = nullptr;
  }
  if (m_colorMap && m_colorMapStamp != m_colorMap->lastDataModified()) {
    cleanupColorMap();
    m_colorMapStamp = m_colorMap->lastDataModified();
    const size_t count = size_t(m_colorMap->size(0)) * m_colorMap->size(1);
    DeviceBuffer staging;
    const void *data = m_colorMap->stagedDataGPU(staging);
    const auto stats = data && count
        ? reduceTexelsDevice(data, TexelFormat::Float32, 3, 3, count)
        : TexelAccum{};
    if (data && count && stats.finite && stats.minValue[0] >= 0.f
        && stats.minValue[1] >= 0.f && stats.minValue[2] >= 0.f) {
      if (auto array = m_colorMap->cudaArray()) {
        m_colorTexture = makeCudaTextureObject2D(
            array, false, "linear", "clampToEdge", "clampToEdge");
        m_colorMapMean = vec3(float(stats.posSum[0] / count),
            float(stats.posSum[1] / count),
            float(stats.posSum[2] / count));
      }
    } else {
      reportMessage(ANARI_SEVERITY_WARNING,
          "quad colorMap requires finite nonnegative RGB; using untextured emission");
    }
  }
  upload();
}

void Rect::commitParameters()
{
  Light::commitParameters();
  auto *image = getParamObject<Array2D>("colorMap");
  if (image != m_colorMap.get()) {
    cleanupColorMap();
    m_colorMap = image;
    m_colorMapStamp = {};
  }
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_edge1 = getParam<vec3>("edge1", vec3(1.f, 0.f, 0.f));
  m_edge2 = getParam<vec3>("edge2", vec3(0.f, 1.f, 0.f));
  m_intensity = std::max(
      getParam<float>("intensity", getParam<float>("power", 1.f)), 0.f);

  m_coneAngle = getParam<float>("coneAngle", 1.57079632679489661923f);
  m_coneSoftness = getParam<float>("coneSoftness", 0.f);
  if (!std::isfinite(m_coneAngle) || !std::isfinite(m_coneSoftness)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "Nonfinite quad cone angle/softness; using unshaped defaults");
    m_coneAngle = 1.57079632679489661923f;
    m_coneSoftness = 0.f;
  }
  m_coneAngle = std::clamp(m_coneAngle, 0.f, 3.14159265358979323846f);
  m_coneSoftness = std::clamp(m_coneSoftness, 0.f, 1.f);

  m_focus = getParam<float>("focus", 0.f);
  m_focusTint = getParam<vec3>("focusTint", vec3(0.f));
  if (!std::isfinite(m_focus) || !std::isfinite(m_focusTint.x)
      || !std::isfinite(m_focusTint.y) || !std::isfinite(m_focusTint.z)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "Nonfinite quad focus/tint; using unfocused defaults");
    m_focus = 0.f;
    m_focusTint = vec3(0.f);
  }
  m_focus = std::max(m_focus, 0.f);

  auto side = getParamString("side", "front");
  if (side == "front")
    m_side = Side::FRONT;
  else if (side == "back")
    m_side = Side::BACK;
  else if (side == "both")
    m_side = Side::BOTH;
  else {
    reportMessage(
        ANARI_SEVERITY_WARNING, "Invalid 'side' parameter on rect light");
    m_side = Side::FRONT;
  }
}

LightGPUData Rect::gpuData() const
{
  auto retval = Light::gpuData();
  retval.type = LightType::RECT;
  retval.rect.position = m_position;
  retval.rect.edge1 = m_edge1;
  retval.rect.edge2 = m_edge2;
  retval.rect.intensity = m_intensity;
  retval.rect.colorMap = m_colorTexture;
  retval.rect.colorMapMean = m_colorMapMean;
  retval.rect.focus = m_focus;
  retval.rect.focusTint = m_focusTint;
  retval.rect.coneAngle = m_coneAngle;
  retval.rect.innerAngle = m_coneAngle * (1.f - m_coneSoftness);
  retval.rect.cosOuterAngle = cosf(retval.rect.coneAngle);
  retval.rect.cosInnerAngle = cosf(retval.rect.innerAngle);
  retval.rect.side.back = (m_side == Side::BACK || m_side == Side::BOTH) ? 1 : 0;
  retval.rect.side.front = (m_side == Side::FRONT || m_side == Side::BOTH) ? 1 : 0;
  auto area = length(cross(m_edge1, m_edge2));
  retval.rect.oneOverArea = area > 0.f ? 1.f / area : 1.f;

  return retval;
}

bool Rect::hasAreaProxy() const
{
  // A degenerate rect has nothing to show and no well-defined plane; skipping
  // it here keeps a zero-area light out of the BLAS entirely.
  return length(cross(m_edge1, m_edge2)) > 0.f;
}

box3 Rect::areaProxyBounds(const mat4 &xfm) const
{
  box3 bounds;
  const vec3 corners[4] = {m_position,
      m_position + m_edge1,
      m_position + m_edge2,
      m_position + m_edge1 + m_edge2};
  for (const vec3 &c : corners)
    bounds.extend(xfmPoint(xfm, c));
  return bounds;
}

} // namespace visrtx
