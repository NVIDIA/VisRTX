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

#include "TransferFunction1D.h"
#include "utility/colorMapHelpers.h"

namespace visrtx {

TransferFunction1D::TransferFunction1D(DeviceGlobalState *d)
    : Volume(d), m_color(this), m_opacity(this), m_field(this)
{}

TransferFunction1D::~TransferFunction1D()
{
  cleanup();
}

void TransferFunction1D::commitParameters()
{
  Volume::commitParameters();
  m_color = getParamObject<Array1D>("color");
  m_uniformColor = vec4(1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_uniformColor);
  getParam("color", ANARI_FLOAT32_VEC4, &m_uniformColor);
  m_opacity = getParamObject<Array1D>("opacity");
  m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
  m_unitDistance = getParam<float>("unitDistance", 1.f);
  m_field = getParamObject<SpatialField>("value");
  getParam("valueRange", ANARI_FLOAT32_VEC2, &m_valueRange);
  getParam("valueRange", ANARI_FLOAT32_BOX1, &m_valueRange);
  double valueRange_d[2] = {0.0, 1.0};
  if (getParam("valueRange", ANARI_FLOAT64_BOX1, valueRange_d)) {
    m_valueRange.lower = float(valueRange_d[0]);
    m_valueRange.upper = float(valueRange_d[1]);
  }
}

void TransferFunction1D::finalize()
{
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing parameter 'value' on transferFunction1D ANARIVolume");
    return;
  }

  if (m_unitDistance <= 0.f) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unitDistance must be positive on transferFunction1D volume, clamping to minimum");
    m_unitDistance = 1e-6f;
  }

  discritizeTFData();
  createTFTexture();
  m_field->m_uniformGrid.computeMaxOpacities(
      deviceState()->stream, m_textureObject, m_tfDim);
  upload();
}

bool TransferFunction1D::isValid() const
{
  return m_field && m_field->isValid();
}

VolumeGPUData TransferFunction1D::gpuData() const
{
  VolumeGPUData retval = Volume::gpuData();
  retval.type = VolumeType::TF1D;
  retval.bounds = m_field->bounds();
  retval.stepSize = m_field->stepSize();
  retval.data.tf1d.tfTex = m_textureObject;
  retval.data.tf1d.valueRange = m_valueRange;
  retval.data.tf1d.oneOverUnitDistance = 1.0f / glm::max(m_unitDistance, 1e-6f);
  retval.data.tf1d.field = m_field->index();
  retval.data.tf1d.uniformColor = vec3(m_uniformColor);
  retval.data.tf1d.uniformOpacity = m_uniformOpacity;
  return retval;
}

void TransferFunction1D::discritizeTFData()
{
  m_tf.resize(m_tfDim);

  Span<float> cPositions;
  Span<float> oPositions;

  std::vector<float> linearColorPositions;
  std::vector<float> linearOpacityPositions;

  if (m_color) {
    linearColorPositions =
        generateLinearPositions(m_color->totalSize(), m_valueRange);
    cPositions =
        make_Span(linearColorPositions.data(), linearColorPositions.size());
  }

  if (m_opacity) {
    linearOpacityPositions =
        generateLinearPositions(m_opacity->totalSize(), m_valueRange);
    oPositions =
        make_Span(linearOpacityPositions.data(), linearOpacityPositions.size());
  }

  bool warnOnce = false;
  for (size_t i = 0; i < m_tf.size(); i++) {
    const float p = float(i) / (m_tf.size() - 1);
    vec4 c = m_uniformColor;
    if (m_color) {
      if (m_color->elementType() == ANARI_FLOAT32_VEC3) {
        c = vec4(getInterpolatedValue(
                     m_color->beginAs<vec3>(), cPositions, m_valueRange, p),
            1.f);
      } else if (m_color->elementType() == ANARI_FLOAT32_VEC4) {
        c = getInterpolatedValue(
            m_color->beginAs<vec4>(), cPositions, m_valueRange, p);
      } else if (!warnOnce) {
        reportMessage(ANARI_SEVERITY_WARNING,
            "unusable tf1D color array type set (%s)",
            anari::toString(m_color->elementType()));
        warnOnce = true;
      }
    }
    const float o = m_opacity
        ? getInterpolatedValue(
              m_opacity->beginAs<float>(), oPositions, m_valueRange, p)
        : m_uniformOpacity;
    m_tf[i] = vec4(c.x, c.y, c.z, c.w * o);
  }
}

void TransferFunction1D::createTFTexture()
{
  cleanup();

  if (m_tf.empty())
    return;

  auto desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&m_cudaArray, &desc, m_tfDim);

  cudaMemcpy3DParms copyParams;
  std::memset(&copyParams, 0, sizeof(copyParams));
  copyParams.srcPtr = make_cudaPitchedPtr(
      m_tf.data(), m_tf.size() * sizeof(vec4), m_tf.size(), 0);
  copyParams.dstArray = m_cudaArray;
  copyParams.extent = make_cudaExtent(m_tfDim, 1, 1);
  copyParams.kind = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&copyParams);

  cudaResourceDesc resDesc;
  std::memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_cudaArray;

  cudaTextureDesc texDesc;
  std::memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(&m_textureObject, &resDesc, &texDesc, nullptr);
}

void TransferFunction1D::cleanup()
{
  if (m_textureObject)
    cudaDestroyTextureObject(m_textureObject);
  if (m_cudaArray)
    cudaFreeArray(m_cudaArray);
  m_textureObject = {};
  m_cudaArray = {};
}

} // namespace visrtx
