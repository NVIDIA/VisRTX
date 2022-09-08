/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ColorMap.h"
#include "utility/colorMapHelpers.h"
// std
#include <array>

namespace visrtx {

ColorMap::~ColorMap()
{
  cleanup();
}

void ColorMap::commit()
{
  Sampler::commit();

  cleanup();

  m_params.color = getParamObject<Array1D>("color");
  m_params.colorPosition = getParamObject<Array1D>("color.position");

  {
    auto valueRangeAsVec2 = getParam<vec2>("valueRange", vec2(0.f, 1.f));
    m_params.valueRange =
        getParam<box1>("valueRange", make_box1(valueRangeAsVec2));
  }

  if (!m_params.color) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'color' on colormap sampler");
    return;
  }

  if (m_params.colorPosition
      && m_params.color->totalSize() != m_params.colorPosition->totalSize()) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "ColorMapSampler 'color' and 'color.position'"
        " arrays are of different size");
    return;
  }

  m_params.color->addCommitObserver(this);
  if (m_params.colorPosition)
    m_params.colorPosition->addCommitObserver(this);

  // Create CUDA texture //

  discritizeTFData();

  auto desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&m_cudaArray, &desc, m_tfDim);

  cudaMemcpy3DParms copyParams;
  std::memset(&copyParams, 0, sizeof(copyParams));
  copyParams.srcPtr =
      make_cudaPitchedPtr(m_tf.data(), m_tfDim * sizeof(vec4), m_tfDim, 0);
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

SamplerGPUData ColorMap::gpuData() const
{
  SamplerGPUData retval = Sampler::gpuData();
  retval.type = SamplerType::COLOR_MAP;
  retval.colormap.tfTex = m_textureObject;
  retval.colormap.valueRange = m_params.valueRange;
  return retval;
}

int ColorMap::numChannels() const
{
  return 3;
}

bool ColorMap::isValid() const
{
  return m_params.color;
}

void ColorMap::discritizeTFData()
{
  m_tf.resize(m_tfDim);

  anari::Span<float> cPositions;

  std::vector<float> linearColorPositions;

  if (m_params.colorPosition) {
    cPositions = anari::make_Span(m_params.colorPosition->beginAs<float>(),
        m_params.colorPosition->size());
  } else {
    linearColorPositions = generateLinearPositions(
        m_params.color->totalSize(), m_params.valueRange);
    cPositions = anari::make_Span(
        linearColorPositions.data(), linearColorPositions.size());
  }

  for (size_t i = 0; i < m_tf.size(); i++) {
    const float p = float(i) / (m_tf.size() - 1);
    auto color = getInterpolatedValue(
        m_params.color->beginAs<vec3>(), cPositions, m_params.valueRange, p);
    m_tf[i] = vec4(color, 1.f);
  }
}

void ColorMap::cleanup()
{
  if (m_textureObject)
    cudaDestroyTextureObject(m_textureObject);
  if (m_cudaArray)
    cudaFreeArray(m_cudaArray);
  m_textureObject = {};
  m_cudaArray = {};
  if (m_params.color)
    m_params.color->removeCommitObserver(this);
  if (m_params.colorPosition)
    m_params.colorPosition->removeCommitObserver(this);
}

} // namespace visrtx
