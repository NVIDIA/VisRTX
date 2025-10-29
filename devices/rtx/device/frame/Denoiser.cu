/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Denoiser.h"
#include "gpu/gpu_util.h"
#include "utility/instrument.h"
// thrust
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace visrtx {

Denoiser::Denoiser(DeviceGlobalState *s) : Object(ANARI_OBJECT, s) {}

Denoiser::~Denoiser()
{
  cleanup();

  if (m_denoiser)
    OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
}

void Denoiser::setup(
    uvec2 size, HostDeviceArray<uint8_t> &pixelBuffer, ANARIDataType format)
{
  init();
  auto &state = *deviceState();

  m_pixelBuffer = &pixelBuffer;

  m_format = format;

  OptixDenoiserSizes sizes;
  OPTIX_CHECK(
      optixDenoiserComputeMemoryResources(m_denoiser, size.x, size.y, &sizes));

  m_state.reserve(sizes.stateSizeInBytes);
  m_scratch.reserve(sizes.withoutOverlapScratchSizeInBytes);

  if (format != ANARI_FLOAT32_VEC4)
    m_uintPixels.resize(size_t(size.x) * size_t(size.y));
  else
    m_uintPixels.clear();

  OPTIX_CHECK(optixDenoiserSetup(m_denoiser,
      state.stream,
      size.x,
      size.y,
      (CUdeviceptr)m_state.ptr(),
      m_state.bytes(),
      (CUdeviceptr)m_scratch.ptr(),
      m_scratch.bytes()));

  m_layer.input.data = (CUdeviceptr)pixelBuffer.dataDevice();
  m_layer.input.width = size.x;
  m_layer.input.height = size.y;
  m_layer.input.pixelStrideInBytes = 0;
  m_layer.input.rowStrideInBytes = 4 * sizeof(float) * size.x;
  m_layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
  std::memcpy(&m_layer.output, &m_layer.input, sizeof(m_layer.output));
}

void Denoiser::cleanup()
{
  m_state.reset();
  m_scratch.reset();
}

void Denoiser::launch()
{
  auto &state = *deviceState();

  instrument::rangePush("optixDenoiserInvoke()");
  OPTIX_CHECK(optixDenoiserInvoke(m_denoiser,
      state.stream,
      &m_params,
      (CUdeviceptr)m_state.ptr(),
      static_cast<unsigned int>(m_state.bytes()),
      &m_guideLayer,
      &m_layer,
      1,
      0, // input offset X
      0, // input offset y
      (CUdeviceptr)m_scratch.ptr(),
      static_cast<unsigned int>(m_scratch.bytes())));
  instrument::rangePop(); // optixDenoiserInvoke()

  if (m_format != ANARI_FLOAT32_VEC4) {
    instrument::rangePush("denoiser transform pixels");
    auto numPixels =
        size_t(m_layer.output.width) * size_t(m_layer.output.height);
    auto begin = thrust::device_ptr<vec4>((vec4 *)m_pixelBuffer->dataDevice());
    auto end = begin + numPixels;
    if (m_format == ANARI_UFIXED8_RGBA_SRGB) {
      thrust::transform(thrust::cuda::par.on(state.stream),
          begin,
          end,
          thrust::device_pointer_cast<uint32_t>(m_uintPixels.dataDevice()),
          [] __device__(const vec4 &in) {
            return glm::packUnorm4x8(glm::convertLinearToSRGB(in));
          });
    } else {
      thrust::transform(thrust::cuda::par.on(state.stream),
          begin,
          end,
          thrust::device_pointer_cast<uint32_t>(m_uintPixels.dataDevice()),
          [] __device__(const vec4 &in) { return glm::packUnorm4x8(in); });
    }
    instrument::rangePop(); // denoiser transform pixels
  }
}

void *Denoiser::mapColorBuffer()
{
  if (m_format == ANARI_FLOAT32_VEC4) {
    m_pixelBuffer->download();
    return m_pixelBuffer->dataHost();
  } else {
    m_uintPixels.download();
    return m_uintPixels.dataHost();
  }
}

void *Denoiser::mapGPUColorBuffer()
{
  return m_format == ANARI_FLOAT32_VEC4 ? (void *)m_pixelBuffer->dataDevice()
                                        : (void *)m_uintPixels.dataDevice();
}

void Denoiser::init()
{
  if (m_denoiser)
    return;

  auto &state = *deviceState();

  OptixDenoiserOptions options = {};
  options.guideAlbedo = 0;
  options.guideNormal = 0;

  OPTIX_CHECK(optixDenoiserCreate(state.optixContext,
      OPTIX_DENOISER_MODEL_KIND_LDR,
      &options,
      &m_denoiser));
}

} // namespace visrtx