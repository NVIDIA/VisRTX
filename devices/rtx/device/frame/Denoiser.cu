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

#include "Denoiser.h"
#include "gpu/gpu_util.h"
#include "utility/DeviceBuffer.h"
#include "utility/instrument.h"
// optix
#include <optix_denoiser_tiling.h>
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

void Denoiser::setup(uvec2 size,
    HostDeviceArray<uint8_t> &outputBuffer,
    ANARIDataType format,
    DeviceBuffer &input,
    DeviceBuffer &albedo,
    DeviceBuffer &normal)
{
  init(albedo, normal);
  auto &state = *deviceState();

  m_pixelBuffer = &outputBuffer;

  m_format = format;

  // OptiX builds an internal tensor whose element count must fit in a uint32.
  // At full-frame resolution large frames overflow that cap, so denoise in
  // overlapping tiles (optixUtilDenoiserInvokeTiled). kMaxTile keeps the
  // per-tile tensor well under the limit; frames smaller than a tile collapse
  // to a single full-frame invoke with no overhead.
  constexpr uint32_t kMaxTile = 2048;
  m_tileW = std::min<uint32_t>(kMaxTile, size.x);
  m_tileH = std::min<uint32_t>(kMaxTile, size.y);

  OptixDenoiserSizes sizes;
  OPTIX_CHECK(
      optixDenoiserComputeMemoryResources(m_denoiser, m_tileW, m_tileH, &sizes));

  m_overlap = sizes.overlapWindowSizeInPixels;
  m_state.reserve(sizes.stateSizeInBytes);
  m_scratch.reserve(sizes.withOverlapScratchSizeInBytes);
  m_intensity.reserve(sizeof(float));

  if (format != ANARI_FLOAT32_VEC4)
    m_uintPixels.resize(size_t(size.x) * size_t(size.y));
  else
    m_uintPixels.clear();

  // Setup must be sized for the largest padded tile the tiled invoke feeds.
  OPTIX_CHECK(optixDenoiserSetup(m_denoiser,
      state.stream,
      m_tileW + 2 * m_overlap,
      m_tileH + 2 * m_overlap,
      (CUdeviceptr)m_state.ptr(),
      m_state.bytes(),
      (CUdeviceptr)m_scratch.ptr(),
      m_scratch.bytes()));

  m_layer.input.data = (CUdeviceptr)input.ptr();
  m_layer.input.width = size.x;
  m_layer.input.height = size.y;
  m_layer.input.pixelStrideInBytes = 0;
  m_layer.input.rowStrideInBytes = 4 * sizeof(float) * size.x;
  m_layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  m_layer.output = m_layer.input;
  m_layer.output.data = (CUdeviceptr)outputBuffer.dataDevice();

  m_guideLayer.albedo.data = (CUdeviceptr)albedo.ptr();
  m_guideLayer.albedo.width = size.x;
  m_guideLayer.albedo.height = size.y;
  m_guideLayer.albedo.pixelStrideInBytes = 3 * sizeof(float);
  m_guideLayer.albedo.rowStrideInBytes = 3 * sizeof(float) * size.x;
  m_guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;

  m_guideLayer.normal.data = (CUdeviceptr)normal.ptr();
  m_guideLayer.normal.width = size.x;
  m_guideLayer.normal.height = size.y;
  m_guideLayer.normal.pixelStrideInBytes = 3 * sizeof(float);
  m_guideLayer.normal.rowStrideInBytes = 3 * sizeof(float) * size.x;
  m_guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
}

void Denoiser::cleanup()
{
  m_state.reset();
  m_scratch.reset();
}

void Denoiser::launch()
{
  auto &state = *deviceState();

  // Tiled invoke normalizes each tile independently unless handed a
  // whole-frame HDR intensity; without it adjacent tiles self-expose and seam.
  // Compute one average log intensity over the full input and share it.
  OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser,
      state.stream,
      &m_layer.input,
      (CUdeviceptr)m_intensity.ptr(),
      (CUdeviceptr)m_scratch.ptr(),
      m_scratch.bytes()));
  m_params.hdrIntensity = (CUdeviceptr)m_intensity.ptr();

  instrument::rangePush("optixDenoiserInvoke()");
  OPTIX_CHECK(optixUtilDenoiserInvokeTiled(m_denoiser,
      state.stream,
      &m_params,
      (CUdeviceptr)m_state.ptr(),
      m_state.bytes(),
      &m_guideLayer,
      &m_layer,
      1,
      (CUdeviceptr)m_scratch.ptr(),
      m_scratch.bytes(),
      m_overlap,
      m_tileW,
      m_tileH));
  instrument::rangePop(); // optixDenoiserInvoke()
}

void Denoiser::convertOutput()
{
  if (m_format == ANARI_FLOAT32_VEC4)
    return;
  auto &state = *deviceState();
  instrument::rangePush("denoiser transform pixels");
  auto numPixels = size_t(m_layer.output.width) * size_t(m_layer.output.height);
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

void Denoiser::init(
    const DeviceBuffer &accumAlbedo, const DeviceBuffer &accumNormal)
{
  const bool useAlbedo = accumAlbedo.ptr() != nullptr;
  const bool useNormal = accumNormal.ptr() != nullptr;

  if (m_denoiser
      && (m_usingAlbedo != useAlbedo || m_usingNormal != useNormal)) {
    OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
    m_denoiser = {};
  }

  m_usingAlbedo = useAlbedo;
  m_usingNormal = useNormal;

  OptixDenoiserOptions options = {};
  options.guideAlbedo = m_usingAlbedo;
  options.guideNormal = m_usingNormal;

  if (!m_denoiser) {
    auto &state = *deviceState();
    OPTIX_CHECK(optixDenoiserCreate(state.optixContext,
        OPTIX_DENOISER_MODEL_KIND_AOV,
        &options,
        &m_denoiser));
  }
}

} // namespace visrtx
