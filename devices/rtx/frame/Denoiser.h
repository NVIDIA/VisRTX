/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Object.h"
#include "optix_visrtx.h"
#include "utility/DeviceBuffer.h"
#include "utility/HostDeviceArray.h"

namespace visrtx {

struct Denoiser : public Object
{
  Denoiser(DeviceGlobalState *s);
  ~Denoiser() override;

  void setup(
      uvec2 size, HostDeviceArray<uint8_t> &pixelBuffer, ANARIDataType format);
  void cleanup();

  void launch();

  void *mapColorBuffer();
  void *mapGPUColorBuffer();

 private:
  void init();

  // Data //

  ANARIDataType m_format{ANARI_UNKNOWN};

  OptixDenoiser m_denoiser{nullptr};
  OptixDenoiserParams m_params{};
  OptixDenoiserGuideLayer m_guideLayer{};
  OptixDenoiserLayer m_layer;

  HostDeviceArray<uint8_t> *m_pixelBuffer{nullptr};

  DeviceBuffer m_state;
  DeviceBuffer m_scratch;

  // This buffer is only used when format != ANARI_FLOAT32_VEC4
  HostDeviceArray<uint32_t> m_uintPixels;
};

} // namespace visrtx
