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

#pragma once

#include "gpu/gpu_util.h"

namespace visrtx {

VISRTX_DEVICE int computePixelX(int idx, int checkerboardID)
{
  return checkerboardID < 0 ? idx : idx * 2 + (checkerboardID & 0x1);
}

VISRTX_DEVICE int computePixelY(int idy, int checkerboardID)
{
  return checkerboardID < 0 ? idy : idy * 2 + ((checkerboardID >> 1) & 0x1);
}

VISRTX_DEVICE ScreenSample createScreenSample(const FrameGPUData &frameData)
{
  ScreenSample ss;

  // random state //

  const auto launchIdx = optixGetLaunchIndex();
  const int x = computePixelX(launchIdx.x, frameData.fb.checkerboardID);
  const int y = computePixelY(launchIdx.y, frameData.fb.checkerboardID);
  const int w = frameData.fb.size.x;
  // Hash the pixel/frame keys before PCG seeding. Adjacent PCG streams with
  // tiny seeds expose visible structure when AO consumes only a few samples.
  const uint64_t pixelLinear = uint64_t(y) * uint64_t(w) + uint64_t(x);
  const uint64_t streamId = detail::pcg_mix64(pixelLinear);
  const uint64_t frameSeed =
      detail::pcg_mix64((uint64_t(frameData.fb.frameID) << 32u) ^ pixelLinear
          ^ 0xD1B54A32D192ED03ULL);
  pcg_init(&ss.rs, frameSeed, streamId);

  ss.pixel.x = x;
  ss.pixel.y = y;
  ss.frameData = &frameData;
  ss.shadowContribWeight = 1.0f;

  return ss;
}

} // namespace visrtx
