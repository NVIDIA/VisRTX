/*
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>

namespace visrtx {

// Must match the order in which the shaders are pushed to the SBT in
// Renderer.cpp
enum class SurfaceShaderEntryPoints
{
  Initialize = 0,
  EvaluateNextRay,
  EvaluateTint,
  EvaluateOpacity,
  EvaluateEmission,
  EvaluateTransmission,
  Shade,
  Count
};

enum class SpatialFieldSamplerEntryPoints
{
  Init = 0,
  Sample,
  Count
};

enum class SbtCallableEntryPoints : uint32_t
{
  Invalid = ~0u,
  Matte = 0,
  PBR = Matte + int(SurfaceShaderEntryPoints::Count),
  SpatialFieldSamplerRegular = PBR + int(SurfaceShaderEntryPoints::Count),
  SpatialFieldSamplerNvdbFp4 =
      SpatialFieldSamplerRegular + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbFp8 =
      SpatialFieldSamplerNvdbFp4 + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbFp16 =
      SpatialFieldSamplerNvdbFp8 + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbFpN =
      SpatialFieldSamplerNvdbFp16 + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbFloat =
      SpatialFieldSamplerNvdbFpN + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerRectilinear =
      SpatialFieldSamplerNvdbFloat + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbRectilinearFp4 = SpatialFieldSamplerRectilinear
      + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbRectilinearFp8 = SpatialFieldSamplerNvdbRectilinearFp4
      + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbRectilinearFp16 = SpatialFieldSamplerNvdbRectilinearFp8
      + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbRectilinearFpN = SpatialFieldSamplerNvdbRectilinearFp16
      + int(SpatialFieldSamplerEntryPoints::Count),
  SpatialFieldSamplerNvdbRectilinearFloat =
      SpatialFieldSamplerNvdbRectilinearFpN
      + int(SpatialFieldSamplerEntryPoints::Count),
  Last = SpatialFieldSamplerNvdbRectilinearFloat
      + int(SpatialFieldSamplerEntryPoints::Count),
};

} // namespace visrtx
