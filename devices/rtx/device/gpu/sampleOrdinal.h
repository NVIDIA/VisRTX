// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu/gpu_decl.h"

namespace visrtx {

// completedSamples already counts samples, not launches. Supported inputs:
// nonnegative completedSamples, positive effective spp, 0 <= iteration < spp,
// and completedSamples + spp <= INT_MAX (the existing signed frame counter).
// The sampler has a uint32 period of 2^32; this does not extend the frame
// limit.
VISRTX_HOST_DEVICE uint32_t sampleOrdinal(int completedSamples, int iteration)
{
  return uint32_t(completedSamples) + uint32_t(iteration);
}

} // namespace visrtx
