// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnalyticalFieldSampler.h"
#include "AnalyticalFieldSampler_ptx.h"

namespace visrtx {

ptx_blob AnalyticalFieldSampler::ptx()
{
  return {AnalyticalFieldSampler_ptx, sizeof(AnalyticalFieldSampler_ptx)};
}

} // namespace visrtx
