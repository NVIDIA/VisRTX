// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CustomFieldSampler.h"
#include "CustomFieldSampler_ptx.h"

namespace visrtx {

ptx_blob CustomFieldSampler::ptx()
{
  return {CustomFieldSampler_ptx, sizeof(CustomFieldSampler_ptx)};
}

} // namespace visrtx
