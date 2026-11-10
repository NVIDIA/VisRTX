// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu/gpu_decl.h"

#include <cstdlib>

namespace visrtx {
namespace detail {

// Sampler programmer errors must fail even with NDEBUG. Kept internal to the
// sampler contracts; this is not a renderer/frame failure mechanism.
[[noreturn]] VISRTX_HOST_DEVICE void samplerContractFailure()
{
#ifdef __CUDA_ARCH__
  asm volatile("trap;");
  __builtin_unreachable();
#else
  std::abort();
#endif
}

} // namespace detail
} // namespace visrtx
