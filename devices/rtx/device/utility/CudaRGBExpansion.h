// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

namespace visrtx {

// Expand device RGB storage directly into an RGBA CUDA array, preserving
// component bits. dst must have cudaArraySurfaceLoadStore and four channels
// matching T's width; src must contain a texel for every destination element.
// Completes on the default stream before returning. Instantiated for float
// and uint8_t/uint16_t/uint32_t; pad supplies the opaque alpha encoding.
template <typename T>
cudaError_t expandRGBtoRGBADevice(const T *src, cudaArray_t dst, T pad);

} // namespace visrtx
