// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "utility/CudaRGBExpansion.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

static void check(bool ok, const char *message)
{
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

static void checkCUDA(cudaError_t error)
{
  check(error == cudaSuccess, cudaGetErrorString(error));
}

static void freeCUDA(void *ptr)
{
  checkCUDA(cudaFree(ptr));
}

static void freeArray(cudaArray *array)
{
  checkCUDA(cudaFreeArray(array));
}

template <typename T>
static void exercise(
    cudaExtent extent, T alpha, bool managed, bool half = false)
{
  // Offset input and non-block-aligned lengths exercise windowed RGB sources.
  // Compare components bitwise, including -0.0f and raw FLOAT16 encodings.
  const size_t width = extent.width;
  const size_t height = extent.height ? extent.height : 1;
  const size_t depth = extent.depth ? extent.depth : 1;
  const size_t texels = width * height * depth;
  constexpr size_t PREFIX = 3;
  std::vector<T> rgb((texels + PREFIX + 1) * 3);
  for (size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = T(i % 251);
  rgb[PREFIX * 3] = T(-0.0f);
  std::vector<T> expected(texels * 4);
  for (size_t i = 0; i < texels; ++i) {
    for (size_t c = 0; c < 3; ++c)
      expected[4 * i + c] = rgb[(PREFIX + i) * 3 + c];
    expected[4 * i + 3] = alpha;
  }
  void *src = nullptr;
  cudaArray_t dst = nullptr;
  checkCUDA(managed ? cudaMallocManaged(&src, rgb.size() * sizeof(T))
                    : cudaMalloc(&src, rgb.size() * sizeof(T)));
  std::unique_ptr<void, decltype(&freeCUDA)> source(src, freeCUDA);
  const int bits = int(sizeof(T) * 8);
  const auto desc = cudaCreateChannelDesc(bits,
      bits,
      bits,
      bits,
      std::is_same_v<T, float> || half ? cudaChannelFormatKindFloat
                                       : cudaChannelFormatKindUnsigned);
  checkCUDA(cudaMalloc3DArray(&dst, &desc, extent, cudaArraySurfaceLoadStore));
  std::unique_ptr<cudaArray, decltype(&freeArray)> destination(dst, freeArray);
  checkCUDA(cudaMemcpy(
      src, rgb.data(), rgb.size() * sizeof(T), cudaMemcpyHostToDevice));
  std::vector<T> actual(expected.size(), T(17));
  // Repeat writes through newly created surfaces onto the same CUDA array.
  for (int iteration = 0; iteration < 2; ++iteration) {
    rgb[PREFIX * 3 + 1] = T(iteration + 19);
    expected[1] = rgb[PREFIX * 3 + 1];
    checkCUDA(cudaMemcpy(
        src, rgb.data(), rgb.size() * sizeof(T), cudaMemcpyHostToDevice));
    // A synchronous argument error must not be attributed to a later,
    // successful expansion. Leave it in the thread's last-error slot.
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, -1) == cudaErrorInvalidDevice,
        "failed to seed an unrelated CUDA error");
    checkCUDA(visrtx::expandRGBtoRGBADevice(
        static_cast<T *>(src) + PREFIX * 3, dst, alpha));
    cudaGetLastError();
    cudaMemcpy3DParms copy{};
    copy.srcArray = dst;
    copy.dstPtr = make_cudaPitchedPtr(
        actual.data(), width * 4 * sizeof(T), width, height);
    copy.extent = make_cudaExtent(width, height, depth);
    copy.kind = cudaMemcpyDeviceToHost;
    checkCUDA(cudaMemcpy3D(&copy));
    check(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T))
            == 0,
        "RGB surface expansion changed components or alpha");
  }
}

int main()
{
  // Invalid destination flags return a CUDA status, not an exception.
  cudaArray_t noSurface = nullptr;
  const auto desc = cudaCreateChannelDesc<float4>();
  checkCUDA(cudaMallocArray(&noSurface, &desc, 1));
  std::unique_ptr<cudaArray, decltype(&freeArray)> destination(
      noSurface, freeArray);
  check(visrtx::expandRGBtoRGBADevice<float>(nullptr, noSurface, 1.f)
          == cudaErrorInvalidValue,
      "array without surface support must be rejected");

  for (bool managed : {false, true}) {
    for (auto extent : {make_cudaExtent(1, 0, 0),
             make_cudaExtent(257, 0, 0),
             make_cudaExtent(257, 1, 0),
             make_cudaExtent(17, 19, 0),
             make_cudaExtent(7, 11, 5)}) {
      exercise<float>(extent, 1.f, managed);
      exercise<uint8_t>(extent, std::numeric_limits<uint8_t>::max(), managed);
      exercise<uint16_t>(extent, std::numeric_limits<uint16_t>::max(), managed);
      exercise<uint16_t>(extent, 0x3c00, managed, true);
      exercise<uint32_t>(extent, std::numeric_limits<uint32_t>::max(), managed);
    }
  }
  std::puts("RGB expansion passed for float32, float16 bits, and uint8/16/32");
}
