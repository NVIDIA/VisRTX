// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <cuda_runtime.h>
#include <anari/anari_cpp.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>

static void check(bool condition, const char *message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

static void checkCUDA(cudaError_t result)
{
  check(result == cudaSuccess, cudaGetErrorString(result));
}

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  // Repeated successful map/unmap cycles must not warn about mapping state.
  check(severity != ANARI_SEVERITY_FATAL_ERROR
          && severity != ANARI_SEVERITY_ERROR
          && severity != ANARI_SEVERITY_WARNING,
      message);
}

static void freeCUDA(void *ptr)
{
  checkCUDA(cudaFree(ptr));
}

static void capturedDeleter(const void *userData, const void *ptr)
{
  ++*static_cast<unsigned *>(const_cast<void *>(userData));
  freeCUDA(const_cast<void *>(ptr));
}

static void exercise(ANARIDevice device,
    void *memory,
    bool cudaMemory,
    ANARIMemoryDeleter deleter = nullptr,
    const void *userData = nullptr)
{
  auto array =
      anariNewArray1D(device, memory, deleter, userData, ANARI_FLOAT32, 4);
  check(array != nullptr, "array creation failed");
  for (int iteration = 0; iteration < 2; ++iteration) {
    void *mapped = anariMapArray(device, array);
    check(mapped != nullptr, "mapping returned null");
    check(!memory || mapped == memory, "mapping changed application pointer");
    const float values[4] = {float(iteration), 1.f, 2.f, 3.f};
    if (cudaMemory) {
      checkCUDA(
          cudaMemcpy(mapped, values, sizeof(values), cudaMemcpyHostToDevice));
      checkCUDA(cudaDeviceSynchronize());
    } else {
      auto *out = static_cast<float *>(mapped);
      for (int i = 0; i < 4; ++i)
        out[i] = values[i];
    }
    anariUnmapArray(device, array);
  }
  anariRelease(device, array);
}

int main()
{
  unsigned deleterCalls = 0;
  auto device = makeVisRTXDevice(statusFunc);
  check(device != nullptr, "device creation failed");

  float host[4] = {};
  exercise(device, host, false);
  exercise(device, nullptr, false);

  for (bool managed : {false, true}) {
    void *ptr = nullptr;
    checkCUDA(managed ? cudaMallocManaged(&ptr, sizeof(host))
                      : cudaMalloc(&ptr, sizeof(host)));
    std::unique_ptr<void, decltype(&freeCUDA)> allocation(ptr, freeCUDA);
    exercise(device, ptr, true);
  }

  void *captured = nullptr;
  checkCUDA(cudaMalloc(&captured, sizeof(host)));
  exercise(device, captured, true, capturedDeleter, &deleterCalls);
  anariRelease(device, device);
  check(deleterCalls == 1, "captured allocation must be deleted exactly once");
  return 0;
}
