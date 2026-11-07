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

#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include "../../../device/utility/StreamRetirement.h"

namespace {
cudaStream_t tracked{};
std::atomic<int> waits{0};
std::atomic<bool> failNext{false};
std::atomic<bool> released{false};
std::atomic<bool> returned{false};
std::mutex mutex;
std::condition_variable condition;
void CUDART_CB block(void *)
{
  while (!released.load())
    std::this_thread::yield();
}
} // namespace

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
  if (stream == tracked) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++waits;
    }
    condition.notify_all();
  }
  auto real = reinterpret_cast<decltype(&cudaStreamSynchronize)>(
      dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
  if (!real)
    std::abort();
  const auto error = real(stream);
  if (stream == tracked && failNext.exchange(false))
    return cudaErrorUnknown;
  return error;
}

int main()
{
  if (cudaStreamCreateWithFlags(&tracked, cudaStreamNonBlocking) != cudaSuccess)
    return 1;
  visrtx::StreamRetirement retirement;
  retirement.initialize(tracked);
  bool ok = !retirement.isRetired();
  ok = retirement.retire() == cudaSuccess && waits == 1
      && retirement.isRetired() && ok;
  for (int i = 0; i < 64; ++i)
    ok = retirement.retire() == cudaSuccess && waits == 1 && ok;

  int *device = nullptr;
  if (cudaMalloc(reinterpret_cast<void **>(&device), sizeof(int))
      != cudaSuccess)
    return 1;
  retirement.invalidate();
  ok = !retirement.isRetired() && ok;
  ok = cudaLaunchHostFunc(tracked, block, nullptr) == cudaSuccess && ok;
  ok = cudaMemsetAsync(device, 0x2a, sizeof(int), tracked) == cudaSuccess && ok;
  int gpu = 0;
  cudaGetDevice(&gpu);
  cudaError_t completion = cudaErrorUnknown;
  std::thread waiter([&] {
    cudaSetDevice(gpu);
    completion = retirement.retire();
    {
      std::lock_guard<std::mutex> lock(mutex);
      returned = true;
    }
    condition.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    const bool entered = condition.wait_for(lock, std::chrono::seconds(5), [] {
      return waits >= 2 || returned.load();
    });
    ok = entered && waits == 2 && !returned && ok;
    // A submission announced during the wait must not inherit its certificate.
    retirement.invalidate();
    released = true;
  }
  waiter.join();
  ok = completion == cudaSuccess && !retirement.isRetired() && ok;
  int result = 0;
  ok = cudaMemcpy(&result, device, sizeof(int), cudaMemcpyDeviceToHost)
          == cudaSuccess
      && result == 0x2a2a2a2a && ok;
  ok = retirement.retire() == cudaSuccess && waits == 3
      && retirement.isRetired() && ok;
  ok = retirement.retire() == cudaSuccess && waits == 3 && ok;

  retirement.invalidate();
  failNext = true;
  ok = retirement.retire() == cudaErrorUnknown && waits == 4
      && !retirement.isRetired() && ok;
  ok = retirement.retire() == cudaSuccess && waits == 5
      && retirement.isRetired() && ok;
  ok = retirement.retire() == cudaSuccess && waits == 5 && ok;
  cudaFree(device);
  cudaStreamDestroy(tracked);
  std::printf(
      "stream retirement waits=%d %s\n", int(waits), ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
