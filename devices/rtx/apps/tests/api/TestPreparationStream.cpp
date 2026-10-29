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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <thread>
#include "../../../device/utility/PreparationStream.h"

namespace {
struct Gate
{
  std::atomic<bool> released{false};
  std::atomic<bool> timedOut{false};
  std::mutex mutex;
  std::condition_variable condition;
};

void CUDART_CB block(void *data)
{
  auto &gate = *static_cast<Gate *>(data);
  while (!gate.released.load())
    std::this_thread::yield();
}

bool check(cudaError_t error)
{
  if (error == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
  return false;
}
} // namespace

int main()
{
  visrtx::PreparationStream preparation;
  cudaStream_t other{};
  cudaEvent_t done{};
  int *device = nullptr;
  if (!check(preparation.initialize())
      || !check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking))
      || !check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming))
      || !check(
          cudaMalloc(reinterpret_cast<void **>(&device), 2 * sizeof(int))))
    return 1;
  unsigned flags = 0;
  bool ok = check(cudaStreamGetFlags(preparation.stream(), &flags))
      && (flags & cudaStreamNonBlocking);

  // Exercise both explicit and legacy-default producers, plus independence
  // from an unrelated nonblocking stream. All allocations precede gated work.
  for (int mode : {0, 1, 2}) {
    const bool dependency = mode != 0;
    const cudaStream_t producer = mode == 2 ? nullptr : other;
    ok =
        check(cudaMemsetAsync(device, 0, 2 * sizeof(int), preparation.stream()))
        && check(preparation.finish()) && ok;
    Gate gate;
    // Watchdog is a deadlock escape, not a latency assertion. A stream-wide
    // or device-wide accidental wait must fail rather than hang the suite.
    std::thread watchdog([&] {
      std::unique_lock<std::mutex> lock(gate.mutex);
      if (!gate.condition.wait_for(lock, std::chrono::seconds(5), [&] {
            return gate.released.load();
          })) {
        gate.timedOut = true;
        gate.released = true;
      }
    });
    ok = check(cudaLaunchHostFunc(producer, block, &gate)) && ok;
    int result = 0;
    if (dependency) {
      ok = check(cudaMemsetAsync(device, 0x2a, sizeof(int), producer)) && ok;
      ok = check(preparation.waitFor(producer)) && ok;
    } else {
      // Importing the default stream must not join an unrelated nonblocking
      // stream. The mode above separately tests a pending default producer.
      ok = check(preparation.waitFor(nullptr)) && ok;
      ok = check(
               cudaMemsetAsync(device, 0x2a, sizeof(int), preparation.stream()))
          && ok;
    }
    // A real consumer distinguishes importing producer data from merely
    // waiting for the stream later during host readback.
    ok = check(cudaMemcpyAsync(device + 1,
             device,
             sizeof(int),
             cudaMemcpyDeviceToDevice,
             preparation.stream()))
        && ok;
    // Keep readback after releasing the gate: a pageable D2H copy itself may
    // block the host and would obscure what the stream dependency test proves.
    ok = check(cudaEventRecord(done, preparation.stream())) && ok;
    if (dependency)
      ok = cudaEventQuery(done) == cudaErrorNotReady && ok;
    else {
      ok = check(preparation.finish()) && ok;
      ok = cudaStreamQuery(other) == cudaErrorNotReady && ok;
    }
    {
      std::lock_guard<std::mutex> lock(gate.mutex);
      gate.released = true;
    }
    gate.condition.notify_one();
    watchdog.join();
    ok = !gate.timedOut && check(preparation.finish()) && ok;
    ok = check(cudaMemcpyAsync(&result,
             device + 1,
             sizeof(result),
             cudaMemcpyDeviceToHost,
             preparation.stream()))
        && check(preparation.finish()) && result == 0x2a2a2a2a && ok;
    ok = check(cudaStreamSynchronize(producer)) && ok;
  }
  cudaFree(device);
  cudaEventDestroy(done);
  cudaStreamDestroy(other);
  preparation.reset();
  std::printf("preparation stream dependencies: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
