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
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include "../../../device/utility/DeviceAllocationCache.h"
#include "../../../device/utility/DeviceBuffer.h"
#include "../../../device/utility/MemoryAllocation.h"

namespace {
std::atomic<int> allocations{0}, frees{0}, syncs{0}, attempts{0};
size_t failBytes = 0;
bool failSync = false;
int capturedDeletes = 0;
std::atomic<bool> trackSync{false}, syncEntered{false}, gateReleased{false},
    cohortReturned{false};
std::mutex gateMutex;
std::condition_variable gateCondition;
void CUDART_CB blockConsumer(void *)
{
  while (!gateReleased.load())
    std::this_thread::yield();
}
template <typename T>
T next(const char *name)
{
  auto result = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  if (!result)
    std::abort();
  return result;
}
void capturedDelete(const void *, const void *pointer)
{
  ++capturedDeletes;
  cudaFree(const_cast<void *>(pointer));
}
} // namespace

extern "C" cudaError_t cudaMalloc(void **pointer, size_t bytes)
{
  ++attempts;
  if (failBytes && bytes == failBytes) {
    failBytes = 0;
    return next<decltype(&cudaMalloc)>("cudaMalloc")(
        pointer, std::numeric_limits<size_t>::max());
  }
  const auto error = next<decltype(&cudaMalloc)>("cudaMalloc")(pointer, bytes);
  if (error == cudaSuccess && *pointer)
    ++allocations;
  return error;
}
extern "C" cudaError_t cudaFree(void *pointer)
{
  if (pointer)
    ++frees;
  return next<decltype(&cudaFree)>("cudaFree")(pointer);
}
extern "C" cudaError_t cudaDeviceSynchronize()
{
  ++syncs;
  if (trackSync) {
    {
      std::lock_guard<std::mutex> lock(gateMutex);
      syncEntered = true;
    }
    gateCondition.notify_all();
  }
  const auto error =
      next<decltype(&cudaDeviceSynchronize)>("cudaDeviceSynchronize")();
  if (failSync) {
    failSync = false;
    return cudaErrorUnknown;
  }
  return error;
}

int main()
{
  using visrtx::DeviceAllocationCache;
  auto complete = [] { return cudaSuccess; };
  bool ok = true;
  {
    DeviceAllocationCache cache(128, 4);
    void *a = nullptr, *b = nullptr, *c = nullptr;
    ok = cache.allocate(32, &a) == cudaSuccess && ok;
    cache.release(a, 32);
    ok = cache.allocate(32, &b) == cudaSuccess && a != b && ok;
    const int beforeSyncs = syncs;
    // Deferred work must be submitted before the shared completion boundary.
    cudaStream_t stream{};
    ok =
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess
        && ok;
    ok = cache.reclaim([&] { return cudaMemsetAsync(a, 0x2a, 32, stream); })
            == cudaSuccess
        && syncs == beforeSyncs + 1 && ok;
    ok = cache.allocate(32, &c) == cudaSuccess && c == a && ok;
    int value = 0;
    ok = cudaMemcpy(&value, c, sizeof(value), cudaMemcpyDeviceToHost)
            == cudaSuccess
        && value == 0x2a2a2a2a && ok;
    cudaStreamDestroy(stream);
    cache.release(c, 32);
    cache.release(b, 32);
    const int beforeFailedFlush = syncs;
    ok = cache.reclaim([] { return cudaErrorInvalidValue; })
            == cudaErrorInvalidValue
        && syncs == beforeFailedFlush && ok;
    failSync = true;
    ok = cache.reclaim(complete) == cudaErrorUnknown && ok;
    void *d = nullptr;
    ok = cache.allocate(32, &d) == cudaSuccess && d != a && d != b && ok;
    cache.release(d, 32);
    ok = cache.reclaim(complete) == cudaSuccess && ok;
    cache.close();
    const int afterClose = frees;
    cache.close();
    ok = frees == afterClose && ok;
  }
  {
    // A consumer outside the preparation stream must also complete before a
    // returned block can be reused. The timeout only escapes a broken wait.
    DeviceAllocationCache cache;
    void *p = nullptr;
    ok = cache.allocate(32, &p) == cudaSuccess && ok;
    cudaStream_t consumer{};
    ok = cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking)
            == cudaSuccess
        && ok;
    ok = cudaLaunchHostFunc(consumer, blockConsumer, nullptr) == cudaSuccess
        && ok;
    ok = cudaMemsetAsync(p, 0x17, 32, consumer) == cudaSuccess && ok;
    cache.release(p, 32);
    int gpu = 0;
    cudaGetDevice(&gpu);
    cudaError_t completion = cudaErrorUnknown;
    trackSync = true;
    std::thread collector([&] {
      cudaSetDevice(gpu);
      completion = cache.reclaim(complete);
      {
        std::lock_guard<std::mutex> lock(gateMutex);
        cohortReturned = true;
      }
      gateCondition.notify_all();
    });
    {
      std::unique_lock<std::mutex> lock(gateMutex);
      const bool entered = gateCondition.wait_for(lock,
          std::chrono::seconds(5),
          [] { return syncEntered.load() || cohortReturned.load(); });
      ok = entered && syncEntered && !cohortReturned && ok;
      gateReleased = true;
    }
    collector.join();
    trackSync = false;
    ok = completion == cudaSuccess && ok;
    cudaStreamSynchronize(consumer);
    void *reused = nullptr;
    ok = cache.allocate(32, &reused) == cudaSuccess && reused == p && ok;
    int value = 0;
    ok = cudaMemcpy(&value, reused, sizeof(value), cudaMemcpyDeviceToHost)
            == cudaSuccess
        && value == 0x17171717 && ok;
    cache.release(reused, 32);
    cudaStreamDestroy(consumer);
  }
  {
    DeviceAllocationCache cache(64, 2);
    void *p[3]{};
    for (auto &pointer : p)
      ok = cache.allocate(32, &pointer) == cudaSuccess && ok;
    const int beforeFree = frees;
    for (auto pointer : p)
      cache.release(pointer, 32);
    ok =
        frees == beforeFree + 1 && ok; // Bound pending as well as ready blocks.
    ok = cache.reclaim(complete) == cudaSuccess && ok;
    void *large = nullptr;
    ok = cache.allocate(65, &large) == cudaSuccess && ok;
    cache.release(large, 65);
    ok = frees == beforeFree + 2 && ok;
  }
  {
    DeviceAllocationCache cache(1024, 1);
    void *a = nullptr, *b = nullptr;
    ok = cache.allocate(16, &a) == cudaSuccess && ok;
    ok = cache.allocate(16, &b) == cudaSuccess && ok;
    const int beforeFree = frees;
    cache.release(a, 16);
    cache.release(b, 16);
    ok =
        frees == beforeFree + 1 && ok; // Entry bound independent of byte bound.
  }
  {
    DeviceAllocationCache cache(32, 1);
    void *a = nullptr, *b = nullptr;
    ok = cache.allocate(32, &a) == cudaSuccess && ok;
    ok = cache.allocate(32, &b) == cudaSuccess && ok;
    cache.release(a, 32);
    const int beforeFree = frees;
    ok = cache.reclaim([&] {
      cache.release(b, 32); // Frozen blocks still consume the cache budget.
      return cudaSuccess;
    }) == cudaSuccess
        && frees == beforeFree + 1 && ok;
  }
  {
    DeviceAllocationCache cache(1024, 64);
    std::array<void *, 64> pointers{};
    for (auto &p : pointers)
      ok = cache.allocate(8, &p) == cudaSuccess && ok;
    const int beforeFree = frees;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker)
      workers.emplace_back([&, worker] {
        for (int i = worker; i < 64; i += 8)
          cache.release(pointers[i], 8);
      });
    for (auto &worker : workers)
      worker.join();
    ok = frees == beforeFree && cache.reclaim(complete) == cudaSuccess && ok;
  }
  {
    DeviceAllocationCache cache;
    void *a = nullptr, *b = nullptr, *c = nullptr;
    ok = cache.allocate(32, &a) == cudaSuccess && ok;
    ok = cache.allocate(32, &b) == cudaSuccess && ok;
    cache.release(a, 32);
    ok = cache.reclaim([&] {
      cache.release(b, 32); // Later returns cannot join this frozen cohort.
      return cudaSuccess;
    }) == cudaSuccess
        && ok;
    ok = cache.allocate(32, &c) == cudaSuccess && c == a && ok;
    void *d = nullptr;
    ok = cache.allocate(32, &d) == cudaSuccess && d != b && ok;
    cache.release(c, 32);
    cache.release(d, 32);
    bool threw = false;
    try {
      cache.reclaim([]() -> cudaError_t { throw std::runtime_error("test"); });
    } catch (const std::runtime_error &) {
      threw = true;
    }
    ok = threw && cache.reclaim(complete) == cudaSuccess && ok;
  }
  {
    DeviceAllocationCache cache;
    void *p = nullptr;
    ok = cache.allocate(32, &p) == cudaSuccess && ok;
    cache.release(p, 32);
    ok = cache.reclaim(complete) == cudaSuccess && ok;
    const int beforeAttempts = attempts, beforeFrees = frees;
    failBytes = 64;
    ok = cache.allocate(64, &p) == cudaSuccess && attempts == beforeAttempts + 2
        && frees == beforeFrees + 1 && ok;
    cache.release(p, 64);
    // With no ready blocks to trim, preserve the original allocation error.
    failBytes = 128;
    ok = cache.allocate(128, &p) == cudaErrorMemoryAllocation && !p && ok;
    cudaGetLastError();
  }
  {
    auto cache = std::make_shared<DeviceAllocationCache>();
    void *p = nullptr;
    ok = cache->allocate(32, &p) == cudaSuccess && ok;
    auto allocation = visrtx::CudaMemoryAllocation::fromCache(p, 32, cache);
    auto moved = std::move(allocation);
    ok = !allocation && moved.ptr() == p && moved.isOwner() && ok;
    cache->close();
    cache.reset(); // A late return must free, not resurrect a closed cache.
    const int beforeFree = frees;
    moved = visrtx::CudaMemoryAllocation();
    ok = frees == beforeFree + 1 && ok;
    ok = cudaMalloc(&p, 32) == cudaSuccess && ok;
    {
      visrtx::CudaMemoryAllocation captured(p, 32, capturedDelete);
    }
    ok = capturedDeletes == 1 && ok;
  }
  {
    // A failed reservation leaves the buffer empty, consumes the runtime
    // error, and lets the same size be retried.
    visrtx::DeviceBuffer buffer;
    failBytes = 96;
    ok = buffer.tryReserve(96) == cudaErrorMemoryAllocation && !buffer
        && buffer.bytes() == 0 && cudaGetLastError() == cudaSuccess && ok;
    ok = buffer.tryReserve(96) == cudaSuccess && buffer && buffer.bytes() >= 96
        && ok;
    const auto *storage = buffer.ptr();
    ok = buffer.tryReserve(48) == cudaSuccess && buffer.ptr() == storage
        && buffer.tryReserve(0) == cudaSuccess && buffer.ptr() == storage && ok;
    failBytes = 192;
    ok = buffer.tryReserve(192) == cudaErrorMemoryAllocation && !buffer
        && buffer.bytes() == 0 && cudaGetLastError() == cudaSuccess && ok;
    ok = buffer.tryReserve(192) == cudaSuccess && buffer.bytes() >= 192 && ok;
  }
  ok = allocations == frees && ok;
  std::printf("allocation cache allocations=%d frees=%d syncs=%d %s\n",
      int(allocations),
      int(frees),
      int(syncs),
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
