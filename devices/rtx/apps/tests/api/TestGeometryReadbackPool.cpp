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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_set>
#include <vector>
#include "../../../device/geometry/ReadbackPool.h"

namespace {
std::atomic<bool> failNext{false};
std::atomic<int> attempts{0};
std::atomic<int> allocations{0};
std::atomic<int> frees{0};

template <typename T>
T next(const char *name)
{
  auto function = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  if (!function)
    std::abort();
  return function;
}
} // namespace

extern "C" cudaError_t cudaMallocHost(void **ptr, size_t bytes)
{
  ++attempts;
  if (failNext.exchange(false)) {
    *ptr = nullptr;
    return next<decltype(&cudaMallocHost)>("cudaMallocHost")(ptr, size_t(-1));
  }
  const auto error =
      next<decltype(&cudaMallocHost)>("cudaMallocHost")(ptr, bytes);
  if (error == cudaSuccess)
    ++allocations;
  return error;
}

extern "C" cudaError_t cudaFreeHost(void *ptr)
{
  const auto error = next<decltype(&cudaFreeHost)>("cudaFreeHost")(ptr);
  if (error == cudaSuccess && ptr)
    ++frees;
  return error;
}

int main(int argc, char **argv)
{
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--no-failure") != 0))
    return 2;
  const bool injectFailures = argc == 1;
  using Pool = visrtx::GeometryReadbackPool;
  bool ok = true;
  {
    Pool pool;
    Pool::Lease failed;
    if (injectFailures) {
      failNext = true;
      ok = pool.acquire(failed) != cudaSuccess
          && cudaGetLastError() == cudaSuccess && !failed.data()
          && allocations == 0 && ok;
    }

    std::vector<Pool::Lease> leases(Pool::SLOTS_PER_PAGE + 1);
    for (size_t i = 0; i < leases.size(); ++i) {
      if (injectFailures && i == Pool::SLOTS_PER_PAGE) {
        failNext = true;
        ok = pool.acquire(leases[i]) != cudaSuccess
            && cudaGetLastError() == cudaSuccess && !leases[i].data()
            && allocations == 1 && ok;
      }
      if (pool.acquire(leases[i]) != cudaSuccess)
        return 1;
      leases[i].data()[0] = {float(i), uint32_t(i)};
      leases[i].data()[1] = {-float(i), uint32_t(i + 1)};
    }
    // Exercise independent asynchronous DMA targets on the same pinned page,
    // including writes to one slot while the CPU initializes a later slot.
    std::vector<visrtx::PreparationResult> source(2 * leases.size());
    for (size_t i = 0; i < leases.size(); ++i) {
      source[2 * i] = leases[i].data()[0];
      source[2 * i + 1] = leases[i].data()[1];
    }
    void *device = nullptr;
    cudaStream_t stream = nullptr;
    const size_t bytes = source.size() * sizeof(source[0]);
    if (cudaMalloc(&device, bytes) != cudaSuccess
        || cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
            != cudaSuccess
        || cudaMemcpyAsync(
               device, source.data(), bytes, cudaMemcpyHostToDevice, stream)
            != cudaSuccess)
      return 1;
    for (size_t i = 0; i < leases.size(); ++i) {
      leases[i].data()[0] = {};
      leases[i].data()[1] = {};
      if (cudaMemcpyAsync(leases[i].data(),
              static_cast<visrtx::PreparationResult *>(device) + 2 * i,
              2 * sizeof(source[0]),
              cudaMemcpyDeviceToHost,
              stream)
          != cudaSuccess)
        return 1;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return 1;
    cudaFree(device);
    cudaStreamDestroy(stream);
    std::unordered_set<void *> addresses;
    for (size_t i = 0; i < leases.size(); ++i) {
      addresses.insert(leases[i].data());
      ok = leases[i].data()[0].value == float(i)
          && leases[i].data()[1].invalid == i + 1 && ok;
    }
    ok = addresses.size() == leases.size() && allocations == 2 && frees == 0
        && ok;
    auto *first = leases[0].data();
    Pool::Lease moved(std::move(leases[0]));
    ok = !leases[0].data() && moved.data() == first && ok;
    leases[1] = std::move(moved); // Returns its previous slot exactly once.
    ok = !moved.data() && leases[1].data() == first && ok;
    Pool::Lease replacement;
    if (pool.acquire(replacement) != cudaSuccess)
      return 1;
    for (auto &lease : leases)
      ok = lease.data() != replacement.data() && ok;
    replacement.reset();
    replacement.reset();
    leases.clear();

    // A full release/reacquire cycle stays within the two cached pages.
    leases.resize(Pool::SLOTS_PER_PAGE + 1);
    addresses.clear();
    for (auto &lease : leases) {
      if (pool.acquire(lease) != cudaSuccess)
        return 1;
      addresses.insert(lease.data());
    }
    ok = addresses.size() == leases.size() && allocations == 2 && frees == 0
        && ok;
  }
  ok = frees == 2 && ok;

  Pool::Lease survivor;
  {
    Pool pool;
    if (pool.acquire(survivor) != cudaSuccess)
      return 1;
    survivor.data()[0].value = 123.f;
  }
  ok = frees == 2 && survivor.data()[0].value == 123.f && ok;
  survivor.reset();
  ok = frees == 3 && ok;

  {
    Pool pool;
    Pool::Lease warm;
    if (pool.acquire(warm) != cudaSuccess)
      return 1;
    warm.reset();
    std::atomic<bool> concurrentOK{true};
    std::vector<std::thread> workers;
    for (unsigned id = 0; id < 8; ++id) {
      workers.emplace_back([&, id] {
        for (unsigned i = 0; i < 1000; ++i) {
          Pool::Lease lease;
          if (pool.acquire(lease) != cudaSuccess) {
            concurrentOK = false;
            return;
          }
          const float value = float(id * 1000 + i);
          lease.data()[0].value = value;
          lease.data()[1].invalid = id;
          std::this_thread::yield();
          if (lease.data()[0].value != value || lease.data()[1].invalid != id)
            concurrentOK = false;
        }
      });
    }
    for (auto &worker : workers)
      worker.join();
    ok = concurrentOK && allocations == 4 && ok;
  }
  ok = allocations == frees
      && attempts == allocations + (injectFailures ? 2 : 0) && ok;
  std::printf("readback pool: attempts=%d allocations=%d frees=%d %s\n",
      int(attempts.load()),
      int(allocations.load()),
      int(frees.load()),
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
