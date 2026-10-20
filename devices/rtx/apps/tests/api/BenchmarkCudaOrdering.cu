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

// Manual semantics probe, not a timing assertion. Uses the same legacy-default
// stream compilation mode as the device. Never change production ordering based
// only on this probe: pageable copies and readbacks impose additional edges.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static void check(cudaError_t error)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

__global__ void delay(unsigned long long cycles)
{
  const auto start = clock64();
  while (clock64() - start < cycles) {
  }
}

int main()
{
  int clockRate = 0;
  check(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0));
  // clockRate is kHz: about 30 ms, not a precision timing source.
  const auto cycles = 30ull * clockRate;
  check(cudaFree(nullptr));
  for (unsigned flags :
      {unsigned(cudaStreamDefault), unsigned(cudaStreamNonBlocking)}) {
    cudaStream_t stream;
    cudaEvent_t done;
    check(cudaStreamCreateWithFlags(&stream, flags));
    check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    for (bool producerDefault : {true, false}) {
      delay<<<1, 1, 0, producerDefault ? nullptr : stream>>>(cycles);
      check(cudaGetLastError());
      // Record completion of an otherwise empty consumer stream. With legacy
      // NULL-stream ordering this event cannot pass the blocking producer.
      check(cudaEventRecord(done, producerDefault ? stream : nullptr));
      const auto start = std::chrono::steady_clock::now();
      check(cudaEventSynchronize(done));
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
                            .count();
      printf("flags=%s producer=%s consumer=%s consumer_wait_ms=%.3f\n",
          flags == cudaStreamDefault ? "blocking" : "nonblocking",
          producerDefault ? "default" : "work",
          producerDefault ? "work" : "default",
          ms);
      check(cudaDeviceSynchronize());
    }
    check(cudaEventDestroy(done));
    check(cudaStreamDestroy(stream));
  }
}
