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

#include <cstdio>
#include "gpu/gpu_objects.h"
#include "utility/DeviceObjectArray.h"
#include "utility/PreparationStream.h"

int main()
{
  visrtx::PreparationStream preparation;
  if (preparation.initialize() != cudaSuccess)
    return 1;
  visrtx::DeviceObjectArray<int> registry;
  int token = 0;
  const auto first = registry.alloc(&token);
  registry.set(first, 11);
  if (preparation.upload(nullptr, registry) != cudaSuccess)
    return 1;
  registry.markUploaded();
  bool ok = !registry.dirty();

  registry.set(first, 22);
  if (registry.enqueueUpload(preparation.stream()) != cudaSuccess)
    return 1;
  // Model object creation/finalization between submission and completion:
  // growing the live host vector must not invalidate the copy's source, and
  // acknowledging that copy must not erase newer edits.
  registry.set(first, 33);
  visrtx::DeviceObjectIndex last = first;
  for (int i = 0; i < 1024; ++i)
    last = registry.alloc(&token);
  registry.set(last, 44);
  if (preparation.finish() != cudaSuccess)
    return 1;
  registry.markUploaded();
  ok = registry.dirty() && ok;
  int value = 0;
  ok = cudaMemcpy(
           &value, registry.devicePtr(), sizeof(value), cudaMemcpyDeviceToHost)
          == cudaSuccess
      && value == 22 && ok;

  if (preparation.upload(nullptr, registry) != cudaSuccess)
    return 1;
  registry.markUploaded();
  ok = !registry.dirty() && ok;
  ok = cudaMemcpy(
           &value, registry.devicePtr(), sizeof(value), cudaMemcpyDeviceToHost)
          == cudaSuccess
      && value == 33 && ok;
  ok = cudaMemcpy(&value,
           registry.devicePtr() + last,
           sizeof(value),
           cudaMemcpyDeviceToHost)
          == cudaSuccess
      && value == 44 && ok;

  visrtx::DeviceObjectArray<int> uncommitted;
  uncommitted.alloc(&token);
  uncommitted.alloc(&token);
  if (preparation.upload(nullptr, uncommitted) != cudaSuccess)
    return 1;
  uncommitted.markUploaded();
  uncommitted.set(0, 7);
  if (preparation.upload(nullptr, uncommitted) != cudaSuccess)
    return 1;
  uncommitted.markUploaded();
  int pair[2]{};
  ok = cudaMemcpy(
           pair, uncommitted.devicePtr(), sizeof(pair), cudaMemcpyDeviceToHost)
          == cudaSuccess
      && pair[0] == 7 && pair[1] == 0 && !uncommitted.dirty() && ok;
  std::printf("registry snapshot: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
