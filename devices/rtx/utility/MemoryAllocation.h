/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

// anari
#include <anari/anari_cpp.hpp>

namespace visrtx {

struct HostMemoryAllocation;
struct CudaMemoryAllocation;

struct HostMemoryAllocation
{
  HostMemoryAllocation(void *externalHostMemory,
      size_t bytes,
      anari::MemoryDeleter deleter = nullptr,
      const void *deleterPtr = nullptr);
  HostMemoryAllocation(size_t bytes);
  HostMemoryAllocation() = default;
  ~HostMemoryAllocation();

  void *ptr() const;
  size_t bytes() const;

  bool isOwner() const;
  bool isValid() const;
  operator bool() const;

  HostMemoryAllocation &operator=(const CudaMemoryAllocation &o);
  HostMemoryAllocation(const CudaMemoryAllocation &o);

  void privatize(); // if !isOwner(), then make internal copy

  // Movable, not copyable //
  HostMemoryAllocation(HostMemoryAllocation &&);
  HostMemoryAllocation &operator=(HostMemoryAllocation &&);
  HostMemoryAllocation(const HostMemoryAllocation &) = delete;
  HostMemoryAllocation &operator=(const HostMemoryAllocation &) = delete;
  ///////////////////////////

 private:
  void freeMemory();
  void clearPointers();

  size_t m_bytes{0};
  void *m_ptr{nullptr};
  anari::MemoryDeleter m_deleter{nullptr};
  const void *m_deleterPtr{nullptr};
  bool m_owner{false};
};

struct CudaMemoryAllocation
{
  CudaMemoryAllocation(void *externalDeviceMemory,
      size_t bytes,
      anari::MemoryDeleter deleter = nullptr,
      const void *deleterPtr = nullptr);
  CudaMemoryAllocation(size_t bytes);
  CudaMemoryAllocation() = default;
  ~CudaMemoryAllocation();

  void *ptr() const;
  size_t bytes() const;

  bool isOwner() const;
  bool isValid() const;
  operator bool() const;

  CudaMemoryAllocation &operator=(const HostMemoryAllocation &o);
  CudaMemoryAllocation(const HostMemoryAllocation &o);

  void privatize(); // if !isOwner(), then make internal copy

  // Movable, not copyable //
  CudaMemoryAllocation(CudaMemoryAllocation &&);
  CudaMemoryAllocation &operator=(CudaMemoryAllocation &&);
  CudaMemoryAllocation(const CudaMemoryAllocation &) = delete;
  CudaMemoryAllocation &operator=(const CudaMemoryAllocation &) = delete;
  ///////////////////////////

 private:
  void freeMemory();
  void clearPointers();

  size_t m_bytes{0};
  void *m_ptr{nullptr};
  anari::MemoryDeleter m_deleter{nullptr};
  const void *m_deleterPtr{nullptr};
  bool m_owner{false};
};

} // namespace visrtx
