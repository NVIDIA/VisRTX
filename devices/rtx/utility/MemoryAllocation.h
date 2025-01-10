/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// cuda
#include <cuda_runtime.h>
// std
#include <cstdlib>

namespace visrtx {

struct HostMemoryAllocation;
struct CudaMemoryAllocation;

struct HostMemoryAllocation
{
  HostMemoryAllocation(void *externalHostMemory,
      size_t bytes,
      ANARIMemoryDeleter deleter = nullptr,
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

  void privatize(); // if !isOwner(), then make internal copy

  // Movable, not copyable //
  HostMemoryAllocation(HostMemoryAllocation &&);
  HostMemoryAllocation &operator=(HostMemoryAllocation &&);
  HostMemoryAllocation(const HostMemoryAllocation &) = delete;
  HostMemoryAllocation &operator=(const HostMemoryAllocation &) = delete;
  ///////////////////////////

 private:
  size_t m_bytes{0};
  void *m_ptr{nullptr};
  ANARIMemoryDeleter m_deleter{nullptr};
  const void *m_deleterPtr{nullptr};
  bool m_owner{false};
};

struct CudaMemoryAllocation
{
  CudaMemoryAllocation(void *externalDeviceMemory,
      size_t bytes,
      ANARIMemoryDeleter deleter = nullptr,
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

  void privatize(); // if !isOwner(), then make internal copy

  // Movable, not copyable //
  CudaMemoryAllocation(CudaMemoryAllocation &&);
  CudaMemoryAllocation &operator=(CudaMemoryAllocation &&);
  CudaMemoryAllocation(const CudaMemoryAllocation &) = delete;
  CudaMemoryAllocation &operator=(const CudaMemoryAllocation &) = delete;
  ///////////////////////////

 private:
  size_t m_bytes{0};
  void *m_ptr{nullptr};
  ANARIMemoryDeleter m_deleter{nullptr};
  const void *m_deleterPtr{nullptr};
  bool m_owner{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

// HostMemoryAllocation //

inline HostMemoryAllocation::HostMemoryAllocation(void *externalMemory,
    size_t bytes,
    ANARIMemoryDeleter deleter,
    const void *deleterPtr)
    : m_bytes(bytes),
      m_ptr(externalMemory),
      m_deleter(deleter),
      m_deleterPtr(deleterPtr),
      m_owner(false)
{}

inline HostMemoryAllocation::HostMemoryAllocation(size_t bytes)
    : m_bytes(bytes), m_owner(true)
{
  if (bytes > 0)
    m_ptr = std::malloc(bytes);
}

inline HostMemoryAllocation::~HostMemoryAllocation()
{
  if (isValid()) {
    if (isOwner())
      std::free(m_ptr);
    else if (m_deleter)
      m_deleter(m_deleterPtr, m_ptr);
  }
}

inline void *HostMemoryAllocation::ptr() const
{
  return m_ptr;
}

inline size_t HostMemoryAllocation::bytes() const
{
  return m_bytes;
}

inline bool HostMemoryAllocation::isOwner() const
{
  return m_owner;
}

inline bool HostMemoryAllocation::isValid() const
{
  return ptr() != nullptr && bytes() > 0;
}

inline HostMemoryAllocation::operator bool() const
{
  return isValid();
}

inline HostMemoryAllocation &HostMemoryAllocation::operator=(
    const CudaMemoryAllocation &o)
{
  if (!isValid())
    *this = HostMemoryAllocation(o.bytes());
  if (bytes() != o.bytes()) {
    throw std::runtime_error(
        "cannot copy CudaMemoryAllocation to host: size mismatch");
  }
  cudaMemcpy(ptr(), o.ptr(), bytes(), cudaMemcpyDeviceToHost);
  return *this;
}

inline void HostMemoryAllocation::privatize()
{
  if (!isValid() || isOwner())
    return;
  const void *src = ptr();
  m_ptr = std::malloc(bytes());
  std::memcpy(m_ptr, src, bytes());
  m_owner = true;
  if (m_deleter)
    m_deleter(m_deleterPtr, src);
}

inline HostMemoryAllocation::HostMemoryAllocation(HostMemoryAllocation &&o)
{
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  if (o.isValid())
    o = HostMemoryAllocation();
}

inline HostMemoryAllocation &HostMemoryAllocation::operator=(
    HostMemoryAllocation &&o)
{
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  if (o.isValid())
    o = HostMemoryAllocation();
  return *this;
}

// CudaMemoryAllocation //

inline CudaMemoryAllocation::CudaMemoryAllocation(void *externalMemory,
    size_t bytes,
    ANARIMemoryDeleter deleter,
    const void *deleterPtr)
    : m_bytes(bytes),
      m_ptr(externalMemory),
      m_deleter(deleter),
      m_deleterPtr(deleterPtr),
      m_owner(false)
{}

inline CudaMemoryAllocation::CudaMemoryAllocation(size_t bytes)
    : m_bytes(bytes), m_owner(true)
{
  if (bytes > 0)
    cudaMalloc(&m_ptr, bytes);
}

inline CudaMemoryAllocation::~CudaMemoryAllocation()
{
  if (isValid()) {
    if (isOwner())
      cudaFree(m_ptr);
    else if (m_deleter)
      m_deleter(m_deleterPtr, m_ptr);
  }
}

inline void *CudaMemoryAllocation::ptr() const
{
  return m_ptr;
}

inline size_t CudaMemoryAllocation::bytes() const
{
  return m_bytes;
}

inline bool CudaMemoryAllocation::isOwner() const
{
  return m_owner;
}

inline bool CudaMemoryAllocation::isValid() const
{
  return ptr() != nullptr && bytes() > 0;
}

inline CudaMemoryAllocation::operator bool() const
{
  return isValid();
}

inline CudaMemoryAllocation &CudaMemoryAllocation::operator=(
    const HostMemoryAllocation &o)
{
  if (!isValid())
    *this = CudaMemoryAllocation(o.bytes());
  if (bytes() != o.bytes()) {
    throw std::runtime_error(
        "cannot copy HostMemoryAllocation to GPU: size mismatch");
  }
  cudaMemcpy(ptr(), o.ptr(), bytes(), cudaMemcpyHostToDevice);
  return *this;
}

inline void CudaMemoryAllocation::privatize()
{
  if (!isValid() || isOwner())
    return;
  const void *src = ptr();
  cudaMalloc(&m_ptr, bytes());
  cudaMemcpy(m_ptr, src, bytes(), cudaMemcpyDeviceToDevice);
  m_owner = true;
  if (m_deleter)
    m_deleter(m_deleterPtr, src);
}

inline CudaMemoryAllocation::CudaMemoryAllocation(CudaMemoryAllocation &&o)
{
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  if (o.isValid())
    o = CudaMemoryAllocation();
}

inline CudaMemoryAllocation &CudaMemoryAllocation::operator=(
    CudaMemoryAllocation &&o)
{
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  if (o.isValid())
    o = CudaMemoryAllocation();
  return *this;
}

} // namespace visrtx
