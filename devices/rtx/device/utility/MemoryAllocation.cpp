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

#include "MemoryAllocation.h"
// cuda
#include <cuda_runtime.h>
// std
#include <cstdlib>

namespace visrtx {

// HostMemoryAllocation definitions ///////////////////////////////////////////

HostMemoryAllocation::HostMemoryAllocation(void *externalMemory,
    size_t bytes,
    anari::MemoryDeleter deleter,
    const void *deleterPtr)
    : m_bytes(bytes),
      m_ptr(externalMemory),
      m_deleter(deleter),
      m_deleterPtr(deleterPtr),
      m_owner(false)
{}

HostMemoryAllocation::HostMemoryAllocation(size_t bytes)
    : m_bytes(bytes), m_owner(true)
{
  if (bytes > 0)
    m_ptr = std::malloc(bytes);
}

HostMemoryAllocation::~HostMemoryAllocation()
{
  freeMemory();
}

void *HostMemoryAllocation::ptr() const
{
  return m_ptr;
}

size_t HostMemoryAllocation::bytes() const
{
  return m_bytes;
}

bool HostMemoryAllocation::isOwner() const
{
  return m_owner;
}

bool HostMemoryAllocation::isValid() const
{
  return (ptr() != nullptr) && (bytes() > 0);
}

HostMemoryAllocation::operator bool() const
{
  return isValid();
}

HostMemoryAllocation &HostMemoryAllocation::operator=(
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

HostMemoryAllocation::HostMemoryAllocation(const CudaMemoryAllocation &o) {
  (*this) = o;
}

void HostMemoryAllocation::privatize()
{
  if (!isValid() || isOwner())
    return;
  void *dst = std::malloc(bytes());
  std::memcpy(dst, ptr(), bytes());
  freeMemory();
  m_ptr = dst;
  m_owner = true;
}

HostMemoryAllocation::HostMemoryAllocation(HostMemoryAllocation &&o)
{
  freeMemory();
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  o.clearPointers();
}

HostMemoryAllocation &HostMemoryAllocation::operator=(HostMemoryAllocation &&o)
{
  freeMemory();
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  o.clearPointers();
  return *this;
}

void HostMemoryAllocation::freeMemory()
{
  if (isValid()) {
    if (isOwner())
      std::free(m_ptr);
    else if (m_deleter)
      m_deleter(m_deleterPtr, m_ptr);
  }
  clearPointers();
}

void HostMemoryAllocation::clearPointers()
{
  m_ptr = nullptr;
  m_deleter = nullptr;
  m_deleterPtr = nullptr;
}

// CudaMemoryAllocation definitions ///////////////////////////////////////////

CudaMemoryAllocation::CudaMemoryAllocation(void *externalMemory,
    size_t bytes,
    anari::MemoryDeleter deleter,
    const void *deleterPtr)
    : m_bytes(bytes),
      m_ptr(externalMemory),
      m_deleter(deleter),
      m_deleterPtr(deleterPtr),
      m_owner(false)
{}

CudaMemoryAllocation::CudaMemoryAllocation(size_t bytes)
    : m_bytes(bytes), m_owner(true)
{
  if (bytes > 0)
    cudaMalloc(&m_ptr, bytes);
}

CudaMemoryAllocation::~CudaMemoryAllocation()
{
  freeMemory();
}

void *CudaMemoryAllocation::ptr() const
{
  return m_ptr;
}

size_t CudaMemoryAllocation::bytes() const
{
  return m_bytes;
}

bool CudaMemoryAllocation::isOwner() const
{
  return m_owner;
}

bool CudaMemoryAllocation::isValid() const
{
  return ptr() != nullptr && bytes() > 0;
}

CudaMemoryAllocation::operator bool() const
{
  return isValid();
}

CudaMemoryAllocation &CudaMemoryAllocation::operator=(
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

CudaMemoryAllocation::CudaMemoryAllocation(const HostMemoryAllocation &o) {
  (*this) = o;
}

void CudaMemoryAllocation::privatize()
{
  if (!isValid() || isOwner())
    return;
  void *dst = nullptr;
  cudaMalloc(&dst, bytes());
  cudaMemcpy(dst, ptr(), bytes(), cudaMemcpyDeviceToDevice);
  freeMemory();
  m_ptr = dst;
  m_owner = true;
}

CudaMemoryAllocation::CudaMemoryAllocation(CudaMemoryAllocation &&o)
{
  freeMemory();
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  o.clearPointers();
}

CudaMemoryAllocation &CudaMemoryAllocation::operator=(CudaMemoryAllocation &&o)
{
  freeMemory();
  m_ptr = o.m_ptr;
  m_bytes = o.m_bytes;
  m_deleter = o.m_deleter;
  m_deleterPtr = o.m_deleterPtr;
  m_owner = o.m_owner;
  o.clearPointers();
  return *this;
}

void CudaMemoryAllocation::freeMemory()
{
  if (isValid()) {
    if (isOwner())
      cudaFree(m_ptr);
    else if (m_deleter)
      m_deleter(m_deleterPtr, m_ptr);
  }
  clearPointers();
}

void CudaMemoryAllocation::clearPointers()
{
  m_ptr = nullptr;
  m_deleter = nullptr;
  m_deleterPtr = nullptr;
}

} // namespace visrtx