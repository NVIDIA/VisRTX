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

#pragma once

#include "MemoryAllocation.h"
// std
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace visrtx {

struct DeviceBuffer
{
  DeviceBuffer() = default;
  ~DeviceBuffer();

  template <typename T>
  void upload(const T *src, size_t numElements = 1, size_t byteOffsetStart = 0);

  template <typename T>
  void upload(const std::vector<T> &v);

  template <typename T>
  void download(T *dst, size_t numElements = 1, size_t byteOffsetStart = 0);

  HostMemoryAllocation hostCopy() const { return m_mem; }

  template <typename T>
  T *ptrAs() const;
  void *ptr() const;
  size_t bytes() const;

  void reset();
  void reserve(size_t bytes);

  operator bool() const;

 private:
  template <typename T>
  size_t bytesof(size_t numElements);

  void alloc(size_t bytes);

  CudaMemoryAllocation m_mem;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline DeviceBuffer::~DeviceBuffer() = default;

template <typename T>
inline void DeviceBuffer::upload(
    const T *src, size_t numElements, size_t byteOffsetStart)
{
  static_assert(std::is_trivially_copyable<T>::value);

  if (numElements == 0)
    return;

  auto neededBytes = bytesof<T>(numElements) + byteOffsetStart;
  if (neededBytes > bytes())
    alloc(neededBytes);

  cudaMemcpy((uint8_t *)ptr() + byteOffsetStart,
      src,
      bytesof<T>(numElements),
      cudaMemcpyHostToDevice);
}

template <typename T>
inline void DeviceBuffer::upload(const std::vector<T> &v)
{
  upload(v.data(), v.size());
}

template <typename T>
inline void DeviceBuffer::download(
    T *dst, size_t numElements, size_t byteOffsetStart)
{
  static_assert(std::is_trivially_copyable<T>::value);

  if (numElements == 0)
    return;

  if (!ptr())
    throw std::runtime_error("downloading from empty DeviceBuffer");
  const auto requestedBytes = bytesof<T>(numElements);
  if ((requestedBytes + byteOffsetStart) > bytes())
    throw std::runtime_error("downloading too much data from DeviceBuffer");
  cudaMemcpy(dst,
      (uint8_t *)ptr() + byteOffsetStart,
      requestedBytes,
      cudaMemcpyDeviceToHost);
}

template <typename T>
inline T *DeviceBuffer::ptrAs() const
{
  return (T *)ptr();
}

inline void *DeviceBuffer::ptr() const
{
  return m_mem.ptr();
}

inline size_t DeviceBuffer::bytes() const
{
  return m_mem.bytes();
}

inline void DeviceBuffer::reset()
{
  m_mem = CudaMemoryAllocation();
}

inline void DeviceBuffer::reserve(size_t numBytes)
{
  if (numBytes > bytes())
    alloc(numBytes);
}

inline DeviceBuffer::operator bool() const
{
  return ptr() != nullptr;
}

template <typename T>
inline size_t DeviceBuffer::bytesof(size_t numElements)
{
  return sizeof(T) * numElements;
}

inline void DeviceBuffer::alloc(size_t bytes)
{
  m_mem = CudaMemoryAllocation(bytes);
}

} // namespace visrtx
