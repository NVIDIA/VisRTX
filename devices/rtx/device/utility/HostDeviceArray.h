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

#include "utility/DeviceBuffer.h"
#include "utility/Span.h"
// std
#include <vector>

namespace visrtx {

template <typename T>
struct HostDeviceArray
{
  HostDeviceArray() = default;
  ~HostDeviceArray() = default;

  void resize(size_t size, bool reserveDeviceMem = true);
  void clear();

  size_t size() const;
  bool empty() const;

  void upload(size_t startIndex, size_t endIndex);
  void download(size_t startIndex, size_t endIndex);

  void upload();
  void download();

  Span<T> hostSpan() const;
  Span<T> deviceSpan() const;

  T *dataHost();
  T *dataDevice();

  const T *dataHost() const;
  const T *dataDevice() const;

  T *begin(); // host
  T *end(); // host

  const T *begin() const; // host
  const T *cbegin() const; // host
  const T *end() const; // host
  const T *cend() const; // host

 private:
  std::vector<T> m_hostArray;
  DeviceBuffer m_deviceBuffer;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline void HostDeviceArray<T>::resize(size_t size, bool reserveDeviceMem)
{
  if (size == 0)
    clear();
  else {
    m_hostArray.resize(size);
    if (reserveDeviceMem)
      m_deviceBuffer.reserve(size * sizeof(T));
  }
}

template <typename T>
inline void HostDeviceArray<T>::clear()
{
  m_hostArray.clear();
  m_deviceBuffer.reset();
}

template <typename T>
inline size_t HostDeviceArray<T>::size() const
{
  return m_hostArray.size();
}

template <typename T>
inline bool HostDeviceArray<T>::empty() const
{
  return m_hostArray.empty();
}

template <typename T>
inline void HostDeviceArray<T>::upload(size_t startIndex, size_t endIndex)
{
  if (endIndex > startIndex) {
    m_deviceBuffer.upload(&m_hostArray[startIndex],
        endIndex - startIndex,
        startIndex * sizeof(T));
  }
}

template <typename T>
inline void HostDeviceArray<T>::download(size_t startIndex, size_t endIndex)
{
  if (endIndex > startIndex) {
    m_deviceBuffer.download(&m_hostArray[startIndex],
        endIndex - startIndex,
        startIndex * sizeof(T));
  }
}

template <typename T>
inline void HostDeviceArray<T>::upload()
{
  upload(0, size());
}

template <typename T>
inline void HostDeviceArray<T>::download()
{
  download(0, size());
}

template <typename T>
inline Span<T> HostDeviceArray<T>::hostSpan() const
{
  return make_Span(dataHost(), size());
}

template <typename T>
inline Span<T> HostDeviceArray<T>::deviceSpan() const
{
  return make_Span(dataDevice(), size());
}

template <typename T>
inline T *HostDeviceArray<T>::dataHost()
{
  return m_hostArray.data();
}

template <typename T>
inline T *HostDeviceArray<T>::dataDevice()
{
  return (T *)m_deviceBuffer.ptr();
}

template <typename T>
inline const T *HostDeviceArray<T>::dataHost() const
{
  return m_hostArray.data();
}

template <typename T>
inline const T *HostDeviceArray<T>::dataDevice() const
{
  return (const T *)m_deviceBuffer.ptr();
}

template <typename T>
inline T *HostDeviceArray<T>::begin()
{
  return dataHost();
}

template <typename T>
inline T *HostDeviceArray<T>::end()
{
  return begin() + size();
}

template <typename T>
inline const T *HostDeviceArray<T>::begin() const
{
  return dataHost();
}

template <typename T>
inline const T *HostDeviceArray<T>::cbegin() const
{
  return begin();
}

template <typename T>
inline const T *HostDeviceArray<T>::end() const
{
  return begin() + size();
}

template <typename T>
inline const T *HostDeviceArray<T>::cend() const
{
  return end();
}

} // namespace visrtx
