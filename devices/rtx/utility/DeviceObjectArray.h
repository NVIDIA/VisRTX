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

#include "../gpu/gpu_decl.h"
#include "utility/HostDeviceArray.h"
// std
#include <mutex>
#include <stack>

namespace visrtx {

template <typename T>
struct DeviceObjectArray
{
  DeviceObjectArray() = default;
  ~DeviceObjectArray() = default;

  DeviceObjectIndex alloc(void *hostObj);
  void free(DeviceObjectIndex idx);

  size_t capacity() const;
  size_t size() const;
  bool empty() const;

  T &map(DeviceObjectIndex idx);
  void unmap(DeviceObjectIndex idx);

  void *hostObject(DeviceObjectIndex idx) const;

  void upload();

  const T *devicePtr();

 private:
  bool m_fullReupload{true};
  HostDeviceArray<T> m_objects;
  std::vector<void *> m_hostObjects;

  using DOI = DeviceObjectIndex;
  std::stack<DOI, std::vector<DOI>> m_freeIndices;

  std::vector<DOI> m_objectsToUpload;
  std::mutex m_allocationMutex;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline DeviceObjectIndex DeviceObjectArray<T>::alloc(void *hostObj)
{
  DeviceObjectIndex i = -1;

  if (!hostObj)
    throw std::runtime_error("invalid DeviceObjectArray<> allocation");

  std::scoped_lock<std::mutex> lock(m_allocationMutex);

  if (!m_freeIndices.empty()) {
    i = m_freeIndices.top();
    m_hostObjects[i] = hostObj;
    m_freeIndices.pop();
  } else {
    i = static_cast<DeviceObjectIndex>(m_objects.size());
    m_objects.resize(m_objects.size() + 1, false);
    m_hostObjects.push_back(hostObj);
    m_fullReupload = true;
  }

  return i;
}

template <typename T>
inline void DeviceObjectArray<T>::free(DeviceObjectIndex idx)
{
  std::scoped_lock<std::mutex> lock(m_allocationMutex);
  m_freeIndices.push(idx);
  m_hostObjects[idx] = nullptr;
}

template <typename T>
inline size_t DeviceObjectArray<T>::capacity() const
{
  return m_objects.size();
}

template <typename T>
inline size_t DeviceObjectArray<T>::size() const
{
  return m_objects.size() - m_freeIndices.size();
}

template <typename T>
inline bool DeviceObjectArray<T>::empty() const
{
  return size() == 0;
}

template <typename T>
inline T &DeviceObjectArray<T>::map(DeviceObjectIndex idx)
{
  return *(m_objects.dataHost() + idx);
}

template <typename T>
inline void DeviceObjectArray<T>::unmap(DeviceObjectIndex idx)
{
  m_objectsToUpload.push_back(idx);
}

template <typename T>
inline void *DeviceObjectArray<T>::hostObject(DeviceObjectIndex idx) const
{
  return m_hostObjects[idx];
}

template <typename T>
inline void DeviceObjectArray<T>::upload()
{
  if (m_objectsToUpload.empty())
    return;

  if (m_fullReupload)
    m_objects.upload();
  else {
    for (auto i : m_objectsToUpload)
      m_objects.upload(i, i + 1);
  }

  m_objectsToUpload.clear();
  m_fullReupload = false;
}

template <typename T>
inline const T *DeviceObjectArray<T>::devicePtr()
{
  upload();
  return m_objects.dataDevice();
}

} // namespace visrtx