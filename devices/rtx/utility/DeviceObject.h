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

#include "Object.h"
#include "utility/DeviceBuffer.h"

namespace visrtx {

template <typename GPU_DATA_T>
struct DeviceObject
{
  using payload_t = GPU_DATA_T;

  DeviceObject();
  virtual ~DeviceObject() = default;

  void upload();

  payload_t &data();
  const payload_t &data() const;

  void *deviceData() const;

  size_t payloadBytes() const;

 private:
  payload_t m_hostData;
  DeviceBuffer m_deviceData;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename GPU_DATA_T>
inline DeviceObject<GPU_DATA_T>::DeviceObject()
{
  m_deviceData.reserve(sizeof(GPU_DATA_T));
}

template <typename GPU_DATA_T>
inline void DeviceObject<GPU_DATA_T>::upload()
{
  m_deviceData.upload(&m_hostData);
}

template <typename GPU_DATA_T>
inline GPU_DATA_T &DeviceObject<GPU_DATA_T>::data()
{
  return m_hostData;
}

template <typename GPU_DATA_T>
inline const GPU_DATA_T &DeviceObject<GPU_DATA_T>::data() const
{
  return m_hostData;
}

template <typename GPU_DATA_T>
inline void *DeviceObject<GPU_DATA_T>::deviceData() const
{
  return m_deviceData.ptr();
}

template <typename GPU_DATA_T>
inline size_t DeviceObject<GPU_DATA_T>::payloadBytes() const
{
  return m_deviceData.bytes();
}

} // namespace visrtx
