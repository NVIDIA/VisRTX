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
#include "utility/DeviceObjectArray.h"

namespace visrtx {

template <typename GPU_DATA_T>
struct RegisteredObject : public Object
{
  using payload_t = GPU_DATA_T;
  using array_t = DeviceObjectArray<payload_t>;

  RegisteredObject(ANARIDataType type, DeviceGlobalState *d);
  virtual ~RegisteredObject();

  void finalize() override;
  void upload();

  DeviceObjectIndex index() const;

  void setRegistry(array_t &registry);

 protected:
  virtual payload_t gpuData() const = 0;

 private:
  DeviceObjectIndex m_index{-1};
  array_t *m_registryArray{nullptr};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename GPU_DATA_T>
inline RegisteredObject<GPU_DATA_T>::RegisteredObject(
    ANARIDataType type, DeviceGlobalState *d)
    : Object(type, d)
{
  helium::BaseObject::markParameterChanged();
  d->commitBuffer.addObjectToCommit(this);
}

template <typename GPU_DATA_T>
inline RegisteredObject<GPU_DATA_T>::~RegisteredObject()
{
  m_registryArray->free(index());
}

template <typename GPU_DATA_T>
inline void RegisteredObject<GPU_DATA_T>::finalize()
{
  upload();
}

template <typename GPU_DATA_T>
inline void RegisteredObject<GPU_DATA_T>::upload()
{
  auto &data = m_registryArray->map(index());
  data = gpuData();
  m_registryArray->unmap(index());
}

template <typename GPU_DATA_T>
inline DeviceObjectIndex RegisteredObject<GPU_DATA_T>::index() const
{
  return m_index;
}

template <typename GPU_DATA_T>
inline void RegisteredObject<GPU_DATA_T>::setRegistry(array_t &a)
{
  m_registryArray = &a;
  if (m_index < 0)
    m_index = a.alloc(this);
}

} // namespace visrtx
