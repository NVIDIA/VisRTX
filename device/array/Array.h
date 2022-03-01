/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "utility/DeviceObject.h"
// std
#include <vector>
// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
// anari
#include "anari/anari_cpp/Traits.h"

namespace visrtx {

enum class ArrayShape
{
  ARRAY1D,
  ARRAY2D,
  ARRAY3D
};

enum class ArrayDataOwnership
{
  SHARED,
  CAPTURED,
  MANAGED,
  INVALID
};

struct Array : public Object
{
  static size_t objectCount();

  Array(void *appMemory,
      ANARIMemoryDeleter deleter,
      void *deleterPtr,
      ANARIDataType elementType);
  virtual ~Array();

  ANARIDataType elementType() const;
  ArrayDataOwnership ownership() const;

  void *hostData() const;
  void *deviceData() const override;

  template <typename T>
  T *hostDataAs() const;

  template <typename T>
  T *deviceDataAs() const;

  virtual ArrayShape shape() const = 0;

  virtual size_t totalSize() const = 0;

  virtual void privatize() = 0;
  bool wasPrivatized() const;

  void *map();
  void unmap();

  bool dataModified() const;
  virtual void uploadArrayData() const;

  void addCommitObserver(Object *obj);
  void removeCommitObserver(Object *obj);

 protected:
  void makePrivatizedCopy(size_t numElements);
  void freeAppMemory();
  void initManagedMemory();

  struct ArrayDescriptor
  {
    struct SharedData
    {
      void *mem{nullptr};
    } shared;

    struct CapturedData
    {
      void *mem{nullptr};
      ANARIMemoryDeleter deleter{nullptr};
      void *deleterPtr{nullptr};
    } captured;

    struct ManagedData
    {
      void *mem{nullptr};
    } managed;

    struct PrivatizedData
    {
      void *mem{nullptr};
    } privatized;
  } m_hostData;

  struct
  {
    mutable DeviceBuffer buffer;
  } m_deviceData;

  TimeStamp m_lastModified{0};
  mutable TimeStamp m_lastUploaded{0};

 private:
  void notifyCommitObservers() const;

  std::vector<Object *> m_observers;

  ArrayDataOwnership m_ownership{ArrayDataOwnership::INVALID};
  ANARIDataType m_elementType{ANARI_UNKNOWN};
  bool m_privatized{false};
  bool m_mapped{false};
  mutable bool m_usedOnDevice{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline T *Array::hostDataAs() const
{
  if (anari::ANARITypeFor<T>::value != m_elementType)
    throw std::runtime_error("incorrect element type queried for array");

  return (T *)hostData();
}

template <typename T>
inline T *Array::deviceDataAs() const
{
  if (anari::ANARITypeFor<T>::value != m_elementType)
    throw std::runtime_error("incorrect element type queried for array");

  return (T *)deviceData();
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Array *, ANARI_ARRAY);
