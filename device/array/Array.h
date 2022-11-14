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

#include "Object.h"
#include "utility/DeviceBuffer.h"
// helium
#include "helium/BaseArray.h"
// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
// anari
#include "anari/anari_cpp/Traits.h"

namespace visrtx {

enum class ArrayDataOwnership
{
  SHARED,
  CAPTURED,
  MANAGED,
  INVALID
};

enum class AddressSpace
{
  HOST,
  GPU
};

struct ArrayMemoryDescriptor
{
  const void *appMemory{nullptr};
  ANARIMemoryDeleter deleter{};
  const void *deleterPtr{nullptr};
  ANARIDataType elementType{ANARI_UNKNOWN};
};

struct Array : public helium::BaseArray
{
  static size_t objectCount();

  Array(ANARIDataType type,
      DeviceGlobalState *state,
      const ArrayMemoryDescriptor &d);
  virtual ~Array();

  ANARIDataType elementType() const;
  ArrayDataOwnership ownership() const;

  void *data(AddressSpace as = AddressSpace::HOST) const;
  virtual void *deviceData() const;

  template <typename T>
  T *dataAs(AddressSpace as = AddressSpace::HOST) const;

  virtual size_t totalSize() const = 0;
  virtual size_t totalCapacity() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;
  void commit() override;
  void *map() override;
  virtual void unmap() override;
  virtual void privatize() = 0;

  bool wasPrivatized() const;

  void markDataModified();
  virtual void uploadArrayData() const;
  void markDataUploaded() const;
  bool needToUploadData() const;

  // CONSOLIDATE INTO visrtx::Object //////////////////////////////////////////
  DeviceGlobalState *deviceState() const;
  /////////////////////////////////////////////////////////////////////////////

 protected:
  void makePrivatizedCopy(size_t numElements);
  void freeAppMemory();
  void initManagedMemory();

  void notifyObserver(BaseObject *) const override;

  struct ArrayDescriptor
  {
    struct SharedData
    {
      const void *mem{nullptr};
    } shared;

    struct CapturedData
    {
      const void *mem{nullptr};
      ANARIMemoryDeleter deleter{nullptr};
      const void *deleterPtr{nullptr};
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

  helium::TimeStamp m_lastDataModified{0};
  mutable helium::TimeStamp m_lastDataUploaded{0};
  bool m_mapped{false};

 private:
  ArrayDataOwnership m_ownership{ArrayDataOwnership::INVALID};
  ANARIDataType m_elementType{ANARI_UNKNOWN};
  bool m_privatized{false};
  mutable bool m_usedOnDevice{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline T *Array::dataAs(AddressSpace as) const
{
  if (anari::ANARITypeFor<T>::value != m_elementType)
    throw std::runtime_error("incorrect element type queried for array");

  return (T *)data(as);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Array *, ANARI_ARRAY);
