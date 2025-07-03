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

#include "UploadableArray.h"
#include "utility/DeviceBuffer.h"
// helium
#include <helium/array/Array.h>

namespace visrtx {

using ArrayDataOwnership = helium::ArrayDataOwnership;
using ArrayMemoryDescriptor = helium::ArrayMemoryDescriptor;

// clang-format off
enum class AddressSpace
{
  HOST,    // host memory
  GPU,     // CUDA memory on VisRTX's GPU device
  FOREIGN, // CUDA memory on GPU device different from VisRTX's
  MANAGED, // CUDA managed memory
  UNKNOWN  // unused, likely an error state
};
// clang-format on

AddressSpace getPointerAddressSpace(const void *ptr);

struct Array : public UploadableArray
{
  Array(ANARIDataType arrayType,
      DeviceGlobalState *s,
      const ArrayMemoryDescriptor &d,
      size_t totalNumElements);
  ~Array() override;

  // helium::BaseObject interface //

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;
  void commitParameters() override;
  void finalize() override;

  // helium::BaseArray interface //

  void *map() override;
  void unmap() override;
  void privatize() override;

  // UploadableArray interface //

  void uploadArrayData() const override;

  // VisRTX interface //

  ANARIDataType elementType() const;
  ArrayDataOwnership ownership() const;
  size_t totalCapacity() const;
  virtual size_t totalSize() const;

  const void *data(AddressSpace as = AddressSpace::HOST) const;
  template <typename T>
  const T *dataAs(AddressSpace as = AddressSpace::HOST) const;

  bool isMapped() const;

  virtual const void *dataGPU() const;
  virtual const void *dataHost() const;

  void evictGPU();

 protected:
  template <typename T>
  void throwIfDifferentElementType() const;

  mutable cudaArray_t m_cuArrayFloat{};
  size_t m_arrayRefCountFloat{0};
  mutable cudaArray_t m_cuArrayUint8{};
  size_t m_arrayRefCountUint8{0};

 private:
  void on_NoInternalReferences() override;
  void on_NoPublicReferences() override;

  struct AnariArrayData
  {
    struct Shared
    {
      HostMemoryAllocation host;
      mutable CudaMemoryAllocation device;
    } shared;

    struct Captured
    {
      HostMemoryAllocation host;
      mutable CudaMemoryAllocation device;
      ANARIMemoryDeleter deleter{nullptr};
      const void *deleterPtr{nullptr};
    } captured;

    struct Managed
    {
      HostMemoryAllocation host;
      mutable CudaMemoryAllocation device;
    } managed;
  } m_data;

  ArrayDataOwnership m_ownership{ArrayDataOwnership::INVALID};
  ANARIDataType m_elementType{ANARI_UNKNOWN};
  bool m_mapped{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline const T *Array::dataAs(AddressSpace as) const
{
  throwIfDifferentElementType<T>();
  return (const T *)data(as);
}

template <typename T>
inline void Array::throwIfDifferentElementType() const
{
  constexpr auto t = anari::ANARITypeFor<T>::value;
  static_assert(
      t != ANARI_UNKNOWN, "unknown type used to query array element type");

  if (t != elementType()) {
    std::stringstream msg;
    msg << "incorrect element type queried for array -- asked for '"
        << anari::toString(t) << "', but array stores '"
        << anari::toString(elementType()) << "'";
    throw std::runtime_error(msg.str());
  }
}

} // namespace visrtx
