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

#include "Array.h"
// std
#include <cstring>

namespace visrtx {

AddressSpace getPointerAddressSpace(const void *ptr)
{
  if (!ptr)
    return AddressSpace::HOST;

  int currentCUDADevice = 0;
  cudaGetDevice(&currentCUDADevice);
  cudaPointerAttributes attr = {};
  cudaError_t error = cudaPointerGetAttributes(&attr, ptr);
  if (error != cudaSuccess)
    return AddressSpace::HOST;

  switch (attr.type) {
  case cudaMemoryTypeDevice:
    return attr.device == currentCUDADevice ? AddressSpace::GPU
                                            : AddressSpace::FOREIGN;
    break;
  case cudaMemoryTypeManaged:
    return AddressSpace::MANAGED;
    break;
  case cudaMemoryTypeHost:
  case cudaMemoryTypeUnregistered:
  default:
    return AddressSpace::HOST;
  }

  return AddressSpace::HOST;
}

// Array implementation ///////////////////////////////////////////////////////

Array::Array(ANARIDataType arrayType,
    DeviceGlobalState *s,
    const ArrayMemoryDescriptor &d,
    size_t totalNumElements)
    : UploadableArray(arrayType, s)
{
  const auto bytes = totalNumElements * anari::sizeOf(d.elementType);
  const auto as = getPointerAddressSpace(d.appMemory);

  if (anari::isObject(d.elementType) && as != AddressSpace::HOST) {
    throw std::runtime_error(
        "illegal operation: cannot create object arrays from GPU memory");
  }

  m_elementType = d.elementType;

  if (d.appMemory) {
    m_ownership =
        d.deleter ? ArrayDataOwnership::CAPTURED : ArrayDataOwnership::SHARED;
    markDataModified();
  } else
    m_ownership = ArrayDataOwnership::MANAGED;

  void *appMem = const_cast<void *>(d.appMemory);

  switch (ownership()) {
  case ArrayDataOwnership::SHARED:
    if (as == AddressSpace::HOST) {
      m_data.shared.host = HostMemoryAllocation(appMem, bytes);
    } else {
      m_data.shared.device = CudaMemoryAllocation(appMem, bytes);
    }
    break;
  case ArrayDataOwnership::CAPTURED:
    if (as == AddressSpace::HOST) {
      m_data.captured.host =
          HostMemoryAllocation(appMem, bytes, d.deleter, d.deleterPtr);
    } else {
      m_data.captured.device =
          CudaMemoryAllocation(appMem, bytes, d.deleter, d.deleterPtr);
    }
    m_data.captured.deleter = d.deleter;
    m_data.captured.deleterPtr = d.deleterPtr;
    break;
  case ArrayDataOwnership::MANAGED:
    m_data.managed.host = HostMemoryAllocation(bytes);
    std::memset(m_data.managed.host.ptr(), 0, bytes);
  default:
    break;
  }
}

Array::~Array() = default;

bool Array::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  return false;
}

void Array::commitParameters()
{
  // no-op
}

void Array::finalize()
{
  // no-op
}

void *Array::map()
{
  if (isMapped()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "array mapped again without being previously unmapped");
  }
  m_mapped = true;
  return const_cast<void *>(data());
}

void Array::unmap()
{
  if (!isMapped()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "array unmapped again without being previously mapped");
    return;
  }
  m_mapped = false;
  markDataModified();
  notifyChangeObservers();
}

void Array::privatize()
{
  if (m_data.shared.host.isValid() && !m_data.shared.host.isOwner()) {
    reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
        "making private copy of shared host array (type '%s') | refs: (%i:%i)",
        anari::toString(elementType()),
        this->useCount(helium::RefType::PUBLIC),
        this->useCount(helium::RefType::INTERNAL));
    m_data.shared.host.privatize();
  }
  if (m_data.shared.device.isValid() && !m_data.shared.device.isOwner()) {
    reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
        "making private copy of shared GPU array (type '%s') | refs: (%i:%i)",
        anari::toString(elementType()),
        this->useCount(helium::RefType::PUBLIC),
        this->useCount(helium::RefType::INTERNAL));
    m_data.shared.device.privatize();
  }
}

void Array::uploadArrayData() const
{
  if (!needToUploadData())
    return;

  if (m_data.shared.host.isValid())
    m_data.shared.device = m_data.shared.host;
  else if (m_data.captured.host.isValid())
    m_data.captured.device = m_data.captured.host;
  else if (m_data.managed.host.isValid())
    m_data.managed.device = m_data.managed.host;

  markDataUploaded();
}

ANARIDataType Array::elementType() const
{
  return m_elementType;
}

ArrayDataOwnership Array::ownership() const
{
  return m_ownership;
}

size_t Array::totalCapacity() const
{
  size_t bytes = 0;

  switch (ownership()) {
  case ArrayDataOwnership::SHARED:
    bytes = m_data.shared.device ? m_data.shared.device.bytes()
                                 : m_data.shared.host.bytes();
    break;
  case ArrayDataOwnership::CAPTURED:
    bytes = m_data.captured.device ? m_data.captured.device.bytes()
                                   : m_data.captured.host.bytes();
    break;
  case ArrayDataOwnership::MANAGED:
    bytes = m_data.managed.device ? m_data.managed.device.bytes()
                                  : m_data.managed.host.bytes();
    break;
  default:
    break;
  }

  return bytes / anari::sizeOf(elementType());
}

size_t Array::totalSize() const
{
  return totalCapacity();
}

const void *Array::data(AddressSpace as) const
{
  return as == AddressSpace::HOST ? dataHost() : dataGPU();
}

bool Array::isMapped() const
{
  return m_mapped;
}

const void *Array::dataGPU() const
{
  uploadArrayData();

  switch (ownership()) {
  case ArrayDataOwnership::SHARED:
    return m_data.shared.device.ptr();
  case ArrayDataOwnership::CAPTURED:
    return m_data.captured.device.ptr();
  case ArrayDataOwnership::MANAGED:
    return m_data.managed.device.ptr();
  default:
    break;
  }

  reportMessage(ANARI_SEVERITY_ERROR,
      "VisRTX::Array unable to return GPU data() pointer");

  return nullptr;
}

const void *Array::dataHost() const
{
  switch (ownership()) {
  case ArrayDataOwnership::SHARED:
    return m_data.shared.host.ptr();
  case ArrayDataOwnership::CAPTURED:
    return m_data.captured.host.ptr();
  case ArrayDataOwnership::MANAGED:
    return m_data.managed.host.ptr();
  default:
    break;
  }

  reportMessage(ANARI_SEVERITY_ERROR,
      "VisRTX::Array unable to return host data() pointer");

  return nullptr;
}

void Array::evictGPU()
{
  if (m_data.shared.host)
    m_data.shared.device = CudaMemoryAllocation();
  if (m_data.captured.host)
    m_data.captured.device = CudaMemoryAllocation();
  if (m_data.managed.host)
    m_data.managed.device = CudaMemoryAllocation();
  markDataModified();
}

void Array::on_NoInternalReferences()
{
  reportMessage(ANARI_SEVERITY_DEBUG,
      "evicting GPU memory for array with no internal references");
  evictGPU();
}

void Array::on_NoPublicReferences()
{
  privatize();
}

} // namespace visrtx
