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

#include "array/Array.h"
// anari
#include "anari/detail/Helpers.h"
#include "anari/type_utility.h"

namespace visrtx {

// Helper functions //

template <typename T>
static void zeroOutStruct(T &v)
{
  std::memset(&v, 0, sizeof(T));
}

// Array //

static size_t s_numArrays = 0;

size_t Array::objectCount()
{
  return s_numArrays;
}

Array::Array(void *appMem,
    ANARIMemoryDeleter deleter,
    void *deleterPtr,
    ANARIDataType elementType)
    : m_elementType(elementType)
{
  s_numArrays++;

  if (appMem) {
    m_ownership =
        deleter ? ArrayDataOwnership::CAPTURED : ArrayDataOwnership::SHARED;
    m_lastModified = newTimeStamp();
  } else
    m_ownership = ArrayDataOwnership::MANAGED;

  switch (ownership()) {
  case ArrayDataOwnership::SHARED:
    m_hostData.shared.mem = appMem;
    break;
  case ArrayDataOwnership::CAPTURED:
    m_hostData.captured.mem = appMem;
    m_hostData.captured.deleter = deleter;
    m_hostData.captured.deleterPtr = deleterPtr;
    break;
  default:
    break;
  }
}

Array::~Array()
{
  freeAppMemory();
  s_numArrays--;
}

ANARIDataType Array::elementType() const
{
  return m_elementType;
}

ArrayDataOwnership Array::ownership() const
{
  return m_ownership;
}

void *Array::hostData() const
{
  switch (ownership()) {
  case ArrayDataOwnership::SHARED:
    return wasPrivatized() ? m_hostData.privatized.mem : m_hostData.shared.mem;
    break;
  case ArrayDataOwnership::CAPTURED:
    return m_hostData.captured.mem;
    break;
  case ArrayDataOwnership::MANAGED:
    return m_hostData.managed.mem;
    break;
  default:
    break;
  }

  return nullptr;
}

void *Array::deviceData() const
{
  m_usedOnDevice = true;
  uploadArrayData();
  return m_deviceData.buffer.ptr();
}

bool Array::wasPrivatized() const
{
  return m_privatized;
}

void *Array::map()
{
  if (m_mapped) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "array mapped again without being previously unmapped");
  }
  m_mapped = true;
  return hostData();
}

void Array::unmap()
{
  if (!m_mapped) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "array unmapped again without being previously mapped");
    return;
  }
  m_mapped = false;
  if (m_deviceData.buffer) {
    auto &state = *deviceState();
    state.uploadBuffer.addArray(this);
  }
  m_lastModified = newTimeStamp();
  notifyCommitObservers();
}

bool Array::dataModified() const
{
  return m_lastModified > m_lastUploaded;
}

void Array::uploadArrayData() const
{
  if (!m_usedOnDevice || (m_deviceData.buffer && !dataModified()))
    return;
  m_deviceData.buffer.upload((uint8_t *)hostData(),
      anari::sizeOfDataType(elementType()) * totalSize());
  m_lastUploaded = newTimeStamp();
}

void Array::addCommitObserver(Object *obj)
{
  m_observers.push_back(obj);
}

void Array::removeCommitObserver(Object *obj)
{
  m_observers.erase(std::remove_if(m_observers.begin(),
                        m_observers.end(),
                        [&](Object *o) -> bool { return o == obj; }),
      m_observers.end());
}

void Array::makePrivatizedCopy(size_t numElements)
{
  if (ownership() != ArrayDataOwnership::SHARED)
    return;

  reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
      "making private copy of shared array data (type '%s')",
      anari::anari_enum_to_string(elementType()));

  size_t numBytes = numElements * anari::sizeOfDataType(elementType());
  m_hostData.privatized.mem = malloc(numBytes);
  std::memcpy(m_hostData.privatized.mem, m_hostData.shared.mem, numBytes);

  m_privatized = true;
  zeroOutStruct(m_hostData.shared);
}

void Array::freeAppMemory()
{
  if (ownership() == ArrayDataOwnership::CAPTURED) {
    auto &captured = m_hostData.captured;
    reportMessage(ANARI_SEVERITY_DEBUG, "invoking array deleter");
    if (captured.deleter)
      captured.deleter(captured.deleterPtr, captured.mem);
    zeroOutStruct(captured);
  } else if (ownership() == ArrayDataOwnership::MANAGED) {
    reportMessage(ANARI_SEVERITY_DEBUG, "freeing managed array");
    free(m_hostData.managed.mem);
  } else if (wasPrivatized()) {
    free(m_hostData.privatized.mem);
    zeroOutStruct(m_hostData.privatized);
  }
}

void Array::initManagedMemory()
{
  if (m_hostData.managed.mem != nullptr)
    return;

  if (ownership() == ArrayDataOwnership::MANAGED) {
    auto totalBytes = totalSize() * anari::sizeOfDataType(elementType());
    m_hostData.managed.mem = malloc(totalBytes);
    std::memset(hostData(), 0, totalBytes);
  }
}

void Array::notifyCommitObservers() const
{
  auto &state = *deviceState();
  for (auto &o : m_observers) {
    o->markUpdated();
    state.commitBuffer.addObject(o);
  }
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Array *);
