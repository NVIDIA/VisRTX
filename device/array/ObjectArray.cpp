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

#include "array/ObjectArray.h"
// anari
#include "anari/type_utility.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

static void refIncObject(Object *obj)
{
  if (obj)
    obj->refInc(anari::RefType::INTERNAL);
}

static void refDecObject(Object *obj)
{
  if (obj)
    obj->refDec(anari::RefType::INTERNAL);
}

// ObjectArray definitions ////////////////////////////////////////////////////

ObjectArray::ObjectArray(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *deleterPtr,
    ANARIDataType type,
    uint64_t numItems,
    uint64_t byteStride)
    : Array(appMemory, deleter, deleterPtr, type),
      m_capacity(numItems),
      m_end(numItems)
{
  if (byteStride != 0)
    throw std::runtime_error("strided arrays not yet supported!");
  m_appHandles.resize(numItems, nullptr);
  initManagedMemory();
  updateInternalHandleArrays();
  markDataModified();
}

ObjectArray::~ObjectArray()
{
  std::for_each(m_appHandles.begin(), m_appHandles.end(), refDecObject);
  std::for_each(
      m_appendedHandles.begin(), m_appendedHandles.end(), refDecObject);
}

void ObjectArray::commit()
{
  auto oldBegin = m_begin;
  auto oldEnd = m_end;

  m_begin = getParam<size_t>("begin", 0);
  m_begin = std::clamp(m_begin, size_t(0), m_capacity - 1);
  m_end = getParam<size_t>("end", m_capacity);
  m_end = std::clamp(m_end, size_t(1), m_capacity);

  if (size() == 0) {
    reportMessage(ANARI_SEVERITY_ERROR, "array size must be greater than zero");
    return;
  }

  if (m_begin > m_end) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "array 'begin' is not less than 'end', swapping values");
    std::swap(m_begin, m_end);
  }

  if (m_begin != oldBegin || m_end != oldEnd) {
    markDataModified();
    notifyCommitObservers();
  }
}

ArrayShape ObjectArray::shape() const
{
  return ArrayShape::ARRAY1D;
}

size_t ObjectArray::totalSize() const
{
  return size() + m_appendedHandles.size();
}

size_t ObjectArray::totalCapacity() const
{
  return m_capacity;
}

size_t ObjectArray::size() const
{
  return m_end - m_begin;
}

void ObjectArray::privatize()
{
  makePrivatizedCopy(size());
  freeAppMemory();
  if (data()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "ObjectArray privatized but host array still present");
  }
}

Object **ObjectArray::handlesBegin(bool uploadData) const
{
  if (uploadData)
    uploadArrayData();
  return m_liveHandles.data() + m_begin;
}

Object **ObjectArray::handlesEnd(bool uploadData) const
{
  return handlesBegin(uploadData) + totalSize();
}

void *ObjectArray::deviceData() const
{
  return thrust::raw_pointer_cast(m_GPUDataDevice.data());
}

void ObjectArray::appendHandle(Object *o)
{
  o->refInc(anari::RefType::INTERNAL);
  m_appendedHandles.push_back(o);
  markDataModified();
}

void ObjectArray::removeAppendedHandles()
{
  m_liveHandles.resize(size());
  for (auto o : m_appendedHandles)
    o->refDec(anari::RefType::INTERNAL);
  m_appendedHandles.clear();
  markDataModified();
}

void ObjectArray::uploadArrayData() const
{
  if (!needToUploadData())
    return;

  updateInternalHandleArrays();

  auto &state = *deviceState();

  const auto type = elementType();
  if (type == ANARI_INSTANCE)
    state.objectUpdates.lastTLASChange = newTimeStamp();
  else if (type == ANARI_SURFACE || type == ANARI_VOLUME)
    state.objectUpdates.lastBLASChange = newTimeStamp();

  m_GPUDataHost.resize(totalSize());

  std::transform(handlesBegin(false),
      handlesEnd(false),
      m_GPUDataHost.begin(),
      [](Object *obj) { return obj ? obj->deviceData() : nullptr; });

  m_GPUDataDevice = m_GPUDataHost;
  markDataUploaded();
}

void ObjectArray::updateInternalHandleArrays() const
{
  m_liveHandles.resize(totalSize());

  if (data()) {
    auto **srcAllBegin = (Object **)data();
    auto **srcAllEnd = srcAllBegin + totalCapacity();
    std::for_each(srcAllBegin, srcAllEnd, refIncObject);
    std::for_each(m_appHandles.begin(), m_appHandles.end(), refDecObject);
    std::copy(srcAllBegin, srcAllEnd, m_appHandles.data());

    auto **srcRegionBegin = srcAllBegin + m_begin;
    auto **srcRegionEnd = srcRegionBegin + size();
    std::copy(srcRegionBegin, srcRegionEnd, m_liveHandles.data());
  }

  std::copy(m_appendedHandles.begin(),
      m_appendedHandles.end(),
      m_liveHandles.begin() + size());
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::ObjectArray *);
