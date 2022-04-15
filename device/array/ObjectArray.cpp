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

ObjectArray::ObjectArray(void *appMemory,
    ANARIMemoryDeleter deleter,
    void *deleterPtr,
    ANARIDataType type,
    uint64_t numItems,
    uint64_t byteStride)
    : Array(appMemory, deleter, deleterPtr, type)
{
  if (byteStride != 0)
    throw std::runtime_error("strided arrays not yet supported!");

  m_deviceData.buffer.reserve(anari::sizeOf(type) * size());

  m_handleArray.resize(numItems, nullptr);

  initManagedMemory();

  updateInternalHandleArray();

  m_lastModified = newTimeStamp();
}

ObjectArray::~ObjectArray()
{
  removeAppendedHandles();
  for (auto *obj : m_handleArray) {
    if (obj)
      obj->refDec(anari::RefType::INTERNAL);
  }
}

ArrayShape ObjectArray::shape() const
{
  return ArrayShape::ARRAY1D;
}

size_t ObjectArray::totalSize() const
{
  return size();
}

size_t ObjectArray::size() const
{
  return m_handleArray.size();
}

void ObjectArray::privatize()
{
  freeAppMemory();
}

Object **ObjectArray::handles()
{
  uploadArrayData();
  return m_handleArray.data();
}

void *ObjectArray::deviceData() const
{
  return thrust::raw_pointer_cast(m_GPUDataDevice.data());
}

void ObjectArray::appendHandle(Object *o)
{
  m_handleArray.push_back(o);
  m_numAppended++;
}

void ObjectArray::removeAppendedHandles()
{
  auto originalSize = m_handleArray.size() - m_numAppended;
  m_handleArray.resize(originalSize);
  m_numAppended = 0;
}

void ObjectArray::uploadArrayData() const
{
  if (!dataModified())
    return;

  auto &state = *deviceState();

  const auto type = elementType();
  if (type == ANARI_INSTANCE)
    state.objectUpdates.lastTLASChange = newTimeStamp();
  else if (type == ANARI_SURFACE || type == ANARI_VOLUME)
    state.objectUpdates.lastBLASChange = newTimeStamp();

  auto begin = m_handleArray.begin();
  auto end = m_handleArray.end();

  m_GPUDataHost.resize(m_handleArray.size());

  std::transform(begin, end, m_GPUDataHost.begin(), [](Object *obj) {
    return obj->deviceData();
  });

  m_GPUDataDevice = m_GPUDataHost;

  m_lastUploaded = newTimeStamp();
}

void ObjectArray::updateInternalHandleArray()
{
  for (auto *obj : m_handleArray) {
    if (obj)
      obj->refDec(anari::RefType::INTERNAL);
  }

  auto **srcBegin = (Object **)data();
  auto **srcEnd = srcBegin + size();

  std::transform(srcBegin, srcEnd, m_handleArray.begin(), [](Object *obj) {
    obj->refInc(anari::RefType::INTERNAL);
    return obj;
  });
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::ObjectArray *);
