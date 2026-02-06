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

#include "array/ObjectArray.h"
#include "../Object.h"
// std
#include <algorithm>

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

static void refIncObject(Object *obj)
{
  if (obj)
    obj->refInc(helium::RefType::INTERNAL);
}

static void refDecObject(Object *obj)
{
  if (obj)
    obj->refDec(helium::RefType::INTERNAL);
}

// ObjectArray definitions ////////////////////////////////////////////////////

ObjectArray::ObjectArray(
    DeviceGlobalState *state, const Array1DMemoryDescriptor &d)
    : Array(ANARI_ARRAY1D, state, d, d.numItems), m_end(d.numItems)
{
  m_appHandles.resize(d.numItems, nullptr);
  updateInternalHandleArrays();
}

ObjectArray::~ObjectArray()
{
  std::for_each(m_appHandles.begin(), m_appHandles.end(), refDecObject);
}

void ObjectArray::commitParameters()
{
  const auto capacity = totalCapacity();
  m_begin = getParam<size_t>("begin", 0);
  m_begin = std::clamp(m_begin, size_t(0), capacity - 1);
  m_end = getParam<size_t>("end", capacity);
  m_end = std::clamp(m_end, size_t(1), capacity);

  if (size() == 0) {
    reportMessage(ANARI_SEVERITY_ERROR, "array size must be greater than zero");
    return;
  }

  if (m_begin > m_end) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "array 'begin' is not less than 'end', swapping values");
    std::swap(m_begin, m_end);
  }
}

void ObjectArray::finalize()
{
  markDataModified();
  notifyChangeObservers();
}

void ObjectArray::unmap()
{
  if (isMapped())
    updateInternalHandleArrays();
  Array::unmap();
}

size_t ObjectArray::totalSize() const
{
  return size();
}

size_t ObjectArray::size() const
{
  return m_end - m_begin;
}

const void *ObjectArray::dataGPU() const
{
  return m_GPUData.dataDevice();
}

Object **ObjectArray::handlesBegin(bool uploadData) const
{
  if (uploadData)
    uploadArrayData();
  return (Object **)(m_liveHandles.data() + m_begin);
}

Object **ObjectArray::handlesEnd(bool uploadData) const
{
  return handlesBegin(uploadData) + totalSize();
}

void ObjectArray::uploadArrayData() const
{
  if (!needToUploadData())
    return;

  m_GPUData.resize(totalSize());
  std::transform(handlesBegin(false),
      handlesEnd(false),
      m_GPUData.begin(),
      [](Object *obj) { return obj ? obj->deviceData() : nullptr; });
  m_GPUData.upload();

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
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::ObjectArray *);
