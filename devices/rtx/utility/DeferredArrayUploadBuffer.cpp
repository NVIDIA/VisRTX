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

#include "DeferredArrayUploadBuffer.h"

namespace visrtx {

DeferredArrayUploadBuffer::DeferredArrayUploadBuffer()
{
  m_arraysToUpload.reserve(100);
}

DeferredArrayUploadBuffer::~DeferredArrayUploadBuffer()
{
  clear();
}

void DeferredArrayUploadBuffer::addArray(helium::Array *arr)
{
  arr->refInc(helium::RefType::INTERNAL);
  m_arraysToUpload.push_back(arr);
}

bool DeferredArrayUploadBuffer::flush()
{
  if (m_arraysToUpload.empty())
    return false;

  for (auto arr : m_arraysToUpload) {
    if (arr->useCount() > 1)
      arr->uploadArrayData();
  }

  clear();
  m_lastFlush = helium::newTimeStamp();
  return true;
}

helium::TimeStamp DeferredArrayUploadBuffer::lastFlush() const
{
  return m_lastFlush;
}

void DeferredArrayUploadBuffer::clear()
{
  for (auto &arr : m_arraysToUpload)
    arr->refDec(helium::RefType::INTERNAL);
  m_arraysToUpload.clear();
  m_lastFlush = 0;
}

bool DeferredArrayUploadBuffer::empty() const
{
  return m_arraysToUpload.empty();
}

} // namespace visrtx