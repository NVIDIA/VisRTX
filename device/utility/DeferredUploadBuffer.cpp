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

#include "DeferredUploadBuffer.h"
#include "array/Array.h"

namespace visrtx {

DeferredUploadBuffer::DeferredUploadBuffer()
{
  m_arraysToUpload.reserve(100);
}

DeferredUploadBuffer::~DeferredUploadBuffer()
{
  clear();
}

void DeferredUploadBuffer::addArray(Array *arr)
{
  arr->refInc(anari::RefType::INTERNAL);
  m_arraysToUpload.push_back(arr);
}

bool DeferredUploadBuffer::flush()
{
  if (m_arraysToUpload.empty())
    return false;

  for (auto arr : m_arraysToUpload) {
    if (arr->useCount() > 1)
      arr->uploadArrayData();
  }

  clear();
  return true;
}

void DeferredUploadBuffer::clear()
{
  for (auto &arr : m_arraysToUpload)
    arr->refDec(anari::RefType::INTERNAL);
  m_arraysToUpload.clear();
}

bool DeferredUploadBuffer::empty() const
{
  return m_arraysToUpload.empty();
}

} // namespace visrtx