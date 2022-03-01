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

#include "DeferredCommitBuffer.h"
#include "Object.h"
// std
#include <algorithm>

namespace visrtx {

DeferredCommitBuffer::DeferredCommitBuffer()
{
  m_commitBuffer.reserve(100);
}

DeferredCommitBuffer::~DeferredCommitBuffer()
{
  clear();
}

void DeferredCommitBuffer::addObject(Object *obj)
{
  obj->refInc(anari::RefType::INTERNAL);
  if (obj->commitPriority() != VISRTX_COMMIT_PRIORITY_DEFAULT)
    m_needToSortCommits = true;
  m_commitBuffer.push_back(obj);
}

bool DeferredCommitBuffer::flush()
{
  if (m_commitBuffer.empty())
    return false;

  if (m_needToSortCommits) {
    std::sort(m_commitBuffer.begin(),
        m_commitBuffer.end(),
        [](Object *o1, Object *o2) {
          return o1->commitPriority() < o2->commitPriority();
        });
  }

  m_needToSortCommits = false;

  for (auto obj : m_commitBuffer) {
    if (obj->lastUpdated() > obj->lastCommitted()) {
      obj->commit();
      obj->upload();
      obj->markCommitted();
    }
  }

  clear();
  return true;
}

void DeferredCommitBuffer::clear()
{
  for (auto &obj : m_commitBuffer)
    obj->refDec(anari::RefType::INTERNAL);
  m_commitBuffer.clear();
}

bool DeferredCommitBuffer::empty() const
{
  return m_commitBuffer.empty();
}

} // namespace visrtx