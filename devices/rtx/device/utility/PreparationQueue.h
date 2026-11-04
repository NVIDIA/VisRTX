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

#pragma once

#include <cassert>
#include <nonstd/scope.hpp>
#include <utility>
#include <vector>

namespace visrtx {

// Serialized preparation ownership and publication. Pointer is an owning
// pointer; its object provides completePreparation() (retryable publication)
// and noexcept markPreparationCollected(). Submission cannot append while a
// collection is running. Reentrant collection is refused.
template <typename Pointer>
struct PreparationQueue
{
  void enqueue(Pointer object);
  bool empty() const;
  bool collecting() const;
  template <typename Function>
  void forEach(Function function);
  template <typename Finish>
  bool collect(Finish finish);

 private:
  std::vector<Pointer> m_pending;
  bool m_collecting{false};
};

template <typename Pointer>
inline void PreparationQueue<Pointer>::enqueue(Pointer object)
{
  assert(!m_collecting);
  m_pending.push_back(std::move(object));
}

template <typename Pointer>
inline bool PreparationQueue<Pointer>::empty() const
{
  return m_pending.empty();
}

template <typename Pointer>
inline bool PreparationQueue<Pointer>::collecting() const
{
  return m_collecting;
}

template <typename Pointer>
template <typename Function>
inline void PreparationQueue<Pointer>::forEach(Function function)
{
  for (auto &object : m_pending)
    function(object.operator->());
}

template <typename Pointer>
template <typename Finish>
inline bool PreparationQueue<Pointer>::collect(Finish finish)
{
  if (m_collecting)
    return false;
  if (m_pending.empty())
    return true;
  m_collecting = true;
  auto collecting = nonstd::make_scope_exit([&] { m_collecting = false; });
  // Keep owners here through both potentially throwing stages. If descriptor
  // allocation or registry publication throws, raw jobs and queued flags still
  // have owners, and another collection can retry without reallocating a queue.
  finish(m_pending.front().operator->());
  for (auto &object : m_pending)
    object->completePreparation();
  // Only retire queue membership after every publication succeeded. Earlier
  // objects must remain queued if a later object's publication throws.
  for (auto &object : m_pending) {
    auto *pointer = object.operator->();
    static_assert(noexcept(pointer->markPreparationCollected()));
    pointer->markPreparationCollected();
  }
  // Remove membership before destructors/deleters can reenter the owner.
  // Unlike the old pre-finish move, this happens only after GPU completion,
  // successful publication and non-throwing flag retirement.
  auto retired = std::move(m_pending);
  retired.clear();
  m_pending.swap(retired); // Reuse capacity without allocating on retry.
  return true;
}

} // namespace visrtx
