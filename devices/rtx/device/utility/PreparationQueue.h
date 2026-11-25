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

// Owners of objects waiting for one stage of batched preparation (input
// uploads, deferred preparation results). Pointer is an owning pointer.
// collect() runs the stage as one transaction:
//
//   run(first)       Batch-wide work, including any per-object publication.
//                    first is the oldest queued object. May throw.
//   retire(object)   Per object, after run() returned. Must be noexcept.
//   unwind()         Only if run() throws, while every owner is still queued.
//                    Runs during stack unwinding, so it must not throw.
//
// Membership is released only after every object is retired, so a throw
// leaves all owners queued for the next collection. Objects are released while
// collecting() is still true, so their destructors cannot start another
// collection. enqueue() is not allowed during a collection.
template <typename Pointer>
class PreparationQueue
{
 public:
  // Returns the object's position, stable until a collection retires it.
  size_t enqueue(Pointer object);
  bool empty() const;
  size_t size() const;
  bool collecting() const;
  template <typename Function>
  void forEach(Function function);
  // Returns false, doing nothing, if a collection is already running.
  template <typename Run, typename Retire, typename Unwind>
  bool collect(Run run, Retire retire, Unwind unwind);

 private:
  std::vector<Pointer> m_pending;
  bool m_collecting{false};
};

template <typename Pointer>
inline size_t PreparationQueue<Pointer>::enqueue(Pointer object)
{
  assert(!m_collecting);
  m_pending.push_back(std::move(object));
  return m_pending.size() - 1;
}

template <typename Pointer>
inline bool PreparationQueue<Pointer>::empty() const
{
  return m_pending.empty();
}

template <typename Pointer>
inline size_t PreparationQueue<Pointer>::size() const
{
  return m_pending.size();
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
template <typename Run, typename Retire, typename Unwind>
inline bool PreparationQueue<Pointer>::collect(
    Run run, Retire retire, Unwind unwind)
{
  if (m_collecting)
    return false;
  if (m_pending.empty())
    return true;
  m_collecting = true;
  bool ran = false;
  // unwind() runs from this guard during stack unwinding, so a throw from it
  // terminates instead of replacing the original error.
  auto collecting = nonstd::make_scope_exit([&] {
    if (!ran)
      unwind();
    m_collecting = false;
  });
  run(m_pending.front().operator->());
  ran = true;
  for (auto &object : m_pending) {
    auto *pointer = object.operator->();
    static_assert(noexcept(retire(pointer)),
        "PreparationQueue retire() must be noexcept: a throw would leave "
        "some objects retired and others not");
    retire(pointer);
  }
  // Release owners only now, and keep the capacity for the next collection.
  auto retired = std::move(m_pending);
  retired.clear();
  m_pending.swap(retired);
  return true;
}

} // namespace visrtx
