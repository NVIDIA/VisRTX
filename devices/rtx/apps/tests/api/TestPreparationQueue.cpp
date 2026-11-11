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

#include <cstdio>
#include <functional>
#include <memory>
#include <new>
#include <vector>
#include "../../../device/utility/PreparationQueue.h"

namespace {
struct Item
{
  explicit Item(int &destroyed);
  ~Item();
  void publish();
  int &destroyed;
  bool queued{true};
  bool failPublication{false};
  int publications{0};
  std::function<void()> onDestroy;
};
Item::Item(int &counter) : destroyed(counter) {}
Item::~Item()
{
  ++destroyed;
  if (onDestroy)
    onDestroy();
}
void Item::publish()
{
  if (failPublication)
    throw std::bad_alloc();
  ++publications;
}
using Queue = visrtx::PreparationQueue<std::shared_ptr<Item>>;

auto retire = [](Item *item) noexcept { item->queued = false; };

// Batch work that throws before touching any object, like a descriptor
// reserve. Owners must survive, unwind must see them all, nothing is retired.
bool runFailure(bool retry)
{
  int destroyed = 0;
  {
    Queue queue;
    std::vector<Item *> deferred;
    for (int i = 0; i < 2; ++i) {
      auto item = std::make_shared<Item>(destroyed);
      deferred.push_back(item.get());
      queue.enqueue(std::move(item)); // The queue is the only owner.
    }
    bool threw = false;
    int unwound = 0;
    try {
      queue.collect([&](Item *) { throw std::bad_alloc(); },
          retire,
          [&] {
            if (queue.collecting())
              queue.forEach([&](Item *) { ++unwound; });
          });
    } catch (const std::bad_alloc &) {
      threw = true;
    }
    if (!threw || unwound != 2 || destroyed || queue.size() != 2
        || queue.collecting())
      return false;
    for (auto *item : deferred)
      if (!item->queued || item->publications)
        return false;
    if (retry) {
      bool refusedReentry = false;
      bool unwoundAgain = false;
      if (!queue.collect(
              [&](Item *) {
                refusedReentry = !queue.collect([](Item *) {}, retire, [] {});
                queue.forEach([](Item *item) { item->publish(); });
              },
              retire,
              [&] { unwoundAgain = true; })
          || !refusedReentry || unwoundAgain || !queue.empty()
          || queue.collecting() || destroyed != 2)
        return false;
    }
  }
  return destroyed == 2; // Also covers destruction after an unretried failure.
}

// A later object's publication throws after an earlier one published. Both
// stay queued and unretired; a retry publishes both and retires both.
bool publicationFailure()
{
  int destroyed = 0;
  Queue queue;
  auto first = std::make_shared<Item>(destroyed);
  auto second = std::make_shared<Item>(destroyed);
  second->failPublication = true;
  queue.enqueue(first);
  queue.enqueue(second);
  auto publishAll = [&](Item *) {
    queue.forEach([](Item *item) { item->publish(); });
  };
  bool threw = false;
  try {
    queue.collect(publishAll, retire, [] {});
  } catch (const std::bad_alloc &) {
    threw = true;
  }
  bool ok = threw && queue.size() == 2 && !queue.collecting() && first->queued
      && second->queued && first->publications == 1
      && second->publications == 0;
  second->failPublication = false;
  ok = queue.collect(publishAll, retire, [] {}) && ok;
  ok = queue.empty() && !first->queued && !second->queued
      && first->publications == 2 && second->publications == 1 && ok;
  ok = queue.collect(publishAll, retire, [] {}) && first->publications == 2
      && ok;
  first.reset();
  second.reset();
  return destroyed == 2 && ok;
}

// Retirement sees the object still queued. The last owner is released after
// membership is gone but while the queue is still collecting, so a destructor
// can neither see it nor collect again.
bool retirementReentry()
{
  int destroyed = 0;
  Queue queue;
  bool safe = false;
  auto item = std::make_shared<Item>(destroyed);
  Item *raw = item.get();
  item->onDestroy = [&] {
    int visible = 0;
    queue.forEach([&](Item *) { ++visible; });
    safe = queue.empty() && queue.collecting() && visible == 0
        && !queue.collect([](Item *) {}, retire, [] {});
  };
  bool retiredBeforeRelease = false;
  queue.enqueue(std::move(item));
  const bool collected = queue.collect([](Item *) {},
      [&](Item *object) noexcept {
        retiredBeforeRelease = object == raw && !destroyed && queue.size() == 1;
      },
      [] {});
  return collected && retiredBeforeRelease && safe && destroyed == 1
      && !queue.collecting();
}

// Positions index the queue in enqueue order and restart after a collection;
// run() receives the oldest object.
bool positionsRestartAfterCollection()
{
  int destroyed = 0;
  Queue queue;
  auto oldest = std::make_shared<Item>(destroyed);
  bool ok = queue.enqueue(oldest) == 0
      && queue.enqueue(std::make_shared<Item>(destroyed)) == 1;
  Item *first = nullptr;
  ok = queue.collect([&](Item *item) { first = item; }, retire, [] {}) && ok;
  ok = first == oldest.get() && ok;
  ok = queue.enqueue(std::make_shared<Item>(destroyed)) == 0 && ok;
  return ok;
}
} // namespace

int main()
{
  bool ok = runFailure(true);
  ok = runFailure(false) && ok;
  ok = publicationFailure() && ok;
  ok = retirementReentry() && ok;
  ok = positionsRestartAfterCollection() && ok;
  std::printf(
      "preparation queue exception ownership: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
