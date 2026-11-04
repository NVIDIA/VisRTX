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
  void completePreparation();
  void markPreparationCollected() noexcept;
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
void Item::completePreparation()
{
  if (failPublication)
    throw std::bad_alloc();
  ++publications;
}
void Item::markPreparationCollected() noexcept
{
  queued = false;
}
using Queue = visrtx::PreparationQueue<std::shared_ptr<Item>>;

bool finishFailure(bool retry)
{
  int destroyed = 0;
  {
    Queue queue;
    std::vector<Item *> deferred;
    for (int i = 0; i < 2; ++i) {
      auto item = std::make_shared<Item>(destroyed);
      deferred.push_back(item.get());
      queue.enqueue(
          std::move(item)); // Queue is the only owner, like an orphan geometry.
    }
    bool threw = false;
    try {
      queue.collect([&](Item *) {
        // SmallGeometryBatch's descriptor reserve can throw before consuming
        // the raw deferred-job list. Test the same ownership boundary directly,
        // without routing a host exception through ANARI's fatal C API handler.
        throw std::bad_alloc();
      });
    } catch (const std::bad_alloc &) {
      threw = true;
    }
    if (!threw || destroyed || queue.empty() || queue.collecting())
      return false; // Never dereference the raw jobs if ownership was lost.
    for (auto *item : deferred)
      if (!item->queued || item->publications)
        return false;
    if (retry) {
      bool refusedReentry = false;
      if (!queue.collect([&](Item *) {
            refusedReentry = !queue.collect([](Item *) {});
            for (auto *item : deferred)
              if (!item->queued)
                throw std::bad_alloc();
            deferred.clear(); // Deferred submission and completion succeeded.
          })
          || !refusedReentry || !queue.empty() || queue.collecting()
          || destroyed != 2)
        return false;
    }
  }
  return destroyed == 2; // Also covers destruction after an unretried failure.
}

bool retirementReentry()
{
  int destroyed = 0;
  Queue queue;
  bool safe = false;
  auto item = std::make_shared<Item>(destroyed);
  item->onDestroy = [&] {
    int visible = 0;
    queue.forEach([&](Item *) { ++visible; });
    safe = queue.empty() && queue.collecting() && visible == 0
        && !queue.collect([](Item *) {});
  };
  queue.enqueue(std::move(item));
  return queue.collect([](Item *) {}) && safe && destroyed == 1;
}

bool publicationFailure()
{
  int destroyed = 0;
  Queue queue;
  auto first = std::make_shared<Item>(destroyed);
  auto second = std::make_shared<Item>(destroyed);
  second->failPublication = true;
  queue.enqueue(first);
  queue.enqueue(second);
  int drains = 0;
  bool threw = false;
  try {
    queue.collect([&](Item *) { ++drains; });
  } catch (const std::bad_alloc &) {
    threw = true;
  }
  bool ok = threw && drains == 1 && !queue.empty() && !queue.collecting()
      && first->queued && second->queued && first->publications == 1
      && second->publications == 0;
  second->failPublication = false;
  ok = queue.collect([&](Item *) { ++drains; }) && ok;
  ok = drains == 2 && queue.empty() && !first->queued && !second->queued
      && first->publications == 2 && second->publications == 1 && ok;
  ok = queue.collect([&](Item *) { ++drains; }) && drains == 2 && ok;
  first.reset();
  second.reset();
  return destroyed == 2 && ok;
}
} // namespace

int main()
{
  bool ok = finishFailure(true);
  ok = finishFailure(false) && ok;
  ok = publicationFailure() && ok;
  ok = retirementReentry() && ok;
  std::printf(
      "preparation queue exception ownership: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
