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

#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <type_traits>
#include <functional>
#include <vector>

#include <iostream>

class queue_thread
{
  std::vector<std::packaged_task<void()>> tasks;
  bool stop;
  int next;
  int last;

  std::mutex mutex;
  std::condition_variable condition;
  std::thread thread;

  static void thread_fun(queue_thread *ct)
  {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(ct->mutex);
        ct->condition.wait(
            lock, [ct] { return ct->stop || ct->next != ct->last; });
        if (ct->stop && ct->next == ct->last) {
          break;
        }
        task = std::move(ct->tasks[ct->next]);
        ct->next = (ct->next + 1) % ct->tasks.size();
        ct->condition.notify_one();
      }
      task();
    }
  }

 public:
  queue_thread(size_t n)
      : tasks(n), stop(false), next(0), last(0), thread(thread_fun, this)
  {}

  template <class F, class... Args>
  std::future<void> enqueue(F &&f, Args &&... args)
  {
    std::packaged_task<void()> task(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<void> future = task.get_future();

    if (std::this_thread::get_id() != thread.get_id()) {
      std::unique_lock<std::mutex> lock(mutex);
      condition.wait(lock, [&] { return (last + 1) % tasks.size() != next; });
      tasks[last] = std::move(task);
      last = (last + 1) % tasks.size();
      condition.notify_one();
    } else {
      task();
    }

    return future;
  }

  ~queue_thread()
  {
    {
      std::unique_lock<std::mutex> lock(mutex);
      stop = true;
      condition.notify_all();
    }
    thread.join();
  }
};
