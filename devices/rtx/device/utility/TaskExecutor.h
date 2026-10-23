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

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace visrtx {

// One dedicated serial thread and a fixed-size worker pool. Worker tasks can
// invokeSerial() for thread-confined bookkeeping without blocking other
// workers. Caller thread-local state is not propagated to either execution
// lane.
//
// Tasks must not synchronously wait on work that needs their occupied lane:
// in particular, a serial task must not wait for a worker that calls back into
// the serial lane, and pool tasks must not exhaust the pool waiting for child
// tasks. Parallel completion order is unspecified.
//
// The owner must quiesce external callers before stop()/destruction, while
// keeping task dependencies alive. stop() runs on an external thread, drains
// workers first (their serial calls still run), then drains the serial thread.
// Calls after stop() returns execute inline for single-threaded cleanup. This
// does not permit concurrent stop() calls or external submissions during stop.
class TaskExecutor
{
 public:
  // Requires workerCount >= 1; throws std::invalid_argument for zero.
  explicit TaskExecutor(std::size_t workerCount);
  ~TaskExecutor();

  TaskExecutor(const TaskExecutor &) = delete;
  TaskExecutor &operator=(const TaskExecutor &) = delete;

  // Configured pool size while running, zero after stop(). No concurrent stop.
  std::size_t numWorkers() const;

  // Execute f on the serial thread and wait for its result (or exception).
  // Reentrant calls from that thread execute inline, avoiding self-deadlock.
  // After stop() returns, executes inline on the calling thread.
  template <typename F>
  std::invoke_result_t<F> invokeSerial(F &&f);

  // Run f on a pool worker and return a future carrying its result or
  // exception. After stop() returns, executes inline and returns a ready
  // future.
  template <typename F>
  std::future<std::invoke_result_t<F>> submit(F &&f);

  // Drain workers, then the serial thread. Idempotent; external caller only.
  void stop();

 private:
  void serialMain();
  void workerMain();

  // Serial lane (invokeSerial()).
  std::mutex m_mutex;
  std::condition_variable m_cv;
  std::deque<std::function<void()>> m_queue;
  bool m_stopping{false};
  std::thread::id m_threadId;
  std::thread m_thread;

  // Worker pool: parallel compilation (submit()).
  std::mutex m_workerMutex;
  std::condition_variable m_workerCv;
  std::deque<std::function<void()>> m_workerQueue;
  bool m_workersStopping{false};
  std::vector<std::thread> m_workers;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline std::size_t TaskExecutor::numWorkers() const
{
  return m_workers.size();
}

template <typename F>
std::invoke_result_t<F> TaskExecutor::invokeSerial(F &&f)
{
  // m_threadId is set in the constructor and never rewritten, so this read
  // needs no lock.
  if (std::this_thread::get_id() == m_threadId)
    return f();

  using R = std::invoke_result_t<F>;
  std::packaged_task<R()> task(std::forward<F>(f));
  std::future<R> result = task.get_future();
  bool queued = false;
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_stopping) {
      // task outlives the queued call: this thread blocks on result.get()
      // below until the serial thread has finished running it.
      m_queue.emplace_back([&task] { task(); });
      queued = true;
    }
  }
  if (!queued) {
    // Stopped: single-threaded cleanup runs inline (lock released).
    task();
    return result.get();
  }
  m_cv.notify_one();
  return result.get();
}

template <typename F>
std::future<std::invoke_result_t<F>> TaskExecutor::submit(F &&f)
{
  using R = std::invoke_result_t<F>;
  auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
  std::future<R> result = task->get_future();
  bool queued = false;
  {
    std::lock_guard<std::mutex> guard(m_workerMutex);
    if (!m_workersStopping) {
      m_workerQueue.emplace_back([task] { (*task)(); });
      queued = true;
    }
  }
  if (!queued) {
    // Stopped: no worker will pick it up, so run inline.
    (*task)();
    return result;
  }
  m_workerCv.notify_one();
  return result;
}

} // namespace visrtx
