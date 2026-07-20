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

namespace visrtx::mdl {

// The single thread that owns all MaterialRegistry and MDL-SDK-database
// bookkeeping (ADR 0009). Callers hand it work through run() and block for the
// result, so the registry and the SDK database are only ever touched by this
// one thread — confinement replaces per-container locking.
//
// The heavy, registry-free per-material compilation (create_compiled_material +
// translate_link_unit) fans out to a worker pool through submit(), which
// returns a future the caller collects later; workers reach back into the
// coordinator via run() for the bookkeeping they need serialized. Worker count
// comes from VISRTX_MDL_COMPILE_THREADS (>=1).
//
// Teardown contract: stop() drains and joins the workers first (they still need
// a live coordinator for their run() bookkeeping), then joins the coordinator.
// Any run() issued after stop() (e.g. a material released inline during device
// teardown) executes on the calling thread, so releases never deadlock on a
// thread that is already gone.
class MdlCompileCoordinator
{
 public:
  MdlCompileCoordinator();
  ~MdlCompileCoordinator();

  MdlCompileCoordinator(const MdlCompileCoordinator &) = delete;
  MdlCompileCoordinator &operator=(const MdlCompileCoordinator &) = delete;

  std::size_t numWorkers() const;

  // Execute f on the coordinator thread and block for its result. Runs inline
  // when the caller already is the coordinator thread (no self-deadlock) or
  // once the coordinator is stopping (the thread may be gone; teardown is
  // single-threaded by then). The enqueue-vs-inline decision is made under the
  // lock so a task is never stranded on an exited thread.
  template <typename F>
  std::invoke_result_t<F> run(F &&f);

  // Run f on a worker thread and return a future for its result. Non-blocking:
  // the caller submits during commitParameters and collects the future during
  // finalize, so every material's compile is in flight before the first wait.
  // f must not touch the registry directly — it reaches back through run().
  template <typename F>
  std::future<std::invoke_result_t<F>> submit(F &&f);

  // Drain and join the workers, then join the coordinator thread. Idempotent;
  // safe to call before the registry it serializes is destroyed.
  void stop();

 private:
  void coordinatorMain();
  void workerMain();

  // Coordinator thread: serialized bookkeeping (run()).
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

inline std::size_t MdlCompileCoordinator::numWorkers() const
{
  return m_workers.size();
}

template <typename F>
std::invoke_result_t<F> MdlCompileCoordinator::run(F &&f)
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
      // below until the coordinator thread has finished running it.
      m_queue.emplace_back([&task] { task(); });
      queued = true;
    }
  }
  if (!queued) {
    // Stopping: the thread may already be gone, so run inline (lock released).
    task();
    return result.get();
  }
  m_cv.notify_one();
  return result.get();
}

template <typename F>
std::future<std::invoke_result_t<F>> MdlCompileCoordinator::submit(F &&f)
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
    // Stopping: no worker will pick it up, so run inline.
    (*task)();
    return result;
  }
  m_workerCv.notify_one();
  return result;
}

} // namespace visrtx::mdl
