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

#include "MdlCompileCoordinator.h"

#include <algorithm>
#include <cstdlib>
#include <thread>

namespace visrtx::mdl {

namespace {

// Worker count from VISRTX_MDL_COMPILE_THREADS (clamped to >=1). Unset defaults
// to the hardware concurrency, capped so a many-core host does not oversubscribe
// the shared MDL SDK compiler far past the point of return (compilation is
// bounded by the material count in practice, and the PTX-identity gate validates
// the parallel path). Set the env to 1 to force serial compilation.
constexpr std::size_t kMaxDefaultWorkers = 8;
// Hard ceiling for an explicit VISRTX_MDL_COMPILE_THREADS, so a typo (e.g.
// 100000) cannot try to spawn an absurd number of threads.
constexpr std::size_t kMaxWorkers = 256;

std::size_t resolveWorkerCount()
{
  if (const char *env = std::getenv("VISRTX_MDL_COMPILE_THREADS")) {
    char *end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end != env && value >= 1)
      return std::clamp<std::size_t>(
          static_cast<std::size_t>(value), 1, kMaxWorkers);
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return std::clamp<std::size_t>(hw, 1, kMaxDefaultWorkers);
}

} // namespace

MdlCompileCoordinator::MdlCompileCoordinator()
{
  m_thread = std::thread([this] { coordinatorMain(); });
  // Set from the constructing thread before run() can be reached, so the
  // re-entrancy check never races the coordinator thread writing its own id.
  m_threadId = m_thread.get_id();

  const std::size_t workerCount = resolveWorkerCount();
  m_workers.reserve(workerCount);
  try {
    for (std::size_t i = 0; i < workerCount; ++i)
      m_workers.emplace_back([this] { workerMain(); });
  } catch (...) {
    // A worker std::thread ctor threw: the destructor will not run during ctor
    // unwinding, so stop the already-started threads here or their joinable
    // members would std::terminate.
    stop();
    throw;
  }
}

MdlCompileCoordinator::~MdlCompileCoordinator()
{
  stop();
}

void MdlCompileCoordinator::stop()
{
  // Workers first: a worker's job reaches back into the coordinator through
  // run() for its bookkeeping, so the coordinator thread must outlive them.
  if (!m_workers.empty()) {
    {
      std::lock_guard<std::mutex> guard(m_workerMutex);
      m_workersStopping = true;
    }
    m_workerCv.notify_all();
    for (auto &worker : m_workers) {
      if (worker.joinable())
        worker.join();
    }
    m_workers.clear();
  }

  if (m_thread.joinable()) {
    {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_stopping = true;
    }
    m_cv.notify_all();
    m_thread.join();
  }
}

void MdlCompileCoordinator::coordinatorMain()
{
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> guard(m_mutex);
      m_cv.wait(guard, [this] { return m_stopping || !m_queue.empty(); });
      // Drain outstanding work before honoring the stop, so a task queued just
      // before stop() still runs and its waiter is released.
      if (m_queue.empty())
        return;
      task = std::move(m_queue.front());
      m_queue.pop_front();
    }
    task();
  }
}

void MdlCompileCoordinator::workerMain()
{
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> guard(m_workerMutex);
      m_workerCv.wait(
          guard, [this] { return m_workersStopping || !m_workerQueue.empty(); });
      // Drain queued compiles before exiting so their waiters are released.
      if (m_workerQueue.empty())
        return;
      task = std::move(m_workerQueue.front());
      m_workerQueue.pop_front();
    }
    task();
  }
}

} // namespace visrtx::mdl
