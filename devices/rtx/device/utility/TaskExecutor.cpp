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

#include "TaskExecutor.h"

#include <stdexcept>

namespace visrtx {

TaskExecutor::TaskExecutor(std::size_t workerCount)
{
  if (workerCount == 0)
    throw std::invalid_argument("TaskExecutor requires at least one worker");
  // Allocation may throw. Do it before starting any joinable thread, since a
  // failed constructor does not run stop() through our destructor.
  m_workers.reserve(workerCount);

  m_thread = std::thread([this] { serialMain(); });
  // Set from the constructing thread before invokeSerial() can be reached, so
  // the re-entrancy check never races the serial thread writing its own id.
  m_threadId = m_thread.get_id();

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

TaskExecutor::~TaskExecutor()
{
  stop();
}

void TaskExecutor::stop()
{
  // Workers may still invokeSerial() while draining, so the serial thread
  // must outlive the pool.
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

void TaskExecutor::serialMain()
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

void TaskExecutor::workerMain()
{
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> guard(m_workerMutex);
      m_workerCv.wait(guard,
          [this] { return m_workersStopping || !m_workerQueue.empty(); });
      // Drain queued tasks before exiting so their waiters are released.
      if (m_workerQueue.empty())
        return;
      task = std::move(m_workerQueue.front());
      m_workerQueue.pop_front();
    }
    task();
  }
}

} // namespace visrtx
