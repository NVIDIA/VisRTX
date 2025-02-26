// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// std
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace visgl2 {

struct TaskQueue
{
  TaskQueue(size_t n);
  ~TaskQueue();

  template <class F, class... Args>
  std::future<void> enqueue(F &&f, Args &&...args);

 private:
  static void thread_fun(TaskQueue *ct);

  std::vector<std::packaged_task<void()>> m_tasks;
  bool m_stop{false};
  int m_next{0};
  int m_last{0};

  std::mutex m_mutex;
  std::condition_variable m_condition;
  std::thread m_thread;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline TaskQueue::TaskQueue(size_t n) : m_tasks(n), m_thread(thread_fun, this)
{}

inline TaskQueue::~TaskQueue()
{
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_stop = true;
    m_condition.notify_all();
  }
  m_thread.join();
}

template <class F, class... Args>
inline std::future<void> TaskQueue::enqueue(F &&f, Args &&...args)
{
  std::packaged_task<void()> task(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));
  std::future<void> future = task.get_future();

  if (std::this_thread::get_id() == m_thread.get_id())
    task();
  else {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condition.wait(
        lock, [&] { return (m_last + 1) % m_tasks.size() != m_next; });
    m_tasks[m_last] = std::move(task);
    m_last = (m_last + 1) % m_tasks.size();
    m_condition.notify_one();
  }

  return future;
}

inline void TaskQueue::thread_fun(TaskQueue *ct)
{
  while (true) {
    std::packaged_task<void()> task;
    {
      std::unique_lock<std::mutex> lock(ct->m_mutex);
      ct->m_condition.wait(
          lock, [ct] { return ct->m_stop || ct->m_next != ct->m_last; });
      if (ct->m_stop && ct->m_next == ct->m_last)
        break;
      task = std::move(ct->m_tasks[ct->m_next]);
      ct->m_next = (ct->m_next + 1) % ct->m_tasks.size();
      ct->m_condition.notify_one();
    }
    task();
  }
}

} // namespace visgl2
