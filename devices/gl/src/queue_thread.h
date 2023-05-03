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

class queue_thread {
  std::vector<std::packaged_task<void()>> tasks;
  bool stop;
  int next;
  int last;

  std::mutex mutex;
  std::condition_variable condition;
  std::thread thread;

  static void thread_fun(queue_thread *ct) {
    for(;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(ct->mutex);
        ct->condition.wait(lock, [ct]{ return ct->stop || ct->next != ct->last; });
        if(ct->stop && ct->next == ct->last) {
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
  queue_thread(size_t n) : tasks(n), stop(false), next(0), last(0), thread(thread_fun, this) {

  }

  template<class F, class... Args>
  std::future<void> enqueue(F&& f, Args&&... args) {
    std::packaged_task<void()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<void> future = task.get_future();

    if(std::this_thread::get_id() != thread.get_id()) {
      std::unique_lock<std::mutex> lock(mutex);
      condition.wait(lock, [&]{ return (last + 1) % tasks.size() != next; });
      tasks[last] = std::move(task);
      last = (last + 1) % tasks.size();
      condition.notify_one();
    } else {
      task();
    }

    return future;
  }

  ~queue_thread() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      stop = true;
      condition.notify_all();
    }
    thread.join();
  }
};
