// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

#include "utility/TaskExecutor.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string_view>

// Isolated executable: inject failure only into the constructing thread's
// worker-vector allocation. The deques and thread state allocate other sizes.
static thread_local std::size_t failAllocationSize = 0;

void *operator new(std::size_t size)
{
  if (size == failAllocationSize && failAllocationSize != 0) {
    failAllocationSize = 0;
    throw std::bad_alloc();
  }
  if (void *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}

void operator delete(void *p) noexcept
{
  std::free(p);
}

void operator delete(void *p, std::size_t) noexcept
{
  std::free(p);
}

namespace {

void check(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

void testExecutor(std::size_t workerCount)
{
  using visrtx::TaskExecutor;
  bool rejectedZero = false;
  try {
    TaskExecutor invalid(0);
  } catch (const std::invalid_argument &) {
    rejectedZero = true;
  }
  check(rejectedZero, "zero workers must be rejected");

  TaskExecutor executor(workerCount);
  check(executor.numWorkers() == workerCount, "explicit worker count ignored");
  const auto caller = std::this_thread::get_id();
  const auto serial =
      executor.invokeSerial([] { return std::this_thread::get_id(); });
  check(serial != caller, "serial work ran on caller");
  check(executor.invokeSerial([&] {
    return executor.invokeSerial([] { return std::this_thread::get_id(); });
  }) == serial,
      "serial reentrancy changed threads");

  auto moveOnly =
      executor.submit([value = std::make_unique<int>(42)]() mutable {
        return std::move(value);
      });
  check(*moveOnly.get() == 42, "move-only task/result lost");
  check(executor.submit([] { return std::this_thread::get_id(); }).get()
          != serial,
      "pool work ran on serial thread");

  bool serialThrew = false, workerThrew = false;
  try {
    executor.invokeSerial([] { throw std::runtime_error("serial failure"); });
  } catch (const std::runtime_error &) {
    serialThrew = true;
  }
  auto throwing =
      executor.submit([] { throw std::runtime_error("worker failure"); });
  try {
    throwing.get();
  } catch (const std::runtime_error &) {
    workerThrew = true;
  }
  check(serialThrew && workerThrew, "task exceptions were not propagated");

  // Hold a whole worker wave at a gate. This proves both that the requested
  // workers are available and that queued work cannot exceed that limit.
  std::atomic<std::size_t> active{0}, peak{0}, started{0};
  std::promise<void> firstWave;
  auto firstWaveReady = firstWave.get_future();
  std::promise<void> release;
  auto gate = release.get_future().share();
  int serialCount = 0;
  std::vector<std::future<int>> jobs;
  for (std::size_t i = 0; i < workerCount * 3; ++i) {
    jobs.push_back(executor.submit([&, gate] {
      const auto current = ++active;
      auto high = peak.load();
      while (high < current && !peak.compare_exchange_weak(high, current)) {
      }
      if (++started == workerCount)
        firstWave.set_value();
      gate.wait();
      const auto value = executor.invokeSerial([&] { return ++serialCount; });
      --active;
      return value;
    }));
  }
  firstWaveReady.get();
  const auto blockedWorkers = active.load();
  release.set_value();
  for (auto &job : jobs)
    check(job.get() > 0, "worker-to-serial task failed");
  check(blockedWorkers == workerCount && peak == workerCount,
      "worker pool did not respect its configured concurrency");
  check(serialCount == int(workerCount * 3), "serial calls were lost");

  // Drain outstanding work, including callbacks onto the serial lane.
  std::atomic<int> completed{0};
  jobs.clear();
  for (int i = 0; i < 32; ++i) {
    jobs.push_back(executor.submit(
        [&] { return executor.invokeSerial([&] { return ++completed; }); }));
  }
  executor.stop();
  executor.stop();
  for (auto &job : jobs)
    check(job.get() > 0, "drained task failed");
  check(
      completed == 32 && executor.numWorkers() == 0, "stop did not drain/join");

  // Preserve the single-threaded post-stop cleanup contract.
  check(executor.invokeSerial([] { return std::this_thread::get_id(); })
          == caller,
      "post-stop serial cleanup was not inline");
  check(executor.submit([] { return std::this_thread::get_id(); }).get()
          == caller,
      "post-stop worker cleanup was not inline");
}

} // namespace

int main(int argc, char **argv)
{
  const std::string_view mode = argc == 2 ? argv[1] : "";
  if (mode == "--reserve-failure") {
    // Allocation failure must propagate as bad_alloc without leaving a
    // joinable thread that would terminate the process during unwinding.
    failAllocationSize = 256 * sizeof(std::thread);
    try {
      visrtx::TaskExecutor executor(256);
    } catch (const std::bad_alloc &) {
      return failAllocationSize == 0 ? 0 : 1;
    }
    std::fprintf(stderr, "FAIL: reserve allocation was not intercepted\n");
    return 1;
  }
  try {
    testExecutor(mode == "--serial" ? 1 : 4);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  std::puts("TaskExecutor lifecycle, concurrency and exceptions passed");
  return 0;
}
