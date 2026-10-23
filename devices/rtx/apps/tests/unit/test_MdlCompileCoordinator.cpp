// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

#include "mdl/MdlCompileCoordinator.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
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

int main(int argc, char **argv)
{
  using visrtx::mdl::MdlCompileCoordinator;
  if (argc == 2 && std::string_view(argv[1]) == "--reserve-failure") {
    // CTest supplies VISRTX_MDL_COMPILE_THREADS=256. Before the fix, this
    // allocation throws with an already-joinable coordinator and terminates.
    failAllocationSize = 256 * sizeof(std::thread);
    try {
      MdlCompileCoordinator coordinator;
    } catch (const std::bad_alloc &) {
      return failAllocationSize == 0 ? 0 : 1;
    }
    std::fprintf(stderr, "FAIL: reserve allocation was not intercepted\n");
    return 1;
  }

  MdlCompileCoordinator coordinator;
  if (coordinator.run([&] { return coordinator.run([] { return 42; }); }) != 42)
    return 1;

  std::atomic<int> completed{0};
  std::vector<std::future<int>> jobs;
  for (int i = 0; i < 32; ++i) {
    jobs.push_back(coordinator.submit(
        [&] { return coordinator.run([&] { return ++completed; }); }));
  }
  coordinator.stop(); // Must drain worker-to-coordinator work before joining.
  coordinator.stop();
  for (auto &job : jobs)
    if (job.get() < 1)
      return 1;
  if (completed != 32)
    return 1;

  if (coordinator.run([] { return 1; }) != 1
      || coordinator.submit([] { return 2; }).get() != 2)
    return 1;
  std::puts("MDL coordinator lifecycle passed");
  return 0;
}
