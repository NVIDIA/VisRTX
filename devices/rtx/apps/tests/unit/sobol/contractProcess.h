// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <thread>

namespace contract_test {

struct Misuse
{
  int kind = 0; // raw, Owen, overlay, null overlay
  uint32_t dimension = 16u;
  uint32_t index = 1u;
  int n = 0;
  int cursor = 0;
};

inline Misuse parse(const char *mode)
{
  Misuse result;
  if (std::sscanf(mode, "qmc:%d:%d", &result.n, &result.cursor) == 2)
    result.kind = 2;
  else if (std::sscanf(mode, "null:%d", &result.cursor) == 1)
    result.kind = 3;
  else if (std::strstr(mode, "owen"))
    result.kind = 1;
  result.dimension = std::strstr(mode, "max") ? UINT32_MAX : 16u;
  result.index = std::strstr(mode, "zero") ? 0u : 1u;
  return result;
}

template <typename Check>
inline bool eachMisuse(Check check)
{
  for (const char *mode : {"raw16",
           "rawmax",
           "raw16zero",
           "rawmaxzero",
           "owen16",
           "owenmax",
           "owen16zero",
           "owenmaxzero",
           "null:-1",
           "null:1",
           "null:4",
           "null:5"}) {
    if (!check(mode))
      return false;
  }
  for (int n = -1; n <= 5; ++n) {
    for (int cursor = -1; cursor <= 5; ++cursor) {
      if (n >= 0 && n <= 4 && cursor >= 0 && cursor <= n)
        continue;
      char mode[40];
      std::snprintf(mode, sizeof(mode), "qmc:%d:%d", n, cursor);
      if (!check(mode))
        return false;
    }
  }
  return true;
}

// Transport only: callers retain distinct SIGABRT vs normal expected-error
// exit predicates. Timeout, exec failure and unrelated crashes never pass.
template <typename Accept>
inline bool run(
    const char *executable, const char *mode, int seconds, Accept accept)
{
  const pid_t child = fork();
  if (child == 0) {
    const rlimit noCore{0, 0};
    setrlimit(RLIMIT_CORE, &noCore);
    execl(executable, executable, mode, nullptr);
    _exit(127);
  }
  if (child < 0)
    return false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto waited = waitpid(child, &status, WNOHANG);
    if (waited == child) {
      if (accept(status))
        return true;
      std::fprintf(stderr, "%s: unexpected child status %d\n", mode, status);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  kill(child, SIGKILL);
  waitpid(child, &status, 0);
  std::fprintf(stderr, "%s: child timed out\n", mode);
  return false;
}

} // namespace contract_test
