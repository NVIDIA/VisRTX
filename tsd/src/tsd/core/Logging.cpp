// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Logging.hpp"
// tsd
#include "tsd/core/Timer.hpp"
// std
#include <chrono>
#include <cstdarg>
#include <string>
#include <vector>
// fmt
#include <fmt/format.h>

namespace tsd::core {

static LoggingCallback g_loggingCallback;
static tsd::core::Timer g_timer;
static int g_initTimer = []() { return g_timer.start(), 1; }();

static void logMessage(LogLevel level, const char *fmt, va_list &args)
{
  if (g_loggingCallback) {
    va_list args_copy;
    va_copy(args_copy, args);
    int len = std::vsnprintf(nullptr, 0, fmt, args_copy);
    va_end(args_copy);

    if (len < 0)
      return;

    std::vector<char> buf(len + 1);
    std::vsnprintf(buf.data(), buf.size(), fmt, args);
    std::string message(buf.data(), len);

    g_timer.end();
    g_loggingCallback(
        level, fmt::format("[{:.{}f}s] {}\n", g_timer.seconds(), 3, message));
  }
}

///////////////////////////////////////////////////////////////////////////////

void logStatus(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  logMessage(LogLevel::STATUS, fmt, args);
  va_end(args);
}

void logError(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  logMessage(LogLevel::ERROR, fmt, args);
  va_end(args);
}

void logWarning(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  logMessage(LogLevel::WARNING, fmt, args);
  va_end(args);
}

void logPerfWarning(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  logMessage(LogLevel::PERF_WARNING, fmt, args);
  va_end(args);
}

void logInfo(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  logMessage(LogLevel::INFO, fmt, args);
  va_end(args);
}

void logDebug(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  logMessage(LogLevel::DEBUG, fmt, args);
  va_end(args);
}

void setLoggingCallback(LoggingCallback cb)
{
  g_loggingCallback = cb;
}

void setLogToStdout()
{
  setLoggingCallback([](LogLevel level, std::string message) {
    fmt::print(stdout, "{}", message);
  });
}

void setLogToStderr()
{
  setLoggingCallback([](LogLevel level, std::string message) {
    fmt::print(stderr, "{}", message);
  });
}

void setNoLogging()
{
  g_loggingCallback = {};
}

} // namespace tsd::core
