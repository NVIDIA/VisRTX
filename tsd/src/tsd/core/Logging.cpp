// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Logging.hpp"

namespace tsd {

static LoggingCallback g_loggingCallback;

static void logMessage(LogLevel level, const char *fmt, va_list &args)
{
  if (g_loggingCallback)
    g_loggingCallback(level, fmt, args);
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

} // namespace tsd
