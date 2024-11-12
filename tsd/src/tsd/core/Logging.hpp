// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <cstdarg>
#include <functional>

namespace tsd {

void logStatus(const char *fmt, ...);
void logError(const char *fmt, ...);
void logWarning(const char *fmt, ...);
void logPerfWarning(const char *fmt, ...);
void logInfo(const char *fmt, ...);
void logDebug(const char *fmt, ...);

enum LogLevel
{
  STATUS,
  ERROR,
  WARNING,
  PERF_WARNING,
  INFO,
  DEBUG,
  UNKNOWN
};

using LoggingCallback = std::function<void(LogLevel, const char *, va_list &)>;

void setLoggingCallback(LoggingCallback cb);

} // namespace tsd
