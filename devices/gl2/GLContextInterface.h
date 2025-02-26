// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <anari/anari.h>

namespace visgl2 {

void anariDeviceReportStatus(ANARIDevice,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *format,
    ...);

struct GLContextInterface
{
  GLContextInterface(ANARIDevice d);
  virtual ~GLContextInterface() = default;

  using loader_func_t = void (*(char const *))();
  virtual void init() = 0;
  virtual void release() = 0;
  virtual void makeCurrent() = 0;
  virtual loader_func_t *loaderFunc() = 0;

  ANARIDevice device() const;

private:
  ANARIDevice m_device{nullptr};
};

} // namespace visgl2
