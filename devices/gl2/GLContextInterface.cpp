// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "GLContextInterface.h"
#include "VisGL2Device.h"
// std
#include <cstdarg>

namespace visgl2 {

void anariDeviceReportStatus(ANARIDevice handle,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *format,
    ...)
{
  if (handle == nullptr)
    return;

  auto *d = (VisGL2Device *)handle;
  auto &state = *d->deviceState();
  if (!state.statusCB)
    return;

  va_list arglist;
  va_list arglist_copy;
  va_start(arglist, format);
  va_copy(arglist_copy, arglist);
  int count = std::vsnprintf(nullptr, 0, format, arglist);
  va_end(arglist);

  std::vector<char> formattedMessage(size_t(count + 1));

  std::vsnprintf(
      formattedMessage.data(), size_t(count + 1), format, arglist_copy);
  va_end(arglist_copy);

  state.statusCB(state.statusCBUserPtr,
      handle,
      handle,
      ANARI_DEVICE,
      severity,
      code,
      formattedMessage.data());
}

// GLContextInterface definitions /////////////////////////////////////////////

GLContextInterface::GLContextInterface(ANARIDevice d) : m_device(d) {}

ANARIDevice GLContextInterface::device() const
{
  return m_device;
}

} // namespace visgl2
