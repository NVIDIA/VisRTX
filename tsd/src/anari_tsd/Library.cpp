// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"
#include "anari/backend/LibraryImpl.h"
#include "anari_library_tsd_export.h"
#include "anari_library_tsd_queries.h"

namespace tsd_device {

struct Library : public anari::LibraryImpl
{
  Library(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

// Definitions ////////////////////////////////////////////////////////////////

Library::Library(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice Library::newDevice(const char * /*subtype*/)
{
  return (ANARIDevice) new Device(this_library());
}

const char **Library::getDeviceExtensions(const char * /*deviceType*/)
{
  return query_extensions();
}

} // namespace tsd_device

// Define library entrypoint //////////////////////////////////////////////////

extern "C" ANARI_TSD_EXPORT ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    tsd, handle, scb, scbPtr)
{
  return (ANARILibrary) new tsd_device::Library(handle, scb, scbPtr);
}
