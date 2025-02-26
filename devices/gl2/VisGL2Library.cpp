// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGL2Device.h"
#include "anari/backend/LibraryImpl.h"
#include "anari_library_visgl2_export.h"

namespace visgl2 {

struct VisGL2Library : public anari::LibraryImpl
{
  VisGL2Library(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

// Definitions ////////////////////////////////////////////////////////////////

VisGL2Library::VisGL2Library(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice VisGL2Library::newDevice(const char * /*subtype*/)
{
  return (ANARIDevice) new VisGL2Device(this_library());
}

const char **VisGL2Library::getDeviceExtensions(const char * /*deviceType*/)
{
  return nullptr;
}

} // namespace visgl2

// Define library entrypoint //////////////////////////////////////////////////

extern "C" VISGL2_EXPORT ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    visgl2, handle, scb, scbPtr)
{
  return (ANARILibrary) new visgl2::VisGL2Library(handle, scb, scbPtr);
}
