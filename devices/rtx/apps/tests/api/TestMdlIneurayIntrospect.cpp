// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

// The INeuray handoff is only usable if an application can discover it, so the
// `ineuray` device parameter and property must be part of the published
// VISRTX_MATERIAL_MDL extension description, not merely accepted by the code.

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <cstdio>
#include <cstring>

namespace {

bool hasParameter(ANARIDevice d, const char *name, ANARIDataType type)
{
  auto *params = (const ANARIParameter *)anariGetObjectInfo(
      d, ANARI_DEVICE, "device", "parameter", ANARI_PARAMETER_LIST);
  for (auto *p = params; p && p->name; ++p)
    if (!std::strcmp(p->name, name) && p->type == type)
      return true;
  return false;
}

// ANARI's query generator has no "property" info name (its vocabulary is
// required/default/minimum/maximum/description/elementType/value/
// sourceExtension/extension/parameter/channel/use), so object properties --
// including the core spec's own geometryMaxIndex -- are not introspectable.
// The property is therefore verified by reading it, not by listing it.
bool propertyReadable(ANARIDevice d, const char *name)
{
  void *value = nullptr;
  return anariGetProperty(
             d, d, name, ANARI_VOID_POINTER, &value, sizeof(value), ANARI_WAIT)
      && value != nullptr;
}

bool check(bool ok, const char *what)
{
  std::printf("%-56s %s\n", what, ok ? "ok" : "MISSING");
  return ok;
}

} // namespace

int main()
{
  auto device = makeVisRTXDevice();
  bool ok = true;
  ok &= check(hasParameter(device, "ineuray", ANARI_VOID_POINTER),
      "device parameter `ineuray` (ANARI_VOID_POINTER)");
  // Readable only once MDL is initialized.
  anari::setParameter(device, device, "forceInit", true);
  anari::commitParameters(device, device);
  ok &= check(propertyReadable(device, "ineuray"),
      "device property  `ineuray` readable (ANARI_VOID_POINTER)");
  ok &= check(hasParameter(device, "mdlSearchPaths", ANARI_STRING),
      "device parameter `mdlSearchPaths` (ANARI_STRING)");
  ok &= check(hasParameter(device, "mdlResourceSearchPaths", ANARI_STRING),
      "device parameter `mdlResourceSearchPaths` (ANARI_STRING)");
  // The core device parameters must survive the merge.
  ok &= check(hasParameter(device, "statusCallback", ANARI_STATUS_CALLBACK),
      "core parameter   `statusCallback` still advertised");
  anari::release(device, device);

  if (!ok) {
    std::fprintf(stderr,
        "FAIL: the MDL instance-sharing contract is not discoverable through "
        "ANARI introspection\n");
    return 1;
  }
  std::puts("MDL device parameters and properties are advertised");
  return 0;
}
