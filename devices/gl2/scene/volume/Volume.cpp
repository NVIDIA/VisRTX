// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Volume.h"

namespace visgl2 {

Volume::Volume(VisGL2DeviceGlobalState *s) : Object(ANARI_VOLUME, s) {}

Volume *Volume::createInstance(
    std::string_view subtype, VisGL2DeviceGlobalState *s)
{
  return (Volume *)new UnknownObject(ANARI_VOLUME, s);
}

void Volume::commitParameters()
{
  m_id = getParam<uint32_t>("id", ~0u);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Volume *);
