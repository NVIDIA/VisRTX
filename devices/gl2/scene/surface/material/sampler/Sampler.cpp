// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Sampler.h"

namespace visgl2 {

Sampler::Sampler(VisGL2DeviceGlobalState *s) : Object(ANARI_SAMPLER, s) {}

Sampler *Sampler::createInstance(
    std::string_view subtype, VisGL2DeviceGlobalState *s)
{
  return (Sampler *)new UnknownObject(ANARI_SAMPLER, s);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Sampler *);
