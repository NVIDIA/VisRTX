// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/Volume.hpp"

namespace tsd {

Volume::Volume(Token stype) : Object(ANARI_VOLUME, stype) {}

anari::Object Volume::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Volume>(d, subtype().c_str());
}

namespace tokens::volume {

const Token transferFunction1D = "transferFunction1D";

} // namespace tokens::volume

} // namespace tsd
