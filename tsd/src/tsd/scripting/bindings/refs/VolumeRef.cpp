// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ObjectPoolRefRegister.hpp"
#include "../ObjectRefBindings.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Volume.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerVolumeRef(sol::table &tsd)
{
  auto volumeRefType = registerObjectPoolRef<scene::Volume>(tsd, "Volume");

  volumeRefType["spatialField"] =
      +[](scene::VolumeRef &r) -> scene::SpatialField * {
    if (!r.valid())
      return nullptr;
    return r.data()->parameterValueAsObject<scene::SpatialField>("value");
  };
}

} // namespace tsd::scripting
