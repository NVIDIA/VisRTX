// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ObjectPoolRefRegister.hpp"
#include "../ObjectRefBindings.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerSpatialFieldRef(sol::table &tsd)
{
  auto fieldRefType =
      registerObjectPoolRef<scene::SpatialField>(tsd, "SpatialField");

  fieldRefType["computeValueRange"] =
      +[](scene::SpatialFieldRef &r) -> math::float2 {
    if (!r.valid())
      return math::float2(0.f);
    return r.data()->computeValueRange();
  };
}

} // namespace tsd::scripting
