// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ObjectPoolRefRegister.hpp"
#include "../ObjectRefBindings.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/Surface.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerSurfaceRef(sol::table &tsd)
{
  auto surfaceRefType = registerObjectPoolRef<scene::Surface>(tsd, "Surface");

  surfaceRefType["geometry"] = +[](scene::SurfaceRef &r) -> scene::Geometry * {
    if (!r.valid())
      return nullptr;
    return r.data()->parameterValueAsObject<scene::Geometry>("geometry");
  };
  surfaceRefType["material"] = +[](scene::SurfaceRef &r) -> scene::Material * {
    if (!r.valid())
      return nullptr;
    return r.data()->parameterValueAsObject<scene::Material>("material");
  };
}

} // namespace tsd::scripting
