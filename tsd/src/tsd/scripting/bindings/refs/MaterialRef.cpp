// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ObjectPoolRefRegister.hpp"
#include "../ObjectRefBindings.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerMaterialRef(sol::table &tsd)
{
  registerObjectPoolRef<scene::Material>(tsd, "Material");
}

} // namespace tsd::scripting
