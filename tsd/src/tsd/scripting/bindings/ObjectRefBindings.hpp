// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-type entry points for registering ObjectPoolRef<T> usertypes.
// Each implementation lives in its own TU under bindings/refs/ so the
// nine sol2 instantiations of `registerObjectMethodsOn` (the heaviest
// template in the bindings) compile in parallel instead of serialising
// through one giant ObjectBindings.cpp.

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerGeometryRef(sol::table &tsd);
void registerMaterialRef(sol::table &tsd);
void registerLightRef(sol::table &tsd);
void registerCameraRef(sol::table &tsd);
void registerSamplerRef(sol::table &tsd);
void registerSurfaceRef(sol::table &tsd);
void registerVolumeRef(sol::table &tsd);
void registerSpatialFieldRef(sol::table &tsd);
void registerArrayRef(sol::table &tsd);

} // namespace tsd::scripting
