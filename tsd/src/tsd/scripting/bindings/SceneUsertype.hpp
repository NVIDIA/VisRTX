// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-group entry points for registering methods on the
// `sol::usertype<scene::Scene>` object. Each group lives in its own TU
// under bindings/scene/ so the ~46 method bindings parallelise across
// cores instead of serialising through one giant CoreBindings.cpp.
//
// The usertype itself is created once by registerContextBindings()
// (with sol::no_constructor / constructors), then each register* call
// layers methods onto it.

#include "tsd/scene/Scene.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerSceneCreators(sol::usertype<scene::Scene> &sceneType);
void registerSceneAccessors(sol::usertype<scene::Scene> &sceneType);
void registerSceneIteration(sol::usertype<scene::Scene> &sceneType);
void registerSceneLayers(sol::usertype<scene::Scene> &sceneType);
void registerSceneNodes(sol::usertype<scene::Scene> &sceneType);

} // namespace tsd::scripting
