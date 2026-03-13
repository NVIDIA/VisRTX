// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/scene/Scene.hpp"
// imgui
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

namespace tsd::ui {

constexpr float INDENT_AMOUNT = 25.f;

void buildUI_object(tsd::scene::Object &o,
    tsd::scene::Scene &scene,
    bool useTableForParameters = false,
    int level = 0);
bool buildUI_parameter(tsd::scene::Object &o,
    tsd::scene::Parameter &p,
    tsd::scene::Scene &scene,
    bool asTable = false);
size_t buildUI_objects_menulist(
    const tsd::scene::Scene &scene, anari::DataType &type);

} // namespace tsd::ui
