// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Scene.hpp"
// imgui
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

namespace tsd::ui {

constexpr float INDENT_AMOUNT = 25.f;

void buildUI_object(tsd::core::Object &o,
    tsd::core::Scene &scene,
    bool useTableForParameters = false,
    int level = 0);
bool buildUI_parameter(tsd::core::Object &o,
    tsd::core::Parameter &p,
    tsd::core::Scene &scene,
    bool asTable = false);
size_t buildUI_objects_menulist(
    const tsd::core::Scene &scene, anari::DataType &type);

} // namespace tsd::ui
