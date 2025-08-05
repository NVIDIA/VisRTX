// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Context.hpp"
// imgui
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

namespace tsd::ui {

constexpr float INDENT_AMOUNT = 25.f;

void buildUI_object(tsd::core::Object &o,
    tsd::core::Context &ctx,
    bool useTableForParameters = false,
    int level = 0);
void buildUI_parameter(tsd::core::Object &o,
    tsd::core::Parameter &p,
    tsd::core::Context &ctx,
    bool asTable = false);
size_t buildUI_objects_menulist(
    const tsd::core::Context &ctx, anari::DataType type);

} // namespace tsd::ui
