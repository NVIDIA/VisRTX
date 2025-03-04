// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/TSD.hpp"
// ui
#include <imgui.h>

namespace tsd::ui {

constexpr float INDENT_AMOUNT = 25.f;

void buildUI_object(tsd::Object &o,
    tsd::Context &ctx,
    bool useTableForParameters = false,
    int level = 0);
void buildUI_parameter(
    tsd::Parameter &p, tsd::Context &ctx, bool asTable = false);
size_t buildUI_objects_menulist(const Context &ctx, anari::DataType type);

void addDefaultRendererParameters(Object &o);
Object parseANARIObject(
    anari::Device d, ANARIDataType type, const char *subtype);

} // namespace tsd::ui
