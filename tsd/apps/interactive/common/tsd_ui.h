// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/TSD.hpp"
// ui
#include <imgui.h>

namespace tsd::ui {

constexpr float INDENT_AMOUNT = 25.f;

void buildUI_object(tsd::Object &o,
    const tsd::Context &ctxeg,
    bool useTableForParameters = false,
    int level = 0);
void buildUI_parameter(
    tsd::Parameter &p, const tsd::Context &ctxeg, bool asTable = false);

void addDefaultRendererParameters(Object &o);
Object parseANARIObject(
    anari::Device d, ANARIDataType type, const char *subtype);

} // namespace tsd::ui