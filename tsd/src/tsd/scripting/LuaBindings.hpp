// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace sol {
class state;
}

namespace tsd::scripting {

void registerAllBindings(sol::state &lua);

// Individual binding registration functions
void registerMathBindings(sol::state &lua);
void registerCoreBindings(sol::state &lua);
void registerObjectBindings(sol::state &lua);
void registerLayerBindings(sol::state &lua);
void registerIOBindings(sol::state &lua);
void registerRenderBindings(sol::state &lua);

} // namespace tsd::scripting
