// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-type entry points for the math usertype registrations. Each lives
// in its own TU under bindings/math/ so the five sol2 instantiations
// (float2/3/4, mat3, mat4) compile in parallel.

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerFloat2Usertype(sol::table &tsd);
void registerFloat3Usertype(sol::table &tsd);
void registerFloat4Usertype(sol::table &tsd);
void registerMat3Usertype(sol::table &tsd);
void registerMat4Usertype(sol::table &tsd);

} // namespace tsd::scripting
