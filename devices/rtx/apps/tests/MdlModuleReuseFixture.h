// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

// Parameter names distinguish compiled-material UUIDs without changing the
// generated code. Defaults and overrides must remain material-specific.
inline std::string moduleReuseSource(int variant)
{
  const auto name = std::string("tint") + std::to_string(variant);
  return "mdl 1.6; import ::df::*; export material test(color " + name
      + " = color(1.0)) = material(surface: material_surface("
        "emission: material_emission(emission: df::diffuse_edf(), intensity: "
      + name + ")));";
}
