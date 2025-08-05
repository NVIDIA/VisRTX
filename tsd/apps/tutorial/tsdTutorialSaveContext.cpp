// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Context.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>
#include <tsd/io/serialization.hpp>

int main()
{
  tsd::core::Context ctx;
  tsd::io::generate_material_orb(ctx, ctx.defaultLayer()->root());
  tsd::io::save_Context(ctx, "saved_context.tsd");
  return 0;
}
