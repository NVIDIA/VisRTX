// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "tsd/TSD.hpp"

int main()
{
  tsd::Context ctx;
  tsd::generate_material_orb(ctx, ctx.defaultLayer()->root());
  tsd::save_Context(ctx, "saved_context.tsdx");
  return 0;
}
