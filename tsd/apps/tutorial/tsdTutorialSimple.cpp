// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Context.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>

int main()
{
  tsd::core::Context ctx;
  tsd::io::generate_randomSpheres(ctx);
  auto geom = ctx.getObject<tsd::core::Geometry>(0);
  geom->setName("main geom");
  tsd::core::print(*geom);
  return 0;
}
