// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "tsd/TSD.hpp"
// std
#include <cstdio>

int main()
{
  tsd::Context ctx;
  tsd::generate_randomSpheres(ctx);
  auto geom = ctx.getObject<tsd::Geometry>(0);
  geom->setName("main geom");
  tsd::print(*geom);
  return 0;
}
