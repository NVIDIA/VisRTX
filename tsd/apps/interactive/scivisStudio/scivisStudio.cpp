// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Application.h"

int main(int argc, const char **argv)
{
  tsd::scivis_studio::Application app(argc, argv);
  app.run(1920, 1080, "SciVis Studio");
  return 0;
}
