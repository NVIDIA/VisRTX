// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main(int argc, char *argv[])
{
  int result = Catch::Session().run(argc, argv);
  return result;
}
