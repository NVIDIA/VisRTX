// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DistributedRenderServer.hpp"

int main(int argc, const char **argv)
{
  tsd::network::DistributedRenderServer server(argc, argv);
  server.run(12345);
  return 0;
}
