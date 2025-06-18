// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "tsd/containers/DataTree.hpp"

int main(int argc, const char *argv[])
{
  if (argc < 2) {
    printf("usage: ./%s <file.tsd>\n", argv[0]);
    return 1;
  }

  tsd::serialization::DataTree tree;
  tree.load(argv[1]);
  tree.print();

  return 0;
}
