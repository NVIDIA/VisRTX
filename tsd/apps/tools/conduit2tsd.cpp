// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/TSD.hpp"

int main(int argc, const char *argv[])
{
  if (argc < 3) {
    printf("usage: ./conduit2tsd file.tsdc [output].tsd\n");
    return 1;
  } else if (argc > 3) {
    printf("usage: ./conduit2tsd file.tsdc [output].tsd\n");
    return 1;
  }

  tsd::Context ctx;
  printf("Loading context from file: %s...", argv[1]);
  fflush(stdout);
  tsd::import_Context_Conduit(ctx, argv[1]);
  printf("done!\n");

  printf("Saving context to file: %s...", argv[2]);
  fflush(stdout);
  tsd::save_Context(ctx, argv[2]);
  printf("done!\n");

  return 0;
}
