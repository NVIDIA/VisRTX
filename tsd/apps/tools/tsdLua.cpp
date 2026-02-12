// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scripting/LuaContext.hpp"

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

// std
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

static void printUsage(const char *progName)
{
  printf("Usage: %s <script.lua> [args...]\n", progName);
  printf("       %s -e \"<lua code>\"\n", progName);
  printf("       %s -i               (interactive mode)\n", progName);
  printf("\n");
  printf("TSD Lua scripting tool for batch scene processing and rendering.\n");
  printf("\n");
  printf("Options:\n");
  printf("  -e <code>   Execute the given Lua code string\n");
  printf("  -i          Start interactive REPL mode\n");
  printf("  -h, --help  Show this help message\n");
  printf("\n");
  printf("Example scripts:\n");
  printf("  -- Load and render a scene\n");
  printf("  local scene = tsd.Scene.new()\n");
  printf("  tsd.io.importOBJ(scene, \"model.obj\")\n");
  printf("  local device = tsd.render.loadDevice(\"visrtx\")\n");
  printf("  local index = tsd.render.createRenderIndex(scene, device)\n");
  printf("  index:populate()\n");
  printf("  local camera = tsd.CameraSetup.new()\n");
  printf("  camera.position = tsd.vec3(0, 0, 5)\n");
  printf("  local pipeline = tsd.render.createPipeline(1920, 1080, device, "
         "index, camera)\n");
  printf(
      "  tsd.render.renderToFile(pipeline, 128, \"output.ppm\", 1920, 1080)\n");
}

static void runInteractiveMode(tsd::scripting::LuaContext &ctx)
{
  printf("TSD Lua Interactive Mode\n");
  printf("Type 'exit' or 'quit' to exit, 'help' for usage hints.\n");
  printf("\n");

  // Create a scene for interactive use
  auto *scene = ctx.createOwnedScene("scene");
  printf("Created scene, available as 'scene'\n");
  (void)scene; // Suppress unused warning

  std::string line;
  while (true) {
    printf("tsd> ");
    fflush(stdout);

    if (!std::getline(std::cin, line)) {
      printf("\n");
      break;
    }

    // Trim whitespace
    size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos) {
      continue;
    }
    line = line.substr(start);

    // Check for special commands
    if (line == "exit" || line == "quit") {
      break;
    }
    if (line == "help") {
      printf("Available globals:\n");
      printf("  scene     - The current TSD scene\n");
      printf("  tsd       - The TSD Lua module\n");
      printf("\n");
      printf("TSD namespaces:\n");
      printf("  tsd.io       - Importers and procedural generators\n");
      printf("  tsd.render   - Rendering functions (loadDevice, "
             "createRenderIndex, etc.)\n");
      printf("\n");
      printf("Example:\n");
      printf("  tsd.io.generateRandomSpheres(scene)\n");
      printf("  print(scene:numberOfObjects(tsd.GEOMETRY))\n");
      continue;
    }

    // Execute the line
    auto result = ctx.executeString(line);
    if (!result.success) {
      fprintf(stderr, "Error: %s\n", result.error.c_str());
    }
    // Output is printed via the print callback
  }
}

int main(int argc, const char *argv[])
{
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  // Check for help
  if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
    printUsage(argv[0]);
    return 0;
  }

  // Create Lua context
  tsd::scripting::LuaContext ctx;

  // Set up print callback to output to stdout
  ctx.setPrintCallback([](const std::string &msg) { printf("%s", msg.c_str()); });

  // Set up module search paths
  ctx.addScriptSearchPaths(tsd::scripting::LuaContext::defaultSearchPaths());

  // Handle -e option (execute string)
  if (strcmp(argv[1], "-e") == 0) {
    if (argc < 3) {
      fprintf(stderr, "Error: -e requires a Lua code argument\n");
      return 1;
    }

    // Create a scene for the script
    ctx.createOwnedScene("scene");

    auto result = ctx.executeString(argv[2]);
    if (!result.success) {
      fprintf(stderr, "Error: %s\n", result.error.c_str());
      return 1;
    }
    return 0;
  }

  // Handle -i option (interactive mode)
  if (strcmp(argv[1], "-i") == 0) {
    runInteractiveMode(ctx);
    return 0;
  }

  // Otherwise, treat first argument as a script file
  const char *scriptPath = argv[1];

  // Pass remaining args to Lua as 'arg' table
  sol::state &lua = ctx.lua();
  sol::table argTable = lua.create_table();
  argTable[0] = scriptPath;
  for (int i = 2; i < argc; i++) {
    argTable[i - 1] = argv[i];
  }
  lua["arg"] = argTable;

  // Create a scene for the script (can be overridden by the script)
  ctx.createOwnedScene("scene");

  // Execute the script
  auto result = ctx.executeFile(scriptPath);
  if (!result.success) {
    fprintf(stderr, "Error executing %s:\n%s\n", scriptPath, result.error.c_str());
    return 1;
  }

  return 0;
}
