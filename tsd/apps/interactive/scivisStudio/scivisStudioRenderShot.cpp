// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectContext.h"
#include "RenderShot.h"
#include "RenderShotCLI.h"

#include "tsd/app/Context.h"

#include <csignal>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define TSD_ISATTY _isatty
#define TSD_FILENO _fileno
#else
#include <unistd.h>
#define TSD_ISATTY isatty
#define TSD_FILENO fileno
#endif

namespace {

volatile std::sig_atomic_t g_canceled = 0;

void handleInterrupt(int)
{
  g_canceled = 1;
}

} // namespace

int main(int argc, const char **argv)
{
  using namespace tsd::scivis_studio;

  std::vector<std::string> args(argv, argv + argc);
  const auto programName = args.empty() ? "scivisStudioRenderShot" : args[0];

  RenderShotCommandLine commandLine;
  std::string error;
  if (!parseRenderShotCommandLine(args, commandLine, error)) {
    std::cerr << error << '\n' << renderShotUsage(programName);
    return 1;
  }

  if (commandLine.showHelp) {
    std::cout << renderShotUsage(programName);
    return 0;
  }

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  if (!projectContext.openProject(
          commandLine.projectDirectory, nullptr, nullptr, nullptr, &error)) {
    std::cerr << "failed to open project: " << error << '\n';
    return 1;
  }

  const bool interactive = TSD_ISATTY(TSD_FILENO(stdin)) != 0;
  auto *shot = selectShotForRender(projectContext.project(),
      commandLine.shotId,
      interactive,
      std::cin,
      std::cout,
      error);
  if (!shot) {
    std::cerr << error << '\n';
    return 1;
  }

  projectContext.project().activeShotId = shot->id;
  projectContext.syncAnimationManagerToActiveShot();
  projectContext.applyActiveShot();

  std::signal(SIGINT, handleInterrupt);

  std::cout << "Rendering shot " << shot->id << " \"" << shot->name
            << "\" from " << commandLine.projectDirectory.string() << '\n';

  RenderShotProgress progress;
  progress.onFrame = [](int frame, int totalFrames) {
    if (g_canceled)
      return false;

    std::cout << "Frame " << frame + 1 << " / " << totalFrames << '\n';
    return true;
  };

  const bool completed = renderActiveShotToFrames(projectContext, &progress);
  if (!completed) {
    if (g_canceled)
      std::cerr << "Canceled\n";
    else
      std::cerr << "Render failed\n";
    return 1;
  }

  std::cout << "Done\n";
  return 0;
}
