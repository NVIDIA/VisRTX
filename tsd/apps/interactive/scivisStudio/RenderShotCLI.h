// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Project.h"

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

struct RenderShotCommandLine
{
  std::filesystem::path projectDirectory;
  std::string shotId;
  bool showHelp{false};
};

bool parseRenderShotCommandLine(const std::vector<std::string> &args,
    RenderShotCommandLine &commandLine,
    std::string &error);

std::string renderShotUsage(const std::string &programName);
std::string formatShotList(const Project &project);

Shot *findShotById(Project &project, const std::string &shotId);
const Shot *findShotById(const Project &project, const std::string &shotId);

Shot *selectShotForRender(Project &project,
    const std::string &shotId,
    bool interactive,
    std::istream &input,
    std::ostream &output,
    std::string &error);

} // namespace tsd::scivis_studio
