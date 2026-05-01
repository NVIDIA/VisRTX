// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Project.h"

#include "tsd/core/DataTree.hpp"

#include <filesystem>
#include <string>

namespace tsd::scivis_studio {

constexpr const char *PROJECT_KIND = "SciVisStudio";
constexpr int SCHEMA_VERSION = 1;
constexpr const char *PROJECT_MANIFEST_FILENAME = "project.tsd";

struct ProjectValidationResult
{
  bool ok{false};
  std::string error;
  std::filesystem::path manifestPath;
};

void projectToNode(const Project &project, tsd::core::DataNode &node);
bool nodeToProject(tsd::core::DataNode &node, Project &project);

ProjectValidationResult validateProjectRoot(
    const std::filesystem::path &directory);

} // namespace tsd::scivis_studio
