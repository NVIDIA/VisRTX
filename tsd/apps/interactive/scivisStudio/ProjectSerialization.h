// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Project.h"

#include "tsd/core/DataTree.hpp"

#include <filesystem>
#include <string>

namespace tsd::scivis_studio {

constexpr const char *PROJECT_KIND = "SciVisStudio";
constexpr const char *PROJECT_FILE_TYPE = "project";
constexpr const char *PROJECT_SCHEMA = "tsd.scivis-studio.project";
constexpr int SCHEMA_VERSION = 3;
constexpr const char *PROJECT_MANIFEST_FILENAME = "project.tsd";

// Standalone rig file schemas, versioned independently of the project schema.
constexpr const char *CAMERA_RIG_FILE_TYPE = "camera-rig";
constexpr const char *CAMERA_RIG_SCHEMA = "tsd.scivis-studio.camerarig";
constexpr const char *LIGHT_RIG_FILE_TYPE = "light-rig";
constexpr const char *LIGHT_RIG_SCHEMA = "tsd.scivis-studio.lightrig";
constexpr int RIG_SCHEMA_VERSION = 1;

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

// Standalone camera-rig file IO. The rig's runtime-only id is intentionally not
// written; only the portable name and value data (ShotCameraRig) are stored.
bool exportCameraRigFile(const std::string &name,
    const ShotCameraRig &rig,
    const std::filesystem::path &file,
    std::string *error = nullptr);
bool importCameraRigFile(const std::filesystem::path &file,
    std::string &nameOut,
    ShotCameraRig &rigOut,
    std::string *error = nullptr);

} // namespace tsd::scivis_studio
