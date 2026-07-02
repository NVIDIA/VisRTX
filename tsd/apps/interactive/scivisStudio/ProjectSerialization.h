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
// v6: the manifest no longer embeds a residual Scene Archive. Camera and
// renderer pools live in required Archives under scene/.
constexpr int DECOMPOSED_SCENE_SCHEMA_VERSION = 6;
// v7: each manifest dataset records its residency (Loaded/Unloaded); an
// absent field means Loaded, so v6 projects behave identically.
constexpr int SCHEMA_VERSION = 7;
constexpr const char *PROJECT_MANIFEST_FILENAME = "project.tsd";

// Standalone rig Archive schemas, versioned independently of the project.
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

// Rig names double as on-disk filenames ("<name>.tsd"), so they are restricted
// to a portable character set (letters, digits, space, '_', '-', '(', ')') with
// no leading/trailing whitespace. validateRigName checks a user-entered name's
// format only (not collection uniqueness); sanitizeRigName coerces an arbitrary
// string (e.g. a loaded rig's stored name) into that set.
bool validateRigName(const std::string &name, std::string *error = nullptr);
std::string sanitizeRigName(const std::string &name);

ProjectValidationResult validateProjectRoot(
    const std::filesystem::path &directory);

} // namespace tsd::scivis_studio
