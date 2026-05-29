// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Dataset.h"
#include "Shot.h"

#include <filesystem>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

struct ColorMapRecord
{
  ColorMapID id;
  std::string name;
};

struct LightRig
{
  LightRigID id;
  std::string name;
  SceneNodeRef rootNode;
};

struct Project
{
  std::string name{"Untitled"};
  std::filesystem::path projectDirectory;
  std::vector<Dataset> datasets;
  std::vector<Shot> shots;
  std::vector<LightRig> lightRigs;
  std::vector<ColorMapRecord> colorMaps;
  ShotID activeShotId;
  bool dirty{false};

  bool isSaved() const;
  void markDirty();
  void markClean();
};

namespace project {

std::string makeGeneratedId(const char *prefix, size_t ordinal);
DatasetID nextDatasetId(const Project &project);
ShotID nextShotId(const Project &project);
ColorMapID nextColorMapId(const Project &project);
LightRigID nextLightRigId(const Project &project);

Dataset *findDataset(Project &project, const DatasetID &id);
const Dataset *findDataset(const Project &project, const DatasetID &id);
Shot *findShot(Project &project, const ShotID &id);
const Shot *findShot(const Project &project, const ShotID &id);
LightRig *findLightRig(Project &project, const LightRigID &id);
const LightRig *findLightRig(const Project &project, const LightRigID &id);
Shot *activeShot(Project &project);
const Shot *activeShot(const Project &project);

} // namespace project

} // namespace tsd::scivis_studio
