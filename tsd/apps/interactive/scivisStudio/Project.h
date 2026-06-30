// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "CameraRig.h"
#include "Dataset.h"
#include "LightRig.h"
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

struct Project
{
  std::string name{"Untitled"};
  std::filesystem::path projectDirectory;
  std::vector<Dataset> datasets;
  std::vector<Shot> shots;
  std::vector<LightRig> lightRigs;
  std::vector<CameraRig> cameraRigs;
  std::vector<ColorMapRecord> colorMaps;
  ShotID activeShotId;
  uint64_t nextDatasetOrdinal{1};
  bool dirty{false};

  bool isSaved() const;
  void markDirty();
  void markClean();
};

namespace project {

std::string makeGeneratedId(const char *prefix, size_t ordinal);
DatasetID nextDatasetId(Project &project);
ShotID nextShotId(const Project &project);
ColorMapID nextColorMapId(const Project &project);

Dataset *findDataset(Project &project, const DatasetID &id);
const Dataset *findDataset(const Project &project, const DatasetID &id);
Shot *findShot(Project &project, const ShotID &id);
const Shot *findShot(const Project &project, const ShotID &id);
Shot *activeShot(Project &project);
const Shot *activeShot(const Project &project);

} // namespace project

} // namespace tsd::scivis_studio
