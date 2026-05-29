// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Dataset.h"
#include "ShotCameraRig.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

struct ShotRenderSettings
{
  uint32_t width{1024};
  uint32_t height{768};
  uint32_t samples{128};
  std::string rendererLibrary;
  size_t rendererObjectIndex{TSD_INVALID_INDEX};
  std::string rendererSubtype{"default"};
  std::string outputFilePrefix;
};

struct DatasetBinding
{
  DatasetID datasetId;
  bool enabled{true};
};

struct Shot
{
  ShotID id;
  std::string name;
  int frameCount{120};
  float fps{24.f};
  int currentFrame{0};
  bool playing{false};
  bool loop{true};
  std::vector<DatasetBinding> datasetBindings;
  LightRigID lightRigId;
  SceneObjectRef camera;
  ShotCameraRig cameraRig;
  ShotRenderSettings renderSettings;
};

namespace shot {

DatasetBinding *findDatasetBinding(Shot &shot, const DatasetID &id);
const DatasetBinding *findDatasetBinding(const Shot &shot, const DatasetID &id);
void setDatasetBinding(Shot &shot, const DatasetID &id, bool enabled);

} // namespace shot

} // namespace tsd::scivis_studio
