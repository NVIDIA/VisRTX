// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Dataset.h"

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/scene/Scene.hpp"

#include <filesystem>
#include <string>

namespace tsd::core {
struct DataNode;
}

namespace tsd::scivis_studio {

constexpr const char *DATASET_FILE_TYPE = "dataset";
constexpr const char *DATASET_SCHEMA = "tsd.scivis-studio.dataset";

struct DatasetAssetValidationResult
{
  bool ok{false};
  std::string error;
  Dataset dataset;
};

DatasetAssetValidationResult validateDatasetAsset(
    const std::filesystem::path &file);
DatasetAssetValidationResult validateDatasetArchive(
    tsd::core::DataNode &archive);

bool saveDatasetArchiveFile(const Dataset &dataset,
    tsd::scene::LayerNodeRef root,
    tsd::animation::AnimationManager &animationManager,
    const std::filesystem::path &file,
    std::string *error = nullptr);

bool loadDatasetArchiveFile(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    const std::filesystem::path &file,
    tsd::scene::LayerNodeRef destinationParent,
    Dataset &datasetOut,
    tsd::scene::LayerNodeRef &rootOut,
    std::string *error = nullptr);

bool deserializeDatasetArchive(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::core::DataNode &archive,
    tsd::scene::LayerNodeRef destinationParent,
    Dataset &datasetOut,
    tsd::scene::LayerNodeRef &rootOut,
    std::string *error = nullptr);

void removeDatasetRuntime(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::scene::LayerNodeRef root);

bool datasetRuntimeContainsObject(tsd::scene::Scene &scene,
    tsd::scene::LayerNodeRef root,
    const tsd::scene::Object *object);

} // namespace tsd::scivis_studio
