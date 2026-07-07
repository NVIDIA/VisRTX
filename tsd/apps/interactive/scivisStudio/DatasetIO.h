// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Dataset.h"

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/scene/Scene.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace tsd::core {
struct DataNode;
}

namespace tsd::scivis_studio {

constexpr const char *DATASET_FILE_TYPE = "dataset";
constexpr const char *DATASET_SCHEMA = "tsd.scivis-studio.dataset";
constexpr const char *SOURCE_LIST_FILE_EXTENSION = ".sources";

// The sibling Source List File for a dataset file: same directory and stem
// with the ".sources" extension. The pairing is a naming convention, not a
// stored path (ADR 0013).
std::filesystem::path sourceListFilePath(
    const std::filesystem::path &datasetFile);

// Read a Source List File: one path per line, lines trimmed, blank lines
// skipped, line order = frame order. Relative entries are anchored here —
// once, at read — to the file's directory; the raw entries stay authoritative
// and are what gets written back. A missing, unreadable, or empty file fails.
bool readSourceListFile(const std::filesystem::path &file,
    std::vector<DatasetSourceFile> &sourceList,
    std::string *error = nullptr);

bool writeSourceListFile(const std::filesystem::path &file,
    const std::vector<DatasetSourceFile> &sourceList,
    std::string *error = nullptr);

// True when a Dataset Archive persists its File Animation Source List in a
// sibling Source List File, i.e. it carries no legacy embedded sourceFiles.
bool datasetArchiveUsesSourceListFile(tsd::core::DataNode &archive);

// Read the raw Source List entries of a managed file-animation dataset asset
// without touching any runtime: from the sibling Source List File, or from
// the legacy embedded sourceFiles when the dataset file still embeds its
// list.
bool readDatasetSourceListEntries(const std::filesystem::path &datasetFile,
    std::vector<std::string> &entries,
    std::string *error = nullptr);

// Rewrite a managed file-animation dataset asset's Source List as an
// external-tool edit (ADR 0013): the Source List File is rewritten verbatim,
// no runtime is touched, no entry is validated, and the edit works regardless
// of the dataset's residency and availability, taking effect at the next
// Dataset Load. A legacy asset that still embeds sourceFiles migrates in
// place as part of the edit: the Source List File is written and the dataset
// file is rewritten without the embedded list at datatree level (the scene
// representation copied through untouched), staged together as one ADR 0004
// pair transaction.
bool writeDatasetSourceListEdit(const std::filesystem::path &datasetFile,
    const std::vector<std::string> &entries,
    std::string *error = nullptr);

struct DatasetAssetValidationResult
{
  bool ok{false};
  std::string error;
  Dataset dataset;
};

// validateDatasetAsset validates the complete asset: for a new-format
// file-animation dataset that includes reading the sibling Source List File.
// validateDatasetArchiveFile validates the dataset file's archive content
// alone (used where the sibling is staged and validated separately).
DatasetAssetValidationResult validateDatasetAsset(
    const std::filesystem::path &file);
DatasetAssetValidationResult validateDatasetArchiveFile(
    const std::filesystem::path &file);
DatasetAssetValidationResult validateDatasetArchive(
    tsd::core::DataNode &archive);

bool saveDatasetArchiveFile(const Dataset &dataset,
    tsd::scene::LayerNodeRef root,
    tsd::animation::AnimationManager &animationManager,
    const std::filesystem::path &file,
    std::string *error = nullptr);

// Serialize a Declared Dataset's asset: importer metadata and an empty
// subtree, no scene representation (ADR 0014). The sibling Source List File
// is staged separately by the caller that owns the pair. Only file-animation
// datasets can be declared.
bool saveDeclaredDatasetArchiveFile(const Dataset &dataset,
    const std::filesystem::path &file,
    std::string *error = nullptr);

bool loadDatasetArchiveFile(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    const std::filesystem::path &file,
    tsd::scene::LayerNodeRef destinationParent,
    Dataset &datasetOut,
    tsd::scene::LayerNodeRef &rootOut,
    std::string *error = nullptr);

// sourceList supplies the File Animation Source List read from the archive's
// sibling Source List File; a new-format file-animation archive fails without
// it. Legacy archives with embedded sourceFiles ignore it and mark the
// dataset for migration.
bool deserializeDatasetArchive(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::core::DataNode &archive,
    tsd::scene::LayerNodeRef destinationParent,
    const std::vector<DatasetSourceFile> *sourceList,
    Dataset &datasetOut,
    tsd::scene::LayerNodeRef &rootOut,
    std::string *error = nullptr);

// Remove a dataset subtree together with its owned animations and the whole
// object closure the subtree owns (objects still used elsewhere survive).
// Returns false — removing nothing — when the subtree cannot be planned.
bool removeDatasetRuntime(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::scene::LayerNodeRef root);

bool datasetRuntimeContainsObject(tsd::scene::Scene &scene,
    tsd::scene::LayerNodeRef root,
    const tsd::scene::Object *object);

} // namespace tsd::scivis_studio
