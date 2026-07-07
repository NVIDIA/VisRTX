// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/FlatMap.hpp"
#include "tsd/core/ObjectPool.hpp"

#include <anari/anari_cpp.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

using DatasetID = std::string;
using ShotID = std::string;
using ColorMapID = std::string;
using LightRigID = std::string;
using CameraRigID = std::string;

struct SceneNodeRef
{
  std::string layerName;
  size_t nodeIndex{TSD_INVALID_INDEX};
};

struct SceneObjectRef
{
  anari::DataType type{ANARI_UNKNOWN};
  size_t objectIndex{TSD_INVALID_INDEX};
};

enum class DatasetSourceKind
{
  Static,
  FileAnimation,
  TimeSeries = FileAnimation, // v1-v4 source compatibility
  Live
};

enum class DatasetStatus
{
  Available,
  Unavailable,
  Missing = Unavailable, // v1-v4 source compatibility
  Importing,
  ImportFailed
};

// Residency is a second axis alongside DatasetStatus: status assesses
// availability and import progress, residency records whether the project has
// been asked to keep the runtime representation in the scene.
enum class DatasetResidency
{
  Loaded,
  Unloaded
};

struct DatasetSourceMetadata
{
  std::string sourcePath;
  tsd::core::FlatMap<std::string, std::string> importerSettings;
};

struct DatasetSourceFile
{
  // The raw entry as authored; entries are opaque and are written back
  // verbatim (ADR 0013).
  std::string path;
  // Runtime-only: a relative entry anchored once, at read, to its Source List
  // File's directory. Empty when the raw entry is used as-is (an absolute
  // entry or a legacy embedded one).
  std::string resolvedPath;
};

struct Dataset
{
  DatasetID id;
  std::string name;
  DatasetSourceKind sourceKind{DatasetSourceKind::Static};
  std::string importerType{"NONE"};
  DatasetSourceMetadata source;
  DatasetStatus status{DatasetStatus::Unavailable};
  DatasetResidency residency{DatasetResidency::Loaded};
  SceneNodeRef rootNode;
  std::vector<DatasetSourceFile> sourceFiles;

  // Runtime-only persistence state. Dataset assets do not store project-local
  // IDs or these bookkeeping fields.
  bool dirty{true};
  bool pendingExtraction{false};
  // The dataset file carries legacy embedded sourceFiles: the next explicit
  // save of the loaded dataset writes the Source List File and rewrites the
  // dataset file without paths (ADR 0013).
  bool pendingSourceListMigration{false};
  std::string persistedName;
};

namespace dataset {

const char *toString(DatasetSourceKind kind);
const char *toString(DatasetStatus status);
const char *toString(DatasetResidency residency);
DatasetSourceKind sourceKindFromString(const std::string &s);
DatasetStatus statusFromString(const std::string &s);
DatasetResidency residencyFromString(const std::string &s);

// One derived status string for display: availability, import progress, and
// residency collapse into Loaded / Unloaded / Unavailable / Importing /
// Import Failed.
const char *displayStatus(const Dataset &dataset);

} // namespace dataset

} // namespace tsd::scivis_studio
