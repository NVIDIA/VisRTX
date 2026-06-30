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

struct DatasetSourceMetadata
{
  std::string sourcePath;
  tsd::core::FlatMap<std::string, std::string> importerSettings;
};

struct DatasetSourceFile
{
  std::string path;
};

struct Dataset
{
  DatasetID id;
  std::string name;
  DatasetSourceKind sourceKind{DatasetSourceKind::Static};
  std::string importerType{"NONE"};
  DatasetSourceMetadata source;
  DatasetStatus status{DatasetStatus::Unavailable};
  SceneNodeRef rootNode;
  std::vector<DatasetSourceFile> sourceFiles;

  // Runtime-only persistence state. Dataset assets do not store project-local
  // IDs or these bookkeeping fields.
  bool dirty{true};
  bool pendingExtraction{false};
  std::string persistedName;
};

namespace dataset {

const char *toString(DatasetSourceKind kind);
const char *toString(DatasetStatus status);
DatasetSourceKind sourceKindFromString(const std::string &s);
DatasetStatus statusFromString(const std::string &s);

} // namespace dataset

} // namespace tsd::scivis_studio
