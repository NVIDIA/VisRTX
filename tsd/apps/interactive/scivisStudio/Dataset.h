// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
  TimeSeries,
  Live
};

enum class DatasetStatus
{
  Available,
  Missing,
  Importing,
  ImportFailed
};

struct DatasetSourceMetadata
{
  std::string absolutePath;
  std::string projectRelativePath;
  uint64_t fileSize{0};
  int64_t modifiedTime{0};
};

struct DatasetSourceFile
{
  std::string absolutePath;
  std::string projectRelativePath;
  uint64_t fileSize{0};
  int64_t modifiedTime{0};
};

struct Dataset
{
  DatasetID id;
  std::string name;
  DatasetSourceKind sourceKind{DatasetSourceKind::Static};
  std::string importerType{"NONE"};
  DatasetSourceMetadata source;
  DatasetStatus status{DatasetStatus::Missing};
  SceneNodeRef rootNode;
  std::vector<DatasetSourceFile> sourceFiles;
};

namespace dataset {

const char *toString(DatasetSourceKind kind);
const char *toString(DatasetStatus status);
DatasetSourceKind sourceKindFromString(const std::string &s);
DatasetStatus statusFromString(const std::string &s);

} // namespace dataset

} // namespace tsd::scivis_studio
