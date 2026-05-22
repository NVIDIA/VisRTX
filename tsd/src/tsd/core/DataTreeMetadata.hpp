// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/DataTree.hpp"
// std
#include <optional>
#include <string>

namespace tsd::core {

inline constexpr const char *DATA_TREE_METADATA_NODE = "__tsd_metadata";
inline constexpr int DATA_TREE_METADATA_ENVELOPE_VERSION = 1;

struct DataTreeMetadata
{
  int envelopeVersion{DATA_TREE_METADATA_ENVELOPE_VERSION};
  std::string fileType;
  std::string schema;
  int schemaVersion{1};
};

enum class DataTreeMetadataReadStatus
{
  Found,
  Missing,
  Malformed
};

struct DataTreeMetadataReadResult
{
  DataTreeMetadataReadStatus status{DataTreeMetadataReadStatus::Missing};
  std::optional<DataTreeMetadata> metadata;
  std::string message;

  bool found() const;
  bool malformed() const;
};

inline bool DataTreeMetadataReadResult::found() const
{
  return status == DataTreeMetadataReadStatus::Found;
}

inline bool DataTreeMetadataReadResult::malformed() const
{
  return status == DataTreeMetadataReadStatus::Malformed;
}

inline void writeDataTreeMetadata(
    DataNode &root, const DataTreeMetadata &metadata)
{
  auto &metadataNode = root[DATA_TREE_METADATA_NODE];
  metadataNode["envelopeVersion"] = metadata.envelopeVersion;
  metadataNode["fileType"] = metadata.fileType;
  metadataNode["schema"] = metadata.schema;
  metadataNode["schemaVersion"] = metadata.schemaVersion;
}

inline DataTreeMetadataReadResult readDataTreeMetadata(const DataNode &root)
{
  auto *metadataNode = root.child(DATA_TREE_METADATA_NODE);
  if (!metadataNode)
    return {};

  auto requiredNode = [&](const char *name, anari::DataType type)
      -> const DataNode * {
    auto *node = metadataNode->child(name);
    if (!node)
      return nullptr;

    const auto actualType = node->getValue().type();
    if (actualType != type)
      return nullptr;

    return node;
  };

  auto malformed = [](std::string message) {
    DataTreeMetadataReadResult result;
    result.status = DataTreeMetadataReadStatus::Malformed;
    result.message = std::move(message);
    return result;
  };

  auto describeMissingOrWrongType =
      [&](const char *name, anari::DataType type) -> std::string {
    std::string message = std::string(DATA_TREE_METADATA_NODE) + "/" + name
        + " must be " + anari::toString(type);
    if (auto *node = metadataNode->child(name)) {
      message += ", got ";
      message += anari::toString(node->getValue().type());
    } else
      message += ", but is missing";
    return message;
  };

  auto *envelopeVersion = requiredNode("envelopeVersion", ANARI_INT32);
  if (!envelopeVersion)
    return malformed(describeMissingOrWrongType("envelopeVersion", ANARI_INT32));

  auto *fileType = requiredNode("fileType", ANARI_STRING);
  if (!fileType)
    return malformed(describeMissingOrWrongType("fileType", ANARI_STRING));

  auto *schema = requiredNode("schema", ANARI_STRING);
  if (!schema)
    return malformed(describeMissingOrWrongType("schema", ANARI_STRING));

  auto *schemaVersion = requiredNode("schemaVersion", ANARI_INT32);
  if (!schemaVersion)
    return malformed(describeMissingOrWrongType("schemaVersion", ANARI_INT32));

  DataTreeMetadata metadata;
  metadata.envelopeVersion = envelopeVersion->getValueAs<int>();
  metadata.fileType = fileType->getValueAs<std::string>();
  metadata.schema = schema->getValueAs<std::string>();
  metadata.schemaVersion = schemaVersion->getValueAs<int>();

  DataTreeMetadataReadResult result;
  result.status = DataTreeMetadataReadStatus::Found;
  result.metadata = std::move(metadata);
  return result;
}

} // namespace tsd::core
