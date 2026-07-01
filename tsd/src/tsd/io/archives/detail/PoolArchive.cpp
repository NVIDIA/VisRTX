// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/detail/PoolArchive.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization/Object.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <algorithm>
#include <string>
#include <vector>

namespace tsd::io::detail {

namespace {

core::DataNode &payloadRoot(core::DataNode &archive)
{
  if (auto *context = archive.child("context"))
    return *context;
  return archive;
}

template <typename OBJECT_POOL_T>
bool serializePool(core::DataNode &objectDB,
    const OBJECT_POOL_T &pool,
    std::string_view poolName,
    anari::DataType objectType)
{
  std::vector<size_t> sourceIndices;
  core::DataNode *poolNode = nullptr;
  foreach_item_const(pool, [&](const auto *object) {
    if (!object)
      return;
    sourceIndices.push_back(object->index());
    if (!poolNode)
      poolNode = &objectDB[poolName];
    serialize_Object(*object, poolNode->append());
  });

  if (!poolNode)
    return true;

  bool valid = true;
  poolNode->traverse([&](core::DataNode &node, int) {
    if (!valid)
      return false;
    if (node.holdsArray() && anari::isObject(node.arrayType())) {
      valid = false;
      return false;
    }
    if (!node.holdsObjectIdx())
      return true;

    anari::DataType type = ANARI_UNKNOWN;
    size_t index = core::INVALID_INDEX;
    node.getValueAsObjectIdx(&type, &index);
    if (index == core::INVALID_INDEX)
      return true;
    if (type != objectType) {
      valid = false;
      return false;
    }
    const auto found =
        std::find(sourceIndices.begin(), sourceIndices.end(), index);
    if (found == sourceIndices.end()) {
      valid = false;
      return false;
    }
    node.setValue(core::Any(type, size_t(found - sourceIndices.begin())));
    return true;
  });
  return valid;
}

template <typename OBJECT_POOL_T>
void removePool(scene::Scene &scene, const OBJECT_POOL_T &pool)
{
  for (size_t i = pool.capacity(); i-- > 0;) {
    if (auto object = pool.at(i))
      scene.removeObject(object.data());
  }
}

ArchiveValidationResult validationFailure(
    ArchiveValidationStatus status, std::string message)
{
  ArchiveValidationResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

bool knownPoolSchema(std::string_view candidate)
{
  return candidate == schema::SCENE_CAMERAS
      || candidate == schema::SCENE_RENDERERS
      || candidate == schema::SCENE_CAMERAS_AND_RENDERERS
      || candidate == schema::SCENE_FULL;
}

} // namespace

bool serializePoolArchive(const scene::Scene &scene,
    core::DataNode &archive,
    anari::DataType objectType,
    std::string_view poolName,
    std::string_view schemaName)
{
  archive.reset();
  core::writeDataTreeMetadata(archive,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene-subset",
          std::string(schemaName),
          1});
  auto &objectDB = archive["objectDB"];
  const auto &db = scene.objectDB();
  bool valid = false;
  if (objectType == ANARI_CAMERA)
    valid = serializePool(objectDB, db.camera, poolName, objectType);
  else if (objectType == ANARI_RENDERER)
    valid = serializePool(objectDB, db.renderer, poolName, objectType);

  if (!valid) {
    core::logError(
        "[serializePoolArchive] %s pool contains an object "
        "reference outside the Archive",
        std::string(poolName).c_str());
    archive.reset();
  }
  return valid;
}

ArchiveValidationResult validatePoolArchive(core::DataNode &archive,
    anari::DataType objectType,
    std::string_view poolName,
    std::string_view schemaName)
{
  auto &root = payloadRoot(archive);
  ArchiveValidationResult result;
  auto metadataResult = core::readDataTreeMetadata(root);
  if (metadataResult.malformed()) {
    return validationFailure(
        ArchiveValidationStatus::MalformedMetadata, metadataResult.message);
  }
  if (metadataResult.found()) {
    const auto &metadata = *metadataResult.metadata;
    result.fileType = metadata.fileType;
    result.schema = metadata.schema;
    result.envelopeVersion = metadata.envelopeVersion;
    result.schemaVersion = metadata.schemaVersion;
    if (metadata.envelopeVersion != core::DATA_TREE_METADATA_ENVELOPE_VERSION) {
      return validationFailure(
          ArchiveValidationStatus::UnsupportedEnvelopeVersion,
          "unsupported Archive envelope version");
    }
    if (metadata.schema != schemaName
        && metadata.schema != schema::SCENE_CAMERAS_AND_RENDERERS
        && metadata.schema != schema::SCENE_FULL) {
      return validationFailure(knownPoolSchema(metadata.schema)
              ? ArchiveValidationStatus::IncompatibleSchema
              : ArchiveValidationStatus::UnknownSchema,
          "schema '" + metadata.schema + "' is not accepted by this Archive");
    }
    if (metadata.schemaVersion != 1) {
      return validationFailure(
          ArchiveValidationStatus::UnsupportedSchemaVersion,
          "unsupported Archive schema version");
    }
  } else {
    result.status = ArchiveValidationStatus::MissingMetadataAccepted;
    result.message = "metadata-less legacy pool Archive accepted";
  }

  auto *objectDB = root.child("objectDB");
  if (!objectDB) {
    return validationFailure(ArchiveValidationStatus::MissingRequiredNode,
        "pool Archive requires objectDB");
  }

  if (metadataResult.found() && result.schema == schemaName) {
    bool containsOtherPool = false;
    objectDB->foreach_child([&](core::DataNode &node) {
      containsOtherPool |= node.name() != poolName && node.numChildren() > 0;
    });
    if (containsOtherPool) {
      return validationFailure(ArchiveValidationStatus::IncompatibleSchema,
          "split pool Archive contains an unrelated object pool");
    }
  }

  auto *pool = objectDB->child(poolName);
  if (!pool)
    return result;

  size_t expectedIndex = 0;
  bool valid = true;
  std::string message;
  pool->foreach_child([&](core::DataNode &object) {
    if (!valid)
      return;
    auto *self = object.child("self");
    auto *subtype = object.child("subtype");
    if (!self || !self->holdsObjectIdx() || !subtype) {
      valid = false;
      message = "pool object requires self and subtype";
      return;
    }
    anari::DataType type = ANARI_UNKNOWN;
    size_t index = core::INVALID_INDEX;
    self->getValueAsObjectIdx(&type, &index);
    if (type != objectType || index != expectedIndex++) {
      valid = false;
      message = "pool object indices must be dense and type-compatible";
      return;
    }
    const auto *rendererDevice = object.child("rendererDeviceName");
    if (objectType == ANARI_RENDERER
        && (!rendererDevice
            || rendererDevice->getValueOr<std::string>("").empty())) {
      valid = false;
      message = "renderer Archive object requires rendererDeviceName";
    }
  });
  if (!valid) {
    return validationFailure(
        ArchiveValidationStatus::IncompatibleSchema, std::move(message));
  }

  const auto poolSize = pool->numChildren();
  pool->traverse([&](core::DataNode &node, int) {
    if (!valid)
      return false;
    if (node.holdsArray() && anari::isObject(node.arrayType())) {
      valid = false;
      message = "pool Archive cannot contain object arrays";
      return false;
    }
    if (!node.holdsObjectIdx())
      return true;
    anari::DataType type = ANARI_UNKNOWN;
    size_t index = core::INVALID_INDEX;
    node.getValueAsObjectIdx(&type, &index);
    if (index != core::INVALID_INDEX
        && (type != objectType || index >= poolSize)) {
      valid = false;
      message = "pool Archive contains an unresolved object reference";
      return false;
    }
    return true;
  });
  if (!valid) {
    return validationFailure(
        ArchiveValidationStatus::IncompatibleSchema, std::move(message));
  }
  return result;
}

bool deserializePoolArchive(scene::Scene &scene,
    core::DataNode &archive,
    anari::DataType objectType,
    std::string_view poolName,
    std::string_view schemaName,
    ArchiveValidationResult *validation)
{
  auto result = validatePoolArchive(archive, objectType, poolName, schemaName);
  if (validation)
    *validation = result;
  if (!result.accepted())
    return false;

  const auto &db = scene.objectDB();
  if (objectType == ANARI_CAMERA)
    removePool(scene, db.camera);
  else if (objectType == ANARI_RENDERER)
    removePool(scene, db.renderer);
  else
    return false;

  auto &root = payloadRoot(archive);
  auto *pool = root["objectDB"].child(poolName);
  if (pool) {
    pool->foreach_child(
        [&](core::DataNode &object) { deserialize_Object(scene, object); });
  }
  return true;
}

} // namespace tsd::io::detail
