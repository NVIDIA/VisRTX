// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization/serialization_closure.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
// std
#include <string>
#include <string_view>
#include <vector>

namespace tsd::io {

using namespace tsd::io::detail;

namespace {

// Schemas the object loader recognizes well enough to report an incompatible
// (rather than unknown) schema error.
const std::vector<std::string_view> KNOWN_OBJECT_SCHEMAS = {
    schema::OBJECT_SURFACE,
    schema::OBJECT_VOLUME,
    schema::SCENE_FULL,
    schema::SCENE_CAMERAS_AND_RENDERERS};

anari::DataType rootTypeForSchema(std::string_view schemaName)
{
  if (schemaName == schema::OBJECT_SURFACE)
    return ANARI_SURFACE;
  if (schemaName == schema::OBJECT_VOLUME)
    return ANARI_VOLUME;
  return ANARI_UNKNOWN;
}

std::string schemaForRootType(anari::DataType rootType)
{
  if (rootType == ANARI_SURFACE)
    return std::string(schema::OBJECT_SURFACE);
  if (rootType == ANARI_VOLUME)
    return std::string(schema::OBJECT_VOLUME);
  return {};
}

bool validateObjectGraph(core::DataNode &root,
    anari::DataType rootType,
    std::vector<FileObjectEntry> &entries,
    PayloadValidationResult &result)
{
  auto *objectDB = root.child("objectDB");
  if (!objectDB) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = "object payload requires objectDB";
    return false;
  }

  auto *rootObject = root.child("rootObject");
  if (!rootObject || !rootObject->holdsObjectIdx()) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = "object payload requires rootObject";
    return false;
  }

  anari::DataType declaredRootType = ANARI_UNKNOWN;
  size_t declaredRootIndex = tsd::core::INVALID_INDEX;
  rootObject->getValueAsObjectIdx(&declaredRootType, &declaredRootIndex);
  if (declaredRootType != rootType || declaredRootIndex != 0) {
    result.status = PayloadValidationStatus::IncompatibleSchema;
    result.message = "rootObject must match schema type at local index 0";
    return false;
  }

  const auto policy = objectFilePolicy(rootType);

  bool ok = true;
  objectDB->foreach_child([&](core::DataNode &poolNode) {
    if (!ok)
      return;
    if (!isKnownObjectPoolName(poolNode.name()) && poolNode.numChildren() > 0) {
      result.status = PayloadValidationStatus::IncompatibleSchema;
      result.message = "object payload contains unsupported object pool '"
          + poolNode.name() + "'";
      ok = false;
      return;
    }
    if (isKnownObjectPoolName(poolNode.name())
        && !poolAllowed(policy, poolNode.name())
        && poolNode.numChildren() > 0) {
      result.status = PayloadValidationStatus::IncompatibleSchema;
      result.message = "object payload contains disallowed pool '"
          + poolNode.name() + "'";
      ok = false;
    }
  });
  if (!ok)
    return false;

  if (!collectFileObjects(*objectDB, entries, result))
    return false;

  const auto rootKey = makeKey(rootType, 0);
  if (!findFileEntry(entries, rootKey)) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = "rootObject entry is missing from objectDB";
    return false;
  }

  return checkGraphConsistency(
      entries, {rootKey}, policy, /*requireAllReachable=*/true, result);
}

PayloadValidationResult validateObjectPayloadImpl(core::DataNode &root,
    const std::vector<std::string_view> &acceptedSchemas)
{
  auto result =
      validateEnvelope(root, "object", acceptedSchemas, KNOWN_OBJECT_SCHEMAS);
  if (!result.accepted())
    return result;

  const auto rootType = rootTypeForSchema(result.schema);
  std::vector<FileObjectEntry> entries;
  validateObjectGraph(root, rootType, entries, result);
  return result;
}

Object *importObjectFromTree(Scene &scene, core::DataNode &root)
{
  auto result = validate_ObjectPayload(root);
  if (!result.accepted()) {
    tsd::core::logError(
        "[import_Object] payload validation failed: %s", result.message.c_str());
    return nullptr;
  }

  const auto rootType = rootTypeForSchema(result.schema);
  std::vector<FileObjectEntry> fileEntries;
  if (!validateObjectGraph(root, rootType, fileEntries, result)) {
    tsd::core::logError(
        "[import_Object] payload validation failed: %s", result.message.c_str());
    return nullptr;
  }

  std::vector<TargetObjectEntry> targetEntries;
  std::vector<Any> createdRefs;
  std::string errorMessage;
  if (!instantiateObjectDB(
          scene, fileEntries, targetEntries, createdRefs, errorMessage)) {
    tsd::core::logError("[import_Object] %s", errorMessage.c_str());
    return nullptr;
  }

  const auto rootKey = makeKey(rootType, 0);
  if (auto *entry = findTargetEntry(targetEntries, rootKey))
    return scene.getObject(entry->target);

  return nullptr;
}

} // namespace

bool export_Object(const char *filename, const Object &obj)
{
  if (!filename) {
    tsd::core::logError("[export_Object] filename is null");
    return false;
  }

  if (obj.type() != ANARI_SURFACE && obj.type() != ANARI_VOLUME) {
    tsd::core::logError("[export_Object] unsupported root object type '%s'",
        anari::toString(obj.type()));
    return false;
  }

  auto *scene = obj.scene();
  if (!scene) {
    tsd::core::logError("[export_Object] root object has no owning Scene");
    return false;
  }

  const auto policy = objectFilePolicy(obj.type());
  const auto rootKey = makeKey(obj.type(), obj.index());

  std::vector<ClosureEntry> entries;
  std::string errorMessage;
  if (!buildClosure(*scene,
          {const_cast<Object *>(&obj)},
          policy,
          rootKey,
          entries,
          errorMessage)) {
    tsd::core::logError("[export_Object] %s", errorMessage.c_str());
    return false;
  }

  core::DataTree tree;
  auto &root = tree.root();
  root.reset();

  core::writeDataTreeMetadata(root,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "object",
          schemaForRootType(obj.type()),
          1});

  root["rootObject"] = Any(obj.type(), size_t(0));

  if (!writeObjectDB(root["objectDB"], entries, errorMessage)) {
    tsd::core::logError("[export_Object] %s", errorMessage.c_str());
    return false;
  }

  if (!tree.save(filename)) {
    tsd::core::logError("[export_Object] failed to write file '%s'", filename);
    return false;
  }

  return true;
}

Object *import_Object(Scene &scene, const char *filename)
{
  if (!filename) {
    tsd::core::logError("[import_Object] filename is null");
    return nullptr;
  }

  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError("[import_Object] failed to load file '%s'", filename);
    return nullptr;
  }

  return importObjectFromTree(scene, tree.root());
}

SurfaceRef import_Surface(Scene &scene, const char *filename)
{
  if (!filename) {
    tsd::core::logError("[import_Surface] filename is null");
    return {};
  }

  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError("[import_Surface] failed to load file '%s'", filename);
    return {};
  }

  auto result = validate_SurfacePayload(tree.root());
  if (!result.accepted()) {
    tsd::core::logError("[import_Surface] payload validation failed: %s",
        result.message.c_str());
    return {};
  }

  auto *object = importObjectFromTree(scene, tree.root());
  if (!object || object->type() != ANARI_SURFACE)
    return {};

  return scene.getObject<Surface>(object->index());
}

VolumeRef import_Volume(Scene &scene, const char *filename)
{
  if (!filename) {
    tsd::core::logError("[import_Volume] filename is null");
    return {};
  }

  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError("[import_Volume] failed to load file '%s'", filename);
    return {};
  }

  auto result = validate_VolumePayload(tree.root());
  if (!result.accepted()) {
    tsd::core::logError("[import_Volume] payload validation failed: %s",
        result.message.c_str());
    return {};
  }

  auto *object = importObjectFromTree(scene, tree.root());
  if (!object || object->type() != ANARI_VOLUME)
    return {};

  return scene.getObject<Volume>(object->index());
}

PayloadValidationResult validate_ObjectPayload(core::DataNode &root)
{
  return validateObjectPayloadImpl(
      root, {schema::OBJECT_SURFACE, schema::OBJECT_VOLUME});
}

PayloadValidationResult validate_SurfacePayload(core::DataNode &root)
{
  return validateObjectPayloadImpl(root, {schema::OBJECT_SURFACE});
}

PayloadValidationResult validate_VolumePayload(core::DataNode &root)
{
  return validateObjectPayloadImpl(root, {schema::OBJECT_VOLUME});
}

} // namespace tsd::io
