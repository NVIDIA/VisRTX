// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/SceneArchive.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/archives/detail/ArchiveClosure.hpp"
#include "tsd/io/serialization/Layer.hpp"
#include "tsd/io/serialization/Object.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <algorithm>
#include <vector>

namespace tsd::io {

namespace {

struct ObjectIndexMapping
{
  detail::ObjectKey source;
  size_t archiveIndex{0};
};

template <typename OBJECT_POOL_T>
void serializeObjectPool(core::DataNode &objectDB,
    const OBJECT_POOL_T &pool,
    const char *name,
    ArrayDataPolicy arrayData)
{
  core::DataNode *poolNode = nullptr;
  foreach_item_const(pool, [&](const auto *object) {
    if (!object)
      return;
    if (!poolNode)
      poolNode = &objectDB[name];
    serialize_Object(
        *object, poolNode->append(), arrayData == ArrayDataPolicy::ProxyOnly);
  });
}

template <typename OBJECT_POOL_T>
void appendPoolMappings(
    const OBJECT_POOL_T &pool, std::vector<ObjectIndexMapping> &mappings)
{
  size_t archiveIndex = 0;
  foreach_item_const(pool, [&](const auto *object) {
    if (!object)
      return;
    mappings.push_back(
        {detail::makeKey(object->type(), object->index()), archiveIndex++});
  });
}

std::vector<ObjectIndexMapping> buildObjectMappings(const scene::Scene &scene)
{
  std::vector<ObjectIndexMapping> mappings;
  const auto &db = scene.objectDB();
  appendPoolMappings(db.array, mappings);
  appendPoolMappings(db.sampler, mappings);
  appendPoolMappings(db.material, mappings);
  appendPoolMappings(db.geometry, mappings);
  appendPoolMappings(db.surface, mappings);
  appendPoolMappings(db.field, mappings);
  appendPoolMappings(db.volume, mappings);
  appendPoolMappings(db.light, mappings);
  appendPoolMappings(db.camera, mappings);
  appendPoolMappings(db.renderer, mappings);
  return mappings;
}

bool rewriteObjectReferences(core::DataNode &root,
    const std::vector<ObjectIndexMapping> &mappings,
    std::string &message)
{
  bool valid = true;
  root.traverse([&](core::DataNode &node, int) {
    if (!valid)
      return false;
    if (node.holdsArray() && anari::isObject(node.arrayType())) {
      anari::DataType type = ANARI_UNKNOWN;
      const void *data = nullptr;
      size_t size = 0;
      node.getValueAsArray(&type, &data, &size);
      const auto *sourceIndices = static_cast<const size_t *>(data);
      std::vector<size_t> archiveIndices(size);
      for (size_t i = 0; i < size; ++i) {
        if (sourceIndices[i] == core::INVALID_INDEX) {
          archiveIndices[i] = core::INVALID_INDEX;
          continue;
        }
        const auto source = detail::makeKey(type, sourceIndices[i]);
        const auto mapping = std::find_if(
            mappings.begin(), mappings.end(), [&](const auto &entry) {
              return detail::sameKey(entry.source, source);
            });
        if (mapping == mappings.end()) {
          message = "serialized object array has no Scene Archive mapping";
          valid = false;
          return false;
        }
        archiveIndices[i] = mapping->archiveIndex;
      }
      node.setValueAsArray(type, archiveIndices.data(), archiveIndices.size());
      return true;
    }
    if (!node.holdsObjectIdx())
      return true;

    anari::DataType type = ANARI_UNKNOWN;
    size_t index = tsd::core::INVALID_INDEX;
    node.getValueAsObjectIdx(&type, &index);
    const auto source = detail::makeKey(type, index);
    const auto mapping =
        std::find_if(mappings.begin(), mappings.end(), [&](const auto &entry) {
          return detail::sameKey(entry.source, source);
        });
    if (mapping == mappings.end()) {
      message = "serialized object reference has no Scene Archive mapping";
      valid = false;
      return false;
    }

    node.setValue(core::Any(type, mapping->archiveIndex));
    return true;
  });
  return valid;
}

struct PoolDescription
{
  std::string_view name;
  anari::DataType type{ANARI_UNKNOWN};
  size_t size{0};
};

anari::DataType poolType(std::string_view name)
{
  if (name == "array")
    return ANARI_ARRAY;
  if (name == "sampler")
    return ANARI_SAMPLER;
  if (name == "material")
    return ANARI_MATERIAL;
  if (name == "geometry")
    return ANARI_GEOMETRY;
  if (name == "surface")
    return ANARI_SURFACE;
  if (name == "spatialfield")
    return ANARI_SPATIAL_FIELD;
  if (name == "volume")
    return ANARI_VOLUME;
  if (name == "light")
    return ANARI_LIGHT;
  if (name == "camera")
    return ANARI_CAMERA;
  if (name == "renderer")
    return ANARI_RENDERER;
  return ANARI_UNKNOWN;
}

const PoolDescription *findPool(
    const std::vector<PoolDescription> &pools, anari::DataType type)
{
  const auto canonical = detail::canonicalObjectType(type);
  const auto found = std::find_if(pools.begin(), pools.end(), [&](auto &pool) {
    return pool.type == canonical;
  });
  return found == pools.end() ? nullptr : &*found;
}

bool validateLayerNode(core::DataNode &node, std::string &message)
{
  auto *children = node.child("children");
  if (!node.child("name") || !node.child("value") || !node.child("enabled")
      || !children) {
    message = "Scene Archive layer node is missing required state";
    return false;
  }
  bool valid = true;
  children->foreach_child([&](core::DataNode &child) {
    if (valid)
      valid = validateLayerNode(child, message);
  });
  return valid;
}

bool validateSceneStructure(
    core::DataNode &archive, ArchiveValidationResult &result)
{
  auto *objectDB = archive.child("objectDB");
  if (!objectDB)
    return false;

  std::vector<PoolDescription> pools;
  bool valid = true;
  objectDB->foreach_child([&](core::DataNode &pool) {
    if (!valid)
      return;
    const auto expectedType = poolType(pool.name());
    if (expectedType == ANARI_UNKNOWN) {
      result.message =
          "Scene Archive contains unknown object pool '" + pool.name() + "'";
      valid = false;
      return;
    }

    size_t expectedIndex = 0;
    pool.foreach_child([&](core::DataNode &object) {
      if (!valid)
        return;
      auto *self = object.child("self");
      auto *subtype = object.child("subtype");
      if (!self || !self->holdsObjectIdx() || !subtype) {
        result.message = "Scene Archive object requires self and subtype";
        valid = false;
        return;
      }
      anari::DataType type = ANARI_UNKNOWN;
      size_t index = core::INVALID_INDEX;
      self->getValueAsObjectIdx(&type, &index);
      if (detail::canonicalObjectType(type) != expectedType
          || index != expectedIndex++) {
        result.message =
            "Scene Archive object pools must use dense compatible indices";
        valid = false;
        return;
      }
      if (expectedType == ANARI_ARRAY
          && (!object.child("arrayDim") || !object.child("arrayData"))) {
        result.message = "Scene Archive array is missing storage state";
        valid = false;
      }
      if (expectedType == ANARI_RENDERER) {
        auto *device = object.child("rendererDeviceName");
        if (!device || device->getValueOr<std::string>("").empty()) {
          result.message = "Scene Archive renderer requires a device name";
          valid = false;
        }
      }
    });
    pools.push_back({pool.name(), expectedType, expectedIndex});
  });
  if (!valid)
    return false;

  if (auto *layers = archive.child("layers")) {
    layers->foreach_child([&](core::DataNode &layer) {
      if (valid)
        valid = validateLayerNode(layer, result.message);
    });
  }
  if (!valid)
    return false;

  archive.traverse([&](core::DataNode &node, int) {
    if (!valid)
      return false;
    auto validateIndex = [&](anari::DataType type, size_t index) {
      if (index == core::INVALID_INDEX)
        return true;
      const auto *pool = findPool(pools, type);
      return pool && index < pool->size;
    };

    if (node.holdsArray() && anari::isObject(node.arrayType())) {
      anari::DataType type = ANARI_UNKNOWN;
      const void *data = nullptr;
      size_t size = 0;
      node.getValueAsArray(&type, &data, &size);
      const auto *indices = static_cast<const size_t *>(data);
      for (size_t i = 0; i < size; ++i) {
        if (!validateIndex(type, indices[i])) {
          result.message =
              "Scene Archive contains an unresolved object-array reference";
          valid = false;
          return false;
        }
      }
      return true;
    }
    if (!node.holdsObjectIdx())
      return true;
    anari::DataType type = ANARI_UNKNOWN;
    size_t index = core::INVALID_INDEX;
    node.getValueAsObjectIdx(&type, &index);
    if (!validateIndex(type, index)) {
      result.message = "Scene Archive contains an unresolved object reference";
      valid = false;
      return false;
    }
    return true;
  });
  return valid;
}

} // namespace

bool serialize_SceneArchive(const scene::Scene &scene,
    core::DataNode &archive,
    ArrayDataPolicy arrayData)
{
  archive.reset();
  core::writeDataTreeMetadata(archive,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene",
          std::string(schema::SCENE_FULL),
          1});

  auto &layers = archive["layers"];
  for (const auto &layer : scene.layers()) {
    if (!layer.second.ptr)
      continue;
    auto &layerNode = layers[layer.first.c_str()];
    serialize_Layer(*layer.second.ptr, layerNode);
    layerNode["isActive"] = layer.second.active;
  }

  auto &objectDB = archive["objectDB"];
  const auto &db = scene.objectDB();
  serializeObjectPool(objectDB, db.array, "array", arrayData);
  serializeObjectPool(objectDB, db.sampler, "sampler", arrayData);
  serializeObjectPool(objectDB, db.material, "material", arrayData);
  serializeObjectPool(objectDB, db.geometry, "geometry", arrayData);
  serializeObjectPool(objectDB, db.surface, "surface", arrayData);
  serializeObjectPool(objectDB, db.field, "spatialfield", arrayData);
  serializeObjectPool(objectDB, db.volume, "volume", arrayData);
  serializeObjectPool(objectDB, db.light, "light", arrayData);
  serializeObjectPool(objectDB, db.camera, "camera", arrayData);
  serializeObjectPool(objectDB, db.renderer, "renderer", arrayData);

  const auto mappings = buildObjectMappings(scene);
  std::string message;
  if (!rewriteObjectReferences(objectDB, mappings, message)
      || !rewriteObjectReferences(layers, mappings, message)) {
    core::logError("[serialize_SceneArchive] %s", message.c_str());
    archive.reset();
    return false;
  }
  return true;
}

ArchiveValidationResult validate_SceneArchive(core::DataNode &archive)
{
  auto result = validate_ScenePayload(archive);
  if (!result.accepted())
    return result;

  auto *context = archive.child("context");
  auto &payload = context ? *context : archive;
  if (!validateSceneStructure(payload, result)) {
    result.status = ArchiveValidationStatus::IncompatibleSchema;
    if (result.message.empty())
      result.message = "Scene Archive structure is invalid";
  }
  return result;
}

bool deserialize_SceneArchive(scene::Scene &scene,
    core::DataNode &archive,
    ArchiveValidationResult *validation)
{
  const auto archiveValidation = validate_SceneArchive(archive);
  if (validation)
    *validation = archiveValidation;
  if (!archiveValidation.accepted())
    return false;

  PayloadValidationResult legacyValidation;
  const bool loaded = tryLoad_Scene(scene, archive, &legacyValidation, nullptr);
  return loaded;
}

bool save_SceneArchive(
    const scene::Scene &scene, const char *filename, ArrayDataPolicy arrayData)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return serialize_SceneArchive(scene, tree.root(), arrayData)
      && tree.save(filename);
}

bool load_SceneArchive(scene::Scene &scene,
    const char *filename,
    ArchiveValidationResult *validation)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return tree.load(filename)
      && deserialize_SceneArchive(scene, tree.root(), validation);
}

} // namespace tsd::io
