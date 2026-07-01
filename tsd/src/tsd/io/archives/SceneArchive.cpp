// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/SceneArchive.hpp"
// tsd_animation
#include "tsd/animation/AnimationManager.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/archives/AnimationManagerArchive.hpp"
#include "tsd/io/archives/detail/AnimationRemap.hpp"
#include "tsd/io/archives/detail/ArchiveClosure.hpp"
#include "tsd/io/serialization/Layer.hpp"
#include "tsd/io/serialization/Object.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <algorithm>
#include <array>
#include <functional>
#include <vector>

namespace tsd::io {

namespace {

struct ObjectIndexMapping
{
  detail::ObjectKey source;
  size_t archiveIndex{0};
};

struct LayerNodeIndexMapping
{
  std::string layerName;
  size_t sourceIndex{core::INVALID_INDEX};
  size_t archiveIndex{core::INVALID_INDEX};
};

struct SceneArchiveMappings
{
  std::vector<ObjectIndexMapping> objects;
  std::vector<LayerNodeIndexMapping> layerNodes;
};

using ObjectVisitor = std::function<void(const scene::Object &)>;

struct ObjectPoolDescription
{
  std::string_view name;
  anari::DataType type{ANARI_UNKNOWN};
  void (*visit)(const scene::Scene &, const ObjectVisitor &){nullptr};
};

template <typename OBJECT_POOL_T>
void visitObjectPool(const OBJECT_POOL_T &pool, const ObjectVisitor &visitor)
{
  foreach_item_const(pool, [&](const auto *object) {
    if (object)
      visitor(*object);
  });
}

constexpr std::array<ObjectPoolDescription, 10> OBJECT_POOLS = {{
    {"array",
        ANARI_ARRAY,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().array, visitor);
        }},
    {"sampler",
        ANARI_SAMPLER,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().sampler, visitor);
        }},
    {"material",
        ANARI_MATERIAL,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().material, visitor);
        }},
    {"geometry",
        ANARI_GEOMETRY,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().geometry, visitor);
        }},
    {"surface",
        ANARI_SURFACE,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().surface, visitor);
        }},
    {"spatialfield",
        ANARI_SPATIAL_FIELD,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().field, visitor);
        }},
    {"volume",
        ANARI_VOLUME,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().volume, visitor);
        }},
    {"light",
        ANARI_LIGHT,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().light, visitor);
        }},
    {"camera",
        ANARI_CAMERA,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().camera, visitor);
        }},
    {"renderer",
        ANARI_RENDERER,
        [](const auto &scene, const auto &visitor) {
          visitObjectPool(scene.objectDB().renderer, visitor);
        }},
}};

const ObjectPoolDescription *findObjectPool(std::string_view name)
{
  const auto found = std::find_if(OBJECT_POOLS.begin(),
      OBJECT_POOLS.end(),
      [&](auto &pool) { return pool.name == name; });
  return found == OBJECT_POOLS.end() ? nullptr : &*found;
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
  if (!objectDB) {
    result.status = ArchiveValidationStatus::MissingRequiredNode;
    result.message = "payload requires root/objectDB";
    return false;
  }

  std::vector<PoolDescription> pools;
  bool valid = true;
  objectDB->foreach_child([&](core::DataNode &pool) {
    if (!valid)
      return;
    const auto *poolDescription = findObjectPool(pool.name());
    if (!poolDescription) {
      result.message =
          "Scene Archive contains unknown object pool '" + pool.name() + "'";
      valid = false;
      return;
    }
    const auto expectedType = poolDescription->type;

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

core::DataNode &sceneArchivePayload(core::DataNode &archive)
{
  if (auto *context = archive.child("context"))
    return *context;
  return archive;
}

ArchiveValidationResult validateSceneEnvelope(core::DataNode &archive)
{
  auto &payload = sceneArchivePayload(archive);
  const auto metadataResult = core::readDataTreeMetadata(payload);
  ArchiveValidationResult result;
  if (metadataResult.malformed()) {
    result.status = ArchiveValidationStatus::MalformedMetadata;
    result.message = metadataResult.message;
    return result;
  }

  if (!metadataResult.found()) {
    result.status = ArchiveValidationStatus::MissingMetadataAccepted;
    result.message = "payload has no __tsd_metadata node; treating as legacy";
    return result;
  }

  const auto &metadata = *metadataResult.metadata;
  result.fileType = metadata.fileType;
  result.schema = metadata.schema;
  result.envelopeVersion = metadata.envelopeVersion;
  result.schemaVersion = metadata.schemaVersion;
  if (metadata.envelopeVersion != core::DATA_TREE_METADATA_ENVELOPE_VERSION) {
    result.status = ArchiveValidationStatus::UnsupportedEnvelopeVersion;
    result.message = "expected envelopeVersion 1, got "
        + std::to_string(metadata.envelopeVersion);
  } else if (metadata.schema != schema::SCENE_FULL) {
    result.status = metadata.schema == schema::SCENE_CAMERAS_AND_RENDERERS
            || metadata.schema == schema::SCENE_CAMERAS
            || metadata.schema == schema::SCENE_RENDERERS
        ? ArchiveValidationStatus::IncompatibleSchema
        : ArchiveValidationStatus::UnknownSchema;
    result.message =
        "schema '" + metadata.schema + "' is not accepted by this loader";
  } else if (metadata.schemaVersion != 1) {
    result.status = ArchiveValidationStatus::UnsupportedSchemaVersion;
    result.message = "schema '" + metadata.schema
        + "' supports version 1..1, got "
        + std::to_string(metadata.schemaVersion);
  }
  return result;
}

size_t mappedObjectIndex(const SceneArchiveMappings &mappings,
    anari::DataType type,
    size_t sourceIndex)
{
  const auto source = detail::makeKey(type, sourceIndex);
  const auto found = std::find_if(mappings.objects.begin(),
      mappings.objects.end(),
      [&](const auto &entry) { return detail::sameKey(entry.source, source); });
  return found == mappings.objects.end() ? core::INVALID_INDEX
                                         : found->archiveIndex;
}

size_t mappedLayerNodeIndex(const SceneArchiveMappings &mappings,
    const std::string &layerName,
    size_t sourceIndex)
{
  const auto found = std::find_if(mappings.layerNodes.begin(),
      mappings.layerNodes.end(),
      [&](const auto &entry) {
        return entry.layerName == layerName && entry.sourceIndex == sourceIndex;
      });
  return found == mappings.layerNodes.end() ? core::INVALID_INDEX
                                            : found->archiveIndex;
}

bool serializeSceneArchive(const scene::Scene &scene,
    core::DataNode &archive,
    ArrayDataPolicy arrayData,
    SceneArchiveMappings &mappings)
{
  archive.reset();
  mappings = {};
  core::writeDataTreeMetadata(archive,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene",
          std::string(schema::SCENE_FULL),
          1});

  auto &layers = archive["layers"];
  for (const auto &layerEntry : scene.layers()) {
    if (!layerEntry.second.ptr)
      continue;
    const auto layerName = layerEntry.first.str();
    auto &layerNode = layers[layerName.c_str()];
    const auto &layer = *layerEntry.second.ptr;
    serialize_Layer(layer, layerNode);
    layerNode["isActive"] = layerEntry.second.active;

    size_t archiveIndex = 0;
    layer.traverse_const(layer.root(), [&](const scene::LayerNode &node, int) {
      mappings.layerNodes.push_back({layerName, node.index(), archiveIndex++});
      return true;
    });
  }

  auto &objectDB = archive["objectDB"];
  for (const auto &pool : OBJECT_POOLS) {
    core::DataNode *poolNode = nullptr;
    size_t archiveIndex = 0;
    pool.visit(scene, [&](const scene::Object &object) {
      if (!poolNode)
        poolNode = &objectDB[pool.name];
      serialize_Object(
          object, poolNode->append(), arrayData == ArrayDataPolicy::ProxyOnly);
      mappings.objects.push_back(
          {detail::makeKey(object.type(), object.index()), archiveIndex++});
    });
  }

  std::string message;
  if (!rewriteObjectReferences(objectDB, mappings.objects, message)
      || !rewriteObjectReferences(layers, mappings.objects, message)) {
    core::logError("[serialize_SceneArchive] %s", message.c_str());
    archive.reset();
    mappings = {};
    return false;
  }
  return true;
}

void reconstructSceneArchive(scene::Scene &scene, core::DataNode &sceneArchive)
{
  scene.removeAllObjects();

  auto &payload = sceneArchivePayload(sceneArchive);
  auto &objectDB = *payload.child("objectDB");
  for (const auto &pool : OBJECT_POOLS) {
    if (auto *objects = objectDB.child(pool.name)) {
      objects->foreach_child(
          [&](core::DataNode &object) { deserialize_Object(scene, object); });
    }
  }

  if (auto *layers = payload.child("layers")) {
    layers->foreach_child([&](core::DataNode &layerNode) {
      const core::Token layerName(layerNode.name());
      auto &layer = *scene.addLayer(layerName);
      deserialize_Layer(layerNode, layer, scene);
      const bool active = layerNode.child("isActive")
          ? layerNode["isActive"].getValueOr(true)
          : true;
      scene.setLayerActive(layerName, active);
      scene.signalLayerStructureChanged(&layer);
    });
  }
  scene.signalActiveLayersChanged();
}

} // namespace

bool serialize_SceneArchive(const scene::Scene &scene,
    core::DataNode &archive,
    ArrayDataPolicy arrayData)
{
  SceneArchiveMappings mappings;
  return serializeSceneArchive(scene, archive, arrayData, mappings);
}

bool serialize_SceneAndAnimationManagerArchives(const scene::Scene &scene,
    const animation::AnimationManager &animationManager,
    core::DataNode &sceneArchive,
    core::DataNode &animationManagerArchive,
    ArrayDataPolicy arrayData)
{
  sceneArchive.reset();
  animationManagerArchive.reset();
  if (animationManager.scene() != &scene) {
    core::logError(
        "[serialize_SceneAndAnimationManagerArchives] Animation Manager must "
        "belong to the archived Scene");
    return false;
  }

  SceneArchiveMappings mappings;
  if (!serializeSceneArchive(scene, sceneArchive, arrayData, mappings)
      || !serialize_AnimationManagerArchive(
          animationManager, animationManagerArchive)) {
    sceneArchive.reset();
    animationManagerArchive.reset();
    return false;
  }

  std::string message;
  if (!detail::remapSceneAnimations(
          animationManagerArchive,
          [&](anari::DataType type, size_t index) {
            return mappedObjectIndex(mappings, type, index);
          },
          [&](const std::string &layerName, size_t index) {
            return mappedLayerNodeIndex(mappings, layerName, index);
          },
          message)) {
    core::logError(
        "[serialize_SceneAndAnimationManagerArchives] %s", message.c_str());
    sceneArchive.reset();
    animationManagerArchive.reset();
    return false;
  }
  return true;
}

ArchiveValidationResult validate_SceneArchive(core::DataNode &archive)
{
  auto result = validateSceneEnvelope(archive);
  if (!result.accepted())
    return result;

  auto &payload = sceneArchivePayload(archive);
  if (!validateSceneStructure(payload, result)) {
    if (result.accepted())
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

  reconstructSceneArchive(scene, archive);
  return true;
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
