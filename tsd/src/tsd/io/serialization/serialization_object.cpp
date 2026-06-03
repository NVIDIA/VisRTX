// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/serialization.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <array>
#include <exception>
#include <string>
#include <vector>

namespace tsd::io {

namespace {

using tsd::core::Any;

struct ObjectKey
{
  anari::DataType type{ANARI_UNKNOWN};
  size_t index{tsd::core::INVALID_INDEX};
};

struct ClosureEntry
{
  ObjectKey source;
  anari::DataType objectType{ANARI_UNKNOWN};
  size_t localIndex{tsd::core::INVALID_INDEX};
  Object *object{nullptr};
};

struct FileObjectEntry
{
  ObjectKey file;
  anari::DataType objectType{ANARI_UNKNOWN};
  core::DataNode *node{nullptr};
};

struct TargetObjectEntry
{
  ObjectKey file;
  Any target;
};

constexpr std::array<const char *, 7> OBJECT_POOL_NAMES = {
    "array",
    "sampler",
    "material",
    "geometry",
    "surface",
    "spatialfield",
    "volume",
};

anari::DataType canonicalObjectType(anari::DataType type)
{
  return anari::isArray(type) ? ANARI_ARRAY : type;
}

ObjectKey makeKey(anari::DataType type, size_t index)
{
  return {canonicalObjectType(type), index};
}

ObjectKey makeKey(const Any &value)
{
  return makeKey(value.type(), value.getAsObjectIndex());
}

bool sameKey(const ObjectKey &a, const ObjectKey &b)
{
  return a.type == b.type && a.index == b.index;
}

anari::DataType nonArrayTypeForPoolName(std::string_view name)
{
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
  return ANARI_UNKNOWN;
}

bool isKnownObjectPoolName(std::string_view name)
{
  return std::find(OBJECT_POOL_NAMES.begin(), OBJECT_POOL_NAMES.end(), name)
      != OBJECT_POOL_NAMES.end();
}

bool poolAllowedForRoot(anari::DataType rootType, std::string_view poolName)
{
  if (poolName == "array" || poolName == "sampler")
    return true;

  if (rootType == ANARI_SURFACE)
    return poolName == "surface" || poolName == "geometry"
        || poolName == "material";

  if (rootType == ANARI_VOLUME)
    return poolName == "volume" || poolName == "spatialfield";

  return false;
}

bool typeAllowedForRoot(anari::DataType rootType, anari::DataType type)
{
  if (anari::isArray(type) || type == ANARI_SAMPLER)
    return true;

  if (rootType == ANARI_SURFACE)
    return type == ANARI_SURFACE || type == ANARI_GEOMETRY
        || type == ANARI_MATERIAL;

  if (rootType == ANARI_VOLUME)
    return type == ANARI_VOLUME || type == ANARI_SPATIAL_FIELD;

  return false;
}

size_t *counterForType(anari::DataType type,
    size_t &arrays,
    size_t &samplers,
    size_t &materials,
    size_t &geometries,
    size_t &surfaces,
    size_t &fields,
    size_t &volumes)
{
  switch (canonicalObjectType(type)) {
  case ANARI_ARRAY:
    return &arrays;
  case ANARI_SAMPLER:
    return &samplers;
  case ANARI_MATERIAL:
    return &materials;
  case ANARI_GEOMETRY:
    return &geometries;
  case ANARI_SURFACE:
    return &surfaces;
  case ANARI_SPATIAL_FIELD:
    return &fields;
  case ANARI_VOLUME:
    return &volumes;
  default:
    return nullptr;
  }
}

ClosureEntry *findEntry(std::vector<ClosureEntry> &entries, const ObjectKey &key)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return sameKey(entry.source, key);
  });
  return it == entries.end() ? nullptr : &*it;
}

const ClosureEntry *findEntry(
    const std::vector<ClosureEntry> &entries, const ObjectKey &key)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return sameKey(entry.source, key);
  });
  return it == entries.end() ? nullptr : &*it;
}

FileObjectEntry *findEntry(
    std::vector<FileObjectEntry> &entries, const ObjectKey &key)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return sameKey(entry.file, key);
  });
  return it == entries.end() ? nullptr : &*it;
}

const FileObjectEntry *findEntry(
    const std::vector<FileObjectEntry> &entries, const ObjectKey &key)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return sameKey(entry.file, key);
  });
  return it == entries.end() ? nullptr : &*it;
}

const TargetObjectEntry *findEntry(
    const std::vector<TargetObjectEntry> &entries, const ObjectKey &key)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return sameKey(entry.file, key);
  });
  return it == entries.end() ? nullptr : &*it;
}

bool hasObjectArrayNode(core::DataNode &node, std::string *message)
{
  bool found = false;
  node.traverse([&](core::DataNode &n, int) {
    if (n.holdsArray() && anari::isObject(n.arrayType())) {
      if (message) {
        *message = "object array values are not supported in object files";
      }
      found = true;
      return false;
    }
    return true;
  });
  return found;
}

bool addReferencedObject(const Scene &scene,
    anari::DataType rootType,
    const ObjectKey &rootKey,
    const Any &value,
    std::vector<ClosureEntry> &entries,
    size_t &arrays,
    size_t &samplers,
    size_t &materials,
    size_t &geometries,
    size_t &surfaces,
    size_t &fields,
    size_t &volumes,
    std::string &errorMessage)
{
  if (!value.holdsObject())
    return true;

  const auto key = makeKey(value);
  if (findEntry(entries, key))
    return true;

  auto *object = scene.getObject(value.type(), value.getAsObjectIndex());
  if (!object) {
    errorMessage = "referenced object ";
    errorMessage += anari::toString(value.type());
    errorMessage += " @";
    errorMessage += std::to_string(value.getAsObjectIndex());
    errorMessage += " is missing";
    return false;
  }

  const auto objectType = object->type();
  if (!typeAllowedForRoot(rootType, objectType)) {
    errorMessage = "object closure reached unsupported type ";
    errorMessage += anari::toString(objectType);
    return false;
  }

  if ((objectType == ANARI_SURFACE || objectType == ANARI_VOLUME)
      && !sameKey(key, rootKey)) {
    errorMessage = "object files support only one root ";
    errorMessage += anari::toString(rootType);
    return false;
  }

  if (anari::isArray(objectType)) {
    auto *array = static_cast<Array *>(object);
    if (array->isProxy()) {
      errorMessage = "object files cannot export proxy arrays";
      return false;
    }
    if (anari::isObject(array->elementType())) {
      errorMessage = "object files cannot export arrays of ANARI objects";
      return false;
    }
  }

  auto *counter = counterForType(
      objectType, arrays, samplers, materials, geometries, surfaces, fields, volumes);
  if (!counter) {
    errorMessage = "object closure reached unsupported type ";
    errorMessage += anari::toString(objectType);
    return false;
  }

  entries.push_back({key, objectType, (*counter)++, object});
  return true;
}

bool addReferencesFromAny(const Scene &scene,
    anari::DataType rootType,
    const ObjectKey &rootKey,
    const Any &value,
    std::vector<ClosureEntry> &entries,
    size_t &arrays,
    size_t &samplers,
    size_t &materials,
    size_t &geometries,
    size_t &surfaces,
    size_t &fields,
    size_t &volumes,
    std::string &errorMessage)
{
  if (!value.holdsObject())
    return true;

  return addReferencedObject(scene,
      rootType,
      rootKey,
      value,
      entries,
      arrays,
      samplers,
      materials,
      geometries,
      surfaces,
      fields,
      volumes,
      errorMessage);
}

bool buildExportClosure(const Object &root,
    std::vector<ClosureEntry> &entries,
    std::string &errorMessage)
{
  const auto rootType = root.type();
  auto *scene = root.scene();
  if (!scene) {
    errorMessage = "root object has no owning Scene";
    return false;
  }

  size_t arrays = 0;
  size_t samplers = 0;
  size_t materials = 0;
  size_t geometries = 0;
  size_t surfaces = 0;
  size_t fields = 0;
  size_t volumes = 0;

  const auto rootKey = makeKey(rootType, root.index());
  entries.push_back({rootKey, rootType, 0, const_cast<Object *>(&root)});
  if (rootType == ANARI_SURFACE)
    surfaces = 1;
  else if (rootType == ANARI_VOLUME)
    volumes = 1;

  for (size_t i = 0; i < entries.size(); i++) {
    auto *object = entries[i].object;

    if (anari::isArray(object->type())) {
      auto *array = static_cast<Array *>(object);
      if (array->isProxy()) {
        errorMessage = "object files cannot export proxy arrays";
        return false;
      }
      if (anari::isObject(array->elementType())) {
        errorMessage = "object files cannot export arrays of ANARI objects";
        return false;
      }
    }

    core::DataTree scratchTree;
    objectToNode(*object, scratchTree.root(), false);
    if (hasObjectArrayNode(scratchTree.root(), &errorMessage))
      return false;

    for (size_t p = 0; p < object->numParameters(); p++) {
      const auto &param = object->parameterAt(p);
      if (!addReferencesFromAny(*scene,
              rootType,
              rootKey,
              param.value(),
              entries,
              arrays,
              samplers,
              materials,
              geometries,
              surfaces,
              fields,
              volumes,
              errorMessage))
        return false;
      if (param.hasMin()
          && !addReferencesFromAny(*scene,
              rootType,
              rootKey,
              param.min(),
              entries,
              arrays,
              samplers,
              materials,
              geometries,
              surfaces,
              fields,
              volumes,
              errorMessage))
        return false;
      if (param.hasMax()
          && !addReferencesFromAny(*scene,
              rootType,
              rootKey,
              param.max(),
              entries,
              arrays,
              samplers,
              materials,
              geometries,
              surfaces,
              fields,
              volumes,
              errorMessage))
        return false;
    }

    for (size_t m = 0; m < object->numMetadata(); m++) {
      const auto *name = object->getMetadataName(m);
      anari::DataType arrayType = ANARI_UNKNOWN;
      const void *arrayPtr = nullptr;
      size_t arraySize = 0;
      object->getMetadataArray(name, &arrayType, &arrayPtr, &arraySize);
      if (anari::isObject(arrayType)) {
        errorMessage = "object-valued metadata arrays are not supported";
        return false;
      }
      if (arrayType != ANARI_UNKNOWN)
        continue;

      if (!addReferencesFromAny(*scene,
              rootType,
              rootKey,
              object->getMetadataValue(name),
              entries,
              arrays,
              samplers,
              materials,
              geometries,
              surfaces,
              fields,
              volumes,
              errorMessage))
        return false;
    }
  }

  return true;
}

bool rewriteObjectReferences(core::DataNode &root,
    const std::vector<ClosureEntry> &entries,
    std::string &errorMessage)
{
  bool ok = true;
  root.traverse([&](core::DataNode &node, int) {
    if (!ok)
      return false;

    if (node.holdsArray() && anari::isObject(node.arrayType())) {
      errorMessage = "object array values are not supported in object files";
      ok = false;
      return false;
    }

    if (!node.holdsObjectIdx())
      return true;

    anari::DataType type = ANARI_UNKNOWN;
    size_t index = tsd::core::INVALID_INDEX;
    node.getValueAsObjectIdx(&type, &index);

    auto *entry = findEntry(entries, makeKey(type, index));
    if (!entry) {
      errorMessage = "serialized object reference has no closure mapping";
      ok = false;
      return false;
    }

    node.setValue(Any(type, entry->localIndex));
    return true;
  });
  return ok;
}

bool rewriteObjectReferences(core::DataNode &root,
    const std::vector<TargetObjectEntry> &entries,
    std::string &errorMessage)
{
  bool ok = true;
  root.traverse([&](core::DataNode &node, int) {
    if (!ok)
      return false;

    if (node.holdsArray() && anari::isObject(node.arrayType())) {
      errorMessage = "object array values are not supported in object files";
      ok = false;
      return false;
    }

    if (!node.holdsObjectIdx())
      return true;

    anari::DataType type = ANARI_UNKNOWN;
    size_t index = tsd::core::INVALID_INDEX;
    node.getValueAsObjectIdx(&type, &index);

    auto *entry = findEntry(entries, makeKey(type, index));
    if (!entry) {
      errorMessage = "serialized object reference has no import mapping";
      ok = false;
      return false;
    }

    node.setValue(Any(type, entry->target.getAsObjectIndex()));
    return true;
  });
  return ok;
}

const ClosureEntry *entryForLocalIndex(const std::vector<ClosureEntry> &entries,
    anari::DataType type,
    size_t localIndex)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return canonicalObjectType(entry.objectType) == canonicalObjectType(type)
        && entry.localIndex == localIndex;
  });
  return it == entries.end() ? nullptr : &*it;
}

PayloadValidationResult makeMetadataFailure(
    PayloadValidationStatus status, std::string message)
{
  PayloadValidationResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

PayloadValidationResult validateObjectMetadata(core::DataNode &root,
    const std::vector<std::string_view> &acceptedSchemas)
{
  auto metadataResult = core::readDataTreeMetadata(root);
  if (metadataResult.malformed()) {
    auto result = makeMetadataFailure(
        PayloadValidationStatus::MalformedMetadata, metadataResult.message);
    return result;
  }

  if (!metadataResult.found()) {
    return makeMetadataFailure(PayloadValidationStatus::MissingRequiredNode,
        "object payload requires __tsd_metadata");
  }

  PayloadValidationResult result;
  const auto &metadata = *metadataResult.metadata;
  result.fileType = metadata.fileType;
  result.schema = metadata.schema;
  result.envelopeVersion = metadata.envelopeVersion;
  result.schemaVersion = metadata.schemaVersion;

  if (metadata.envelopeVersion != core::DATA_TREE_METADATA_ENVELOPE_VERSION) {
    result.status = PayloadValidationStatus::UnsupportedEnvelopeVersion;
    result.message = "expected envelopeVersion 1, got "
        + std::to_string(metadata.envelopeVersion);
    return result;
  }

  if (metadata.fileType != "object") {
    result.status = PayloadValidationStatus::IncompatibleSchema;
    result.message = "fileType '" + metadata.fileType
        + "' is not accepted by object import";
    return result;
  }

  const auto schemaMatches = [&](std::string_view schema) {
    return metadata.schema == schema;
  };

  if (std::none_of(
          acceptedSchemas.begin(), acceptedSchemas.end(), schemaMatches)) {
    result.status =
        metadata.schema == schema::OBJECT_SURFACE
            || metadata.schema == schema::OBJECT_VOLUME
            || metadata.schema == schema::SCENE_FULL
            || metadata.schema == schema::SCENE_CAMERAS_AND_RENDERERS
        ? PayloadValidationStatus::IncompatibleSchema
        : PayloadValidationStatus::UnknownSchema;
    result.message =
        "schema '" + metadata.schema + "' is not accepted by this loader";
    return result;
  }

  if (metadata.schemaVersion != 1) {
    result.status = PayloadValidationStatus::UnsupportedSchemaVersion;
    result.message = "schema '" + metadata.schema
        + "' supports version 1..1, got "
        + std::to_string(metadata.schemaVersion);
    return result;
  }

  return result;
}

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

bool collectFileObjects(core::DataNode &objectDB,
    std::vector<FileObjectEntry> &entries,
    PayloadValidationResult &result)
{
  for (auto poolName : OBJECT_POOL_NAMES) {
    auto *poolNode = objectDB.child(poolName);
    if (!poolNode)
      continue;

    const auto expectedType = nonArrayTypeForPoolName(poolName);
    size_t expectedIndex = 0;
    bool ok = true;
    poolNode->foreach_child([&](core::DataNode &objectNode) {
      if (!ok)
        return;

      auto *selfNode = objectNode.child("self");
      if (!selfNode || !selfNode->holdsObjectIdx()) {
        result.status = PayloadValidationStatus::MalformedMetadata;
        result.message = std::string("objectDB/") + poolName
            + " entry is missing object self";
        ok = false;
        return;
      }

      anari::DataType objectType = ANARI_UNKNOWN;
      size_t index = tsd::core::INVALID_INDEX;
      selfNode->getValueAsObjectIdx(&objectType, &index);

      if (std::string_view(poolName) == "array") {
        if (!anari::isArray(objectType)) {
          result.status = PayloadValidationStatus::MalformedMetadata;
          result.message = "objectDB/array entry has non-array self type";
          ok = false;
          return;
        }
      } else if (objectType != expectedType) {
        result.status = PayloadValidationStatus::MalformedMetadata;
        result.message = std::string("objectDB/") + poolName
            + " entry self type does not match its pool";
        ok = false;
        return;
      }

      if (index != expectedIndex) {
        result.status = PayloadValidationStatus::MalformedMetadata;
        result.message = std::string("objectDB/") + poolName
            + " entries must use dense local indices";
        ok = false;
        return;
      }

      auto *subtypeNode = objectNode.child("subtype");
      if (!subtypeNode || subtypeNode->getValue().type() != ANARI_STRING) {
        result.status = PayloadValidationStatus::MalformedMetadata;
        result.message = std::string("objectDB/") + poolName
            + " entry is missing subtype";
        ok = false;
        return;
      }

      if (anari::isArray(objectType)) {
        auto *arrayDim = objectNode.child("arrayDim");
        auto *arrayData = objectNode.child("arrayData");
        if (!arrayDim || !arrayData) {
          result.status = PayloadValidationStatus::MissingRequiredNode;
          result.message = "array entries require arrayDim and arrayData";
          ok = false;
          return;
        }
        if (arrayDim->getValue().type() != ANARI_UINT32_VEC3) {
          result.status = PayloadValidationStatus::MalformedMetadata;
          result.message = "arrayDim must be uint3";
          ok = false;
          return;
        }
        if (!arrayData->holdsArray()) {
          result.status = PayloadValidationStatus::MalformedMetadata;
          result.message = "object files cannot import proxy arrays";
          ok = false;
          return;
        }
        if (anari::isObject(arrayData->arrayType())) {
          result.status = PayloadValidationStatus::MalformedMetadata;
          result.message = "object files cannot import arrays of ANARI objects";
          ok = false;
          return;
        }

        auto dim = arrayDim->getValueAs<tsd::math::uint3>();
        const bool is2D = objectType == ANARI_ARRAY2D;
        const bool is3D = objectType == ANARI_ARRAY3D;
        const size_t expectedArraySize =
            size_t(dim[0]) * (is2D || is3D ? size_t(dim[1]) : size_t(1))
            * (is3D ? size_t(dim[2]) : size_t(1));

        anari::DataType arrayElementType = ANARI_UNKNOWN;
        const void *arrayPtr = nullptr;
        size_t arraySize = 0;
        arrayData->getValueAsArray(&arrayElementType, &arrayPtr, &arraySize);
        if (arraySize != expectedArraySize) {
          result.status = PayloadValidationStatus::MalformedMetadata;
          result.message = "arrayData size does not match arrayDim";
          ok = false;
          return;
        }
      }

      const auto key = makeKey(objectType, index);
      if (findEntry(entries, key)) {
        result.status = PayloadValidationStatus::MalformedMetadata;
        result.message = "objectDB contains duplicate object indices";
        ok = false;
        return;
      }

      entries.push_back({key, objectType, &objectNode});
      expectedIndex++;
    });

    if (!ok)
      return false;
  }

  return true;
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
        && !poolAllowedForRoot(rootType, poolNode.name())
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
  if (!findEntry(entries, rootKey)) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = "rootObject entry is missing from objectDB";
    return false;
  }

  size_t rootCount = 0;
  for (const auto &entry : entries) {
    if (!typeAllowedForRoot(rootType, entry.objectType)) {
      result.status = PayloadValidationStatus::IncompatibleSchema;
      result.message = "object payload contains unsupported object type ";
      result.message += anari::toString(entry.objectType);
      return false;
    }

    if (entry.objectType == rootType)
      rootCount++;
  }

  if (rootCount != 1) {
    result.status = PayloadValidationStatus::IncompatibleSchema;
    result.message = "object payload must contain exactly one root object";
    return false;
  }

  std::vector<ObjectKey> reachable;
  reachable.push_back(rootKey);

  for (size_t cursor = 0; cursor < reachable.size(); cursor++) {
    auto *entry = findEntry(entries, reachable[cursor]);
    if (!entry)
      continue;

    bool traversalOK = true;
    entry->node->traverse([&](core::DataNode &node, int) {
      if (!traversalOK)
        return false;

      if (node.holdsArray() && anari::isObject(node.arrayType())) {
        result.status = PayloadValidationStatus::MalformedMetadata;
        result.message =
            "object array values are not supported in object files";
        traversalOK = false;
        return false;
      }

      if (!node.holdsObjectIdx())
        return true;

      anari::DataType refType = ANARI_UNKNOWN;
      size_t refIndex = tsd::core::INVALID_INDEX;
      node.getValueAsObjectIdx(&refType, &refIndex);
      auto refKey = makeKey(refType, refIndex);
      if (!findEntry(entries, refKey)) {
        result.status = PayloadValidationStatus::MalformedMetadata;
        result.message = "object payload references missing object ";
        result.message += anari::toString(refType);
        result.message += " @";
        result.message += std::to_string(refIndex);
        traversalOK = false;
        return false;
      }

      if (std::none_of(reachable.begin(), reachable.end(), [&](auto &key) {
            return sameKey(key, refKey);
          }))
        reachable.push_back(refKey);

      return true;
    });

    if (!traversalOK)
      return false;
  }

  for (const auto &entry : entries) {
    if (std::none_of(reachable.begin(), reachable.end(), [&](auto &key) {
          return sameKey(key, entry.file);
        })) {
      result.status = PayloadValidationStatus::IncompatibleSchema;
      result.message = "object payload contains unreferenced objects";
      return false;
    }
  }

  return true;
}

PayloadValidationResult validateObjectPayloadImpl(core::DataNode &root,
    const std::vector<std::string_view> &acceptedSchemas)
{
  auto result = validateObjectMetadata(root, acceptedSchemas);
  if (!result.accepted())
    return result;

  const auto rootType = rootTypeForSchema(result.schema);
  std::vector<FileObjectEntry> entries;
  validateObjectGraph(root, rootType, entries, result);
  return result;
}

Object *createTargetObject(Scene &scene, core::DataNode &node)
{
  const auto self = node["self"].getValue();
  const auto type = self.type();
  const Token subtype(node["subtype"].getValueAs<std::string>());

  if (anari::isArray(type)) {
    auto &arrayData = node["arrayData"];
    auto dim = node["arrayDim"].getValueAs<tsd::math::uint3>();

    const bool is2D = type == ANARI_ARRAY2D;
    const bool is3D = type == ANARI_ARRAY3D;
    const size_t dim_x = dim[0];
    const size_t dim_y = is2D || is3D ? dim[1] : size_t(0);
    const size_t dim_z = is3D ? dim[2] : size_t(0);

    anari::DataType arrayElementType = ANARI_UNKNOWN;
    const void *arrayPtr = nullptr;
    size_t arraySize = 0;
    arrayData.getValueAsArray(&arrayElementType, &arrayPtr, &arraySize);

    auto array = scene.createArray(arrayElementType, dim_x, dim_y, dim_z);
    if (!array)
      return nullptr;

    const size_t expectedSize = array->size();
    if (arraySize != expectedSize) {
      scene.removeObject(array.data());
      return nullptr;
    }

    if (expectedSize > 0) {
      auto *memOut = array->map();
      std::memcpy(memOut, arrayPtr, array->size() * array->elementSize());
      array->unmap();
    }
    return array.data();
  }

  switch (type) {
  case ANARI_GEOMETRY:
    return scene.createObject<Geometry>(subtype).data();
  case ANARI_MATERIAL:
    return scene.createObject<Material>(subtype).data();
  case ANARI_SAMPLER:
    return scene.createObject<Sampler>(subtype).data();
  case ANARI_SURFACE:
    return scene.createSurface().data();
  case ANARI_SPATIAL_FIELD:
    return scene.createObject<SpatialField>(subtype).data();
  case ANARI_VOLUME:
    return scene.createObject<Volume>(subtype).data();
  default:
    break;
  }

  return nullptr;
}

void clearObjectPayload(Object &object)
{
  object.removeAllParameters();
  while (object.numMetadata() > 0) {
    std::string name = object.getMetadataName(0);
    object.removeMetadata(name);
  }
}

void rollbackCreatedObjects(Scene &scene, const std::vector<Any> &created)
{
  for (auto &ref : created) {
    if (auto *object = scene.getObject(ref))
      clearObjectPayload(*object);
  }

  for (auto it = created.rbegin(); it != created.rend(); ++it)
    scene.removeObject(*it);
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
  targetEntries.reserve(fileEntries.size());
  createdRefs.reserve(fileEntries.size());

  try {
    for (auto &fileEntry : fileEntries) {
      auto *object = createTargetObject(scene, *fileEntry.node);
      if (!object) {
        rollbackCreatedObjects(scene, createdRefs);
        tsd::core::logError("[import_Object] failed to create target object");
        return nullptr;
      }

      clearObjectPayload(*object);
      Any targetRef(object->type(), object->index());
      createdRefs.push_back(targetRef);
      targetEntries.push_back({fileEntry.file, targetRef});
    }

    for (auto &fileEntry : fileEntries) {
      auto *targetEntry = findEntry(targetEntries, fileEntry.file);
      if (!targetEntry) {
        rollbackCreatedObjects(scene, createdRefs);
        tsd::core::logError("[import_Object] missing target object mapping");
        return nullptr;
      }

      core::DataTree rewrittenTree;
      rewrittenTree.root() = *fileEntry.node;
      std::string errorMessage;
      if (!rewriteObjectReferences(
              rewrittenTree.root(), targetEntries, errorMessage)) {
        rollbackCreatedObjects(scene, createdRefs);
        tsd::core::logError(
            "[import_Object] %s", errorMessage.c_str());
        return nullptr;
      }

      auto *targetObject = scene.getObject(targetEntry->target);
      if (!targetObject) {
        rollbackCreatedObjects(scene, createdRefs);
        tsd::core::logError("[import_Object] target object was removed");
        return nullptr;
      }

      nodeToObject(rewrittenTree.root(), *targetObject);
    }
  } catch (const std::exception &e) {
    rollbackCreatedObjects(scene, createdRefs);
    tsd::core::logError("[import_Object] import failed: %s", e.what());
    return nullptr;
  }

  const auto rootKey = makeKey(rootType, 0);
  if (auto *entry = findEntry(targetEntries, rootKey))
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

  std::vector<ClosureEntry> entries;
  std::string errorMessage;
  if (!buildExportClosure(obj, entries, errorMessage)) {
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

  auto &objectDB = root["objectDB"];
  for (auto poolName : OBJECT_POOL_NAMES)
    objectDB[poolName];

  for (auto poolName : OBJECT_POOL_NAMES) {
    anari::DataType poolType = std::string_view(poolName) == "array"
        ? ANARI_ARRAY
        : nonArrayTypeForPoolName(poolName);
    size_t localIndex = 0;
    while (auto *entry = entryForLocalIndex(entries, poolType, localIndex++)) {
      auto &node = objectDB[poolName].append();
      objectToNode(*entry->object, node, false);
      if (!rewriteObjectReferences(node, entries, errorMessage)) {
        tsd::core::logError("[export_Object] %s", errorMessage.c_str());
        return false;
      }
    }
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
