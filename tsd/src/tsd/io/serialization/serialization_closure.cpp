// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/serialization/serialization_closure.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <cstring>

namespace tsd::io::detail {

const std::array<const char *, 8> OBJECT_POOL_NAMES = {
    "array",
    "sampler",
    "material",
    "geometry",
    "surface",
    "spatialfield",
    "volume",
    "light",
};

// Keys / pools ///////////////////////////////////////////////////////////////

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
  if (name == "light")
    return ANARI_LIGHT;
  return ANARI_UNKNOWN;
}

bool isKnownObjectPoolName(std::string_view name)
{
  return std::find(OBJECT_POOL_NAMES.begin(), OBJECT_POOL_NAMES.end(), name)
      != OBJECT_POOL_NAMES.end();
}

// Policy /////////////////////////////////////////////////////////////////////

bool ClosurePolicy::contains(anari::DataType canonicalType) const
{
  for (size_t i = 0; i < numAllowed; i++)
    if (allowed[i] == canonicalType)
      return true;
  return false;
}

static ClosurePolicy makePolicy(
    std::initializer_list<anari::DataType> types, bool singleRoot,
    anari::DataType singleRootType)
{
  ClosurePolicy policy;
  for (auto t : types)
    policy.allowed[policy.numAllowed++] = t;
  policy.singleRoot = singleRoot;
  policy.singleRootType = singleRootType;
  return policy;
}

ClosurePolicy objectFilePolicy(anari::DataType rootType)
{
  if (rootType == ANARI_SURFACE) {
    return makePolicy(
        {ANARI_ARRAY, ANARI_SAMPLER, ANARI_MATERIAL, ANARI_GEOMETRY,
            ANARI_SURFACE},
        true,
        ANARI_SURFACE);
  }
  if (rootType == ANARI_VOLUME) {
    return makePolicy(
        {ANARI_ARRAY, ANARI_SAMPLER, ANARI_SPATIAL_FIELD, ANARI_VOLUME},
        true,
        ANARI_VOLUME);
  }
  return {};
}

ClosurePolicy layerSubtreePolicy()
{
  return makePolicy({ANARI_ARRAY,
                        ANARI_SAMPLER,
                        ANARI_MATERIAL,
                        ANARI_GEOMETRY,
                        ANARI_SURFACE,
                        ANARI_SPATIAL_FIELD,
                        ANARI_VOLUME,
                        ANARI_LIGHT},
      false,
      ANARI_UNKNOWN);
}

ClosurePolicy lightRigPolicy()
{
  return makePolicy({ANARI_ARRAY, ANARI_LIGHT}, false, ANARI_UNKNOWN);
}

bool typeAllowed(const ClosurePolicy &policy, anari::DataType type)
{
  return policy.contains(canonicalObjectType(type));
}

bool poolAllowed(const ClosurePolicy &policy, std::string_view poolName)
{
  if (poolName == "array")
    return policy.contains(ANARI_ARRAY);
  return policy.contains(nonArrayTypeForPoolName(poolName));
}

static size_t *counterForType(PoolCounters &counters, anari::DataType type)
{
  switch (canonicalObjectType(type)) {
  case ANARI_ARRAY:
    return &counters.arrays;
  case ANARI_SAMPLER:
    return &counters.samplers;
  case ANARI_MATERIAL:
    return &counters.materials;
  case ANARI_GEOMETRY:
    return &counters.geometries;
  case ANARI_SURFACE:
    return &counters.surfaces;
  case ANARI_SPATIAL_FIELD:
    return &counters.fields;
  case ANARI_VOLUME:
    return &counters.volumes;
  case ANARI_LIGHT:
    return &counters.lights;
  default:
    return nullptr;
  }
}

// Lookups ////////////////////////////////////////////////////////////////////

ClosureEntry *findClosureEntry(
    std::vector<ClosureEntry> &entries, const ObjectKey &key)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return sameKey(entry.source, key);
  });
  return it == entries.end() ? nullptr : &*it;
}

const ClosureEntry *entryForLocalIndex(
    const std::vector<ClosureEntry> &entries,
    anari::DataType type,
    size_t localIndex)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return canonicalObjectType(entry.objectType) == canonicalObjectType(type)
        && entry.localIndex == localIndex;
  });
  return it == entries.end() ? nullptr : &*it;
}

FileObjectEntry *findFileEntry(
    std::vector<FileObjectEntry> &entries, const ObjectKey &key)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
    return sameKey(entry.file, key);
  });
  return it == entries.end() ? nullptr : &*it;
}

const TargetObjectEntry *findTargetEntry(
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
      if (message)
        *message = "object array values are not supported in object files";
      found = true;
      return false;
    }
    return true;
  });
  return found;
}

// Export-side closure construction ///////////////////////////////////////////

bool admitObject(const Scene &scene,
    const ClosurePolicy &policy,
    const ObjectKey &rootKey,
    Object *object,
    std::vector<ClosureEntry> &entries,
    PoolCounters &counters,
    std::string &errorMessage)
{
  if (!object)
    return true;

  const auto key = makeKey(object->type(), object->index());
  if (findClosureEntry(entries, key))
    return true;

  const auto objectType = object->type();
  if (!typeAllowed(policy, objectType)) {
    errorMessage = "object closure reached unsupported type ";
    errorMessage += anari::toString(objectType);
    return false;
  }

  if (policy.singleRoot
      && (objectType == ANARI_SURFACE || objectType == ANARI_VOLUME)
      && !sameKey(key, rootKey)) {
    errorMessage = "object files support only one root ";
    errorMessage += anari::toString(policy.singleRootType);
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

  auto *counter = counterForType(counters, objectType);
  if (!counter) {
    errorMessage = "object closure reached unsupported type ";
    errorMessage += anari::toString(objectType);
    return false;
  }

  entries.push_back({key, objectType, (*counter)++, object});
  return true;
}

static bool admitReference(const Scene &scene,
    const ClosurePolicy &policy,
    const ObjectKey &rootKey,
    const Any &value,
    std::vector<ClosureEntry> &entries,
    PoolCounters &counters,
    std::string &errorMessage)
{
  if (!value.holdsObject())
    return true;

  const auto key = makeKey(value);
  if (findClosureEntry(entries, key))
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

  return admitObject(
      scene, policy, rootKey, object, entries, counters, errorMessage);
}

bool buildClosure(const Scene &scene,
    const std::vector<Object *> &seeds,
    const ClosurePolicy &policy,
    const ObjectKey &rootKey,
    std::vector<ClosureEntry> &entries,
    std::string &errorMessage)
{
  PoolCounters counters;

  for (auto *seed : seeds) {
    if (!admitObject(
            scene, policy, rootKey, seed, entries, counters, errorMessage))
      return false;
  }

  for (size_t i = 0; i < entries.size(); i++) {
    auto *object = entries[i].object;

    core::DataTree scratchTree;
    objectToNode(*object, scratchTree.root(), false);
    if (hasObjectArrayNode(scratchTree.root(), &errorMessage))
      return false;

    for (size_t p = 0; p < object->numParameters(); p++) {
      const auto &param = object->parameterAt(p);
      if (!admitReference(scene,
              policy,
              rootKey,
              param.value(),
              entries,
              counters,
              errorMessage))
        return false;
      if (param.hasMin()
          && !admitReference(scene,
              policy,
              rootKey,
              param.min(),
              entries,
              counters,
              errorMessage))
        return false;
      if (param.hasMax()
          && !admitReference(scene,
              policy,
              rootKey,
              param.max(),
              entries,
              counters,
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

      if (!admitReference(scene,
              policy,
              rootKey,
              object->getMetadataValue(name),
              entries,
              counters,
              errorMessage))
        return false;
    }
  }

  return true;
}

bool rewriteRefsToLocal(core::DataNode &root,
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

    auto it = std::find_if(entries.begin(), entries.end(), [&](auto &entry) {
      return sameKey(entry.source, makeKey(type, index));
    });
    if (it == entries.end()) {
      errorMessage = "serialized object reference has no closure mapping";
      ok = false;
      return false;
    }

    node.setValue(Any(type, it->localIndex));
    return true;
  });
  return ok;
}

bool writeObjectDB(core::DataNode &objectDB,
    const std::vector<ClosureEntry> &entries,
    std::string &errorMessage)
{
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
      if (!rewriteRefsToLocal(node, entries, errorMessage))
        return false;
    }
  }

  return true;
}

// Import-side validation /////////////////////////////////////////////////////

static PayloadValidationResult makeMetadataFailure(
    PayloadValidationStatus status, std::string message)
{
  PayloadValidationResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

PayloadValidationResult validateEnvelope(core::DataNode &root,
    std::string_view expectedFileType,
    const std::vector<std::string_view> &acceptedSchemas,
    const std::vector<std::string_view> &knownSchemas)
{
  auto metadataResult = core::readDataTreeMetadata(root);
  if (metadataResult.malformed()) {
    return makeMetadataFailure(
        PayloadValidationStatus::MalformedMetadata, metadataResult.message);
  }

  if (!metadataResult.found()) {
    return makeMetadataFailure(PayloadValidationStatus::MissingRequiredNode,
        std::string(expectedFileType) + " payload requires __tsd_metadata");
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

  if (metadata.fileType != expectedFileType) {
    result.status = PayloadValidationStatus::IncompatibleSchema;
    result.message = "fileType '" + metadata.fileType
        + "' is not accepted by this loader";
    return result;
  }

  const auto schemaMatches = [&](std::string_view schema) {
    return metadata.schema == schema;
  };

  if (std::none_of(
          acceptedSchemas.begin(), acceptedSchemas.end(), schemaMatches)) {
    result.status =
        std::any_of(knownSchemas.begin(), knownSchemas.end(), schemaMatches)
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
      if (findFileEntry(entries, key)) {
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

bool checkGraphConsistency(std::vector<FileObjectEntry> &entries,
    const std::vector<ObjectKey> &seedKeys,
    const ClosurePolicy &policy,
    bool requireAllReachable,
    PayloadValidationResult &result)
{
  size_t rootCount = 0;
  for (const auto &entry : entries) {
    if (!typeAllowed(policy, entry.objectType)) {
      result.status = PayloadValidationStatus::IncompatibleSchema;
      result.message = "object payload contains unsupported object type ";
      result.message += anari::toString(entry.objectType);
      return false;
    }

    if (policy.singleRoot && entry.objectType == policy.singleRootType)
      rootCount++;
  }

  if (policy.singleRoot && rootCount != 1) {
    result.status = PayloadValidationStatus::IncompatibleSchema;
    result.message = "object payload must contain exactly one root object";
    return false;
  }

  std::vector<ObjectKey> reachable = seedKeys;

  for (size_t cursor = 0; cursor < reachable.size(); cursor++) {
    auto *entry = findFileEntry(entries, reachable[cursor]);
    if (!entry) {
      result.status = PayloadValidationStatus::MissingRequiredNode;
      result.message = "object payload references missing object ";
      result.message += anari::toString(reachable[cursor].type);
      result.message += " @";
      result.message += std::to_string(reachable[cursor].index);
      return false;
    }

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
      if (!findFileEntry(entries, refKey)) {
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

  if (requireAllReachable) {
    for (const auto &entry : entries) {
      if (std::none_of(reachable.begin(), reachable.end(), [&](auto &key) {
            return sameKey(key, entry.file);
          })) {
        result.status = PayloadValidationStatus::IncompatibleSchema;
        result.message = "object payload contains unreferenced objects";
        return false;
      }
    }
  }

  return true;
}

// Import-side object instantiation ///////////////////////////////////////////

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
  case ANARI_LIGHT:
    return scene.createObject<Light>(subtype).data();
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

bool rewriteRefsToTarget(core::DataNode &root,
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

    auto *entry = findTargetEntry(entries, makeKey(type, index));
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

bool instantiateObjectDB(Scene &scene,
    std::vector<FileObjectEntry> &fileEntries,
    std::vector<TargetObjectEntry> &targetEntries,
    std::vector<Any> &createdRefs,
    std::string &errorMessage)
{
  targetEntries.reserve(fileEntries.size());
  createdRefs.reserve(fileEntries.size());

  try {
    for (auto &fileEntry : fileEntries) {
      auto *object = createTargetObject(scene, *fileEntry.node);
      if (!object) {
        rollbackCreatedObjects(scene, createdRefs);
        errorMessage = "failed to create target object";
        return false;
      }

      clearObjectPayload(*object);
      Any targetRef(object->type(), object->index());
      createdRefs.push_back(targetRef);
      targetEntries.push_back({fileEntry.file, targetRef});
    }

    for (auto &fileEntry : fileEntries) {
      auto *targetEntry = findTargetEntry(targetEntries, fileEntry.file);
      if (!targetEntry) {
        rollbackCreatedObjects(scene, createdRefs);
        errorMessage = "missing target object mapping";
        return false;
      }

      core::DataTree rewrittenTree;
      rewrittenTree.root() = *fileEntry.node;
      if (!rewriteRefsToTarget(
              rewrittenTree.root(), targetEntries, errorMessage)) {
        rollbackCreatedObjects(scene, createdRefs);
        return false;
      }

      auto *targetObject = scene.getObject(targetEntry->target);
      if (!targetObject) {
        rollbackCreatedObjects(scene, createdRefs);
        errorMessage = "target object was removed";
        return false;
      }

      nodeToObject(rewrittenTree.root(), *targetObject);
    }
  } catch (const std::exception &e) {
    rollbackCreatedObjects(scene, createdRefs);
    errorMessage = std::string("import failed: ") + e.what();
    return false;
  }

  return true;
}

} // namespace tsd::io::detail
