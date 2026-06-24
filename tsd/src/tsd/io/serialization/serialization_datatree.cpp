// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_CUDA
#define TSD_USE_CUDA 1
#endif

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/animation/EnSightFileBinding.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/io/serialization/serialization_closure.hpp"
// std
#include <algorithm>
#include <stack>
#include <stdexcept>
#include <type_traits>
#include <vector>
#if TSD_USE_CUDA
// cuda
#include <cuda_runtime.h>
#endif

namespace tsd::io {

static core::DataNode &resolveScenePayloadRoot(core::DataNode &root)
{
  if (auto *context = root.child("context"))
    return *context;
  return root;
}

static std::string validationStatusToString(PayloadValidationStatus status)
{
  switch (status) {
  case PayloadValidationStatus::Valid:
    return "valid";
  case PayloadValidationStatus::MissingMetadataAccepted:
    return "missing metadata accepted";
  case PayloadValidationStatus::UnknownSchema:
    return "unknown schema";
  case PayloadValidationStatus::IncompatibleSchema:
    return "incompatible schema";
  case PayloadValidationStatus::UnsupportedEnvelopeVersion:
    return "unsupported envelope version";
  case PayloadValidationStatus::UnsupportedSchemaVersion:
    return "unsupported schema version";
  case PayloadValidationStatus::MalformedMetadata:
    return "malformed metadata";
  case PayloadValidationStatus::MissingRequiredNode:
    return "missing required node";
  }

  return "unknown validation status";
}

static void logValidationFailure(
    const char *prefix, const PayloadValidationResult &result)
{
  logError("[%s] payload validation failed: %s%s%s",
      prefix,
      validationStatusToString(result.status).c_str(),
      result.message.empty() ? "" : ": ",
      result.message.c_str());
}

static PayloadValidationResult validateScenePayloadImpl(core::DataNode &root,
    const std::vector<std::string_view> &acceptedSchemas,
    const std::vector<std::string_view> &knownSchemas)
{
  auto &payloadRoot = resolveScenePayloadRoot(root);
  auto metadataResult = core::readDataTreeMetadata(payloadRoot);

  PayloadValidationResult result;
  if (metadataResult.malformed()) {
    result.status = PayloadValidationStatus::MalformedMetadata;
    result.message = metadataResult.message;
    return result;
  }

  if (metadataResult.found()) {
    const auto &metadata = *metadataResult.metadata;
    result.fileType = metadata.fileType;
    result.schema = metadata.schema;
    result.envelopeVersion = metadata.envelopeVersion;
    result.schemaVersion = metadata.schemaVersion;

    if (metadata.envelopeVersion
        != core::DATA_TREE_METADATA_ENVELOPE_VERSION) {
      result.status = PayloadValidationStatus::UnsupportedEnvelopeVersion;
      result.message = "expected envelopeVersion 1, got "
          + std::to_string(metadata.envelopeVersion);
      return result;
    }

    const auto schemaMatches = [&](std::string_view schema) {
      return metadata.schema == schema;
    };

    if (std::none_of(
            acceptedSchemas.begin(), acceptedSchemas.end(), schemaMatches)) {
      result.status = std::any_of(
                          knownSchemas.begin(), knownSchemas.end(), schemaMatches)
          ? PayloadValidationStatus::IncompatibleSchema
          : PayloadValidationStatus::UnknownSchema;
      result.message = "schema '" + metadata.schema
          + "' is not accepted by this loader";
      return result;
    }

    if (metadata.schemaVersion != 1) {
      result.status = PayloadValidationStatus::UnsupportedSchemaVersion;
      result.message = "schema '" + metadata.schema
          + "' supports version 1..1, got "
          + std::to_string(metadata.schemaVersion);
      return result;
    }
  } else {
    result.status = PayloadValidationStatus::MissingMetadataAccepted;
    result.message = "payload has no __tsd_metadata node; treating as legacy";
  }

  if (!payloadRoot.child("objectDB")) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = "payload requires root/objectDB";
  }

  return result;
}

static bool isExcludedKey(
    const std::vector<detail::ObjectKey> *excluded, const Object &obj)
{
  if (!excluded)
    return false;
  const auto key = detail::makeKey(obj.type(), obj.index());
  return std::any_of(excluded->begin(), excluded->end(), [&](const auto &k) {
    return detail::sameKey(k, key);
  });
}

template <typename OBJECT_POOL_T>
static void objectPoolToNode(core::DataNode &objPoolRoot,
    const OBJECT_POOL_T &objPool,
    const char *poolName,
    bool forceProxyArrays,
    const std::vector<detail::ObjectKey> *excluded = nullptr)
{
  if (objPool.empty())
    return;

  tsd::core::logStatus(
      "    ...serializing %zu %s objects", size_t(objPool.size()), poolName);

  // Create the pool node lazily so a pool whose objects are all excluded leaves
  // no empty node behind.
  core::DataNode *childNode = nullptr;
  foreach_item_const(objPool, [&](const auto *obj) {
    if (!obj || isExcludedKey(excluded, *obj))
      return;
    if (!childNode)
      childNode = &objPoolRoot[poolName];
    objectToNode(*obj, childNode->append(), forceProxyArrays);
  });
}

// Parameters /////////////////////////////////////////////////////////////////

void parameterToNode(const Parameter &p, core::DataNode &node)
{
  node["value"] = p.value();
  node["enabled"] = p.isEnabled();
  if (!p.description().empty())
    node["description"] = p.description();
  if (p.usage() != ParameterUsageHint::NONE)
    node["usage"] = static_cast<int>(p.usage());
  if (p.hasMin())
    node["min"] = p.min();
  if (p.hasMax())
    node["max"] = p.max();

  if (!p.stringValues().empty()) {
    auto &stringValues = node["stringValues"];
    for (const auto &sv : p.stringValues())
      stringValues.append() = sv;
    node["stringSelection"] = p.stringSelection();
  }
}

void nodeToParameter(core::DataNode &node, Parameter &p)
{
  if (auto *c = node.child("description"); c != nullptr)
    p.setDescription(c->getValueAs<std::string>().c_str());

  if (auto *c = node.child("usage"); c != nullptr)
    p.setUsage(static_cast<ParameterUsageHint>(c->getValueAs<int>()));

  if (auto *c = node.child("min"); c != nullptr)
    p.setMin(c->getValue());

  if (auto *c = node.child("max"); c != nullptr)
    p.setMax(c->getValue());

  if (auto *c = node.child("stringValues"); c != nullptr) {
    std::vector<std::string> stringValues;
    c->foreach_child([&](core::DataNode &child) {
      stringValues.push_back(child.getValueAs<std::string>());
    });
    p.setStringValues(stringValues);
    p.setStringSelection(node["stringSelection"].getValueAs<int>());
  }

  if (auto *c = node.child("enabled"); c != nullptr)
    p.setEnabled(c->getValueAs<bool>());

  p.setValue(node["value"].getValue());
}

void nodeToObjectParameters(core::DataNode &node, Object &obj)
{
  node.foreach_child([&](core::DataNode &parameterNode) {
    const Token parameterName(parameterNode.name().c_str());
    auto &p = obj.addParameter(parameterName);
    nodeToParameter(parameterNode, p);
  });
}

// Objects ////////////////////////////////////////////////////////////////////

// Helper function for arrays
static void arrayToNode(
    const Array &arr, core::DataNode &node, bool forceArraysAsProxies)
{
  node["arrayDim"] = tsd::math::uint3{
      uint32_t(arr.dim(0)), uint32_t(arr.dim(1)), uint32_t(arr.dim(2))};

  auto &arrayData = node.append("arrayData");

  bool isProxy =
      forceArraysAsProxies ? true : (arr.kind() == Array::MemoryKind::PROXY);
  if (isProxy) {
    arrayData = static_cast<int>(arr.elementType());
    return;
  }

  const void *mem = arr.data();
#if TSD_USE_CUDA
  if (arr.kind() == Array::MemoryKind::CUDA) {
    const size_t numBytes = arr.size() * arr.elementSize();
    std::vector<uint8_t> hostBuf(numBytes);
    cudaMemcpy(hostBuf.data(), mem, numBytes, cudaMemcpyDeviceToHost);
    arrayData.setValueAsArray(arr.elementType(), hostBuf.data(), arr.size());
  } else
#endif
    arrayData.setValueAsExternalArray(arr.elementType(), mem, arr.size());
}

void objectToNode(
    const Object &obj, core::DataNode &node, bool forceProxyArrays)
{
  node["name"] = obj.name();
  node["self"] = Any(obj.type(), obj.index());
  node["subtype"] = obj.subtype().c_str();

  if (obj.type() == ANARI_RENDERER && obj.rendererDeviceName())
    node["rendererDeviceName"] = obj.rendererDeviceName().c_str();

  if (obj.numParameters() > 0) {
    auto &params = node["parameters"];
    for (size_t i = 0; i < obj.numParameters(); i++) {
      const auto &p = obj.parameterAt(i);
      parameterToNode(p, params.append(p.name().c_str()));
    }
  }

  if (obj.numMetadata() > 0) {
    auto &metadata = node["metadata"];
    for (size_t i = 0; i < obj.numMetadata(); i++) {
      std::string n = obj.getMetadataName(i);
      anari::DataType type = ANARI_UNKNOWN;
      const void *ptr = nullptr;
      size_t size = 0;
      obj.getMetadataArray(n, &type, &ptr, &size);
      if (type != ANARI_UNKNOWN)
        metadata[n].setValueAsExternalArray(type, ptr, size);
      else if (auto v = obj.getMetadataValue(n); v.valid())
        metadata[n] = v;
    }
  }

  if (anari::isArray(obj.type())) {
    const Array &arr = static_cast<const Array &>(obj);
    arrayToNode(arr, node, forceProxyArrays);
  }
}

void nodeToObject(core::DataNode &node, Object &obj)
{
  if (auto *c = node.child("name"); c != nullptr)
    obj.setName(c->getValueAs<std::string>().c_str());

  if (auto *c = node.child("parameters"); c != nullptr)
    nodeToObjectParameters(*c, obj);

  if (auto *c = node.child("metadata"); c != nullptr)
    nodeToObjectMetadata(*c, obj);
}

void nodeToObjectMetadata(core::DataNode &node, Object &obj)
{
  node.foreach_child([&](core::DataNode &n) {
    if (n.holdsArray()) {
      anari::DataType type = ANARI_UNKNOWN;
      const void *ptr = nullptr;
      size_t size = 0;
      n.getValueAsArray(&type, &ptr, &size);
      obj.setMetadataArray(n.name(), type, ptr, size);
    } else {
      obj.setMetadataValue(n.name(), n.getValue());
    }
  });
}

void nodeToNewObject(Scene &scene, core::DataNode &node)
{
  const Any self = node["self"].getValue();
  const auto type = self.type();
  const size_t index = self.getAsObjectIndex();
  const Token subtype(node["subtype"].getValueAs<std::string>());

  if (!anari::isObject(type)) {
    logError("[nodeToObject] parsed invalid object type '%s'",
        anari::toString(type));
    return;
  }

  Object *obj = nullptr;
  switch (type) {
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D: {
    auto &arrayData = node["arrayData"];
    auto &arrayDim = node["arrayDim"];
    auto isProxy = !arrayData.holdsArray();

    auto dim = arrayDim.getValueAs<tsd::math::uint3>();

    const bool is2D = type == ANARI_ARRAY2D;
    const bool is3D = type == ANARI_ARRAY3D;
    const size_t dim_x = dim[0];
    const size_t dim_y = is2D || is3D ? dim[1] : size_t(0);
    const size_t dim_z = is3D ? dim[2] : size_t(0);

    anari::DataType arrayElementType = ANARI_UNKNOWN;
    const void *arrayPtr = nullptr;
    size_t arraySize = 0;

    if (isProxy) {
      arrayElementType =
          static_cast<anari::DataType>(arrayData.getValueAs<int>());
    } else {
      arrayData.getValueAsArray(&arrayElementType, &arrayPtr, &arraySize);
    }

    auto arr = isProxy
        ? scene.createArrayProxy(arrayElementType, dim_x, dim_y, dim_z)
        : scene.createArray(arrayElementType, dim_x, dim_y, dim_z);

    if (arr) {
      obj = arr.data();
      if (!isProxy) {
        auto *memOut = arr->map();
        std::memcpy(memOut, arrayPtr, arr->size() * arr->elementSize());
        arr->unmap();
      }
    }
  } break;
  case ANARI_GEOMETRY:
    obj = scene.createObject<Geometry>(subtype).data();
    break;
  case ANARI_MATERIAL:
    obj = scene.createObject<Material>(subtype).data();
    break;
  case ANARI_SAMPLER:
    obj = scene.createObject<Sampler>(subtype).data();
    break;
  case ANARI_SURFACE:
    obj = scene.createSurface().data();
    break;
  case ANARI_SPATIAL_FIELD:
    obj = scene.createObject<SpatialField>(subtype).data();
    break;
  case ANARI_VOLUME:
    obj = scene.createObject<Volume>(subtype).data();
    break;
  case ANARI_LIGHT:
    obj = scene.createObject<Light>(subtype).data();
    break;
  case ANARI_CAMERA:
    obj = scene.createObject<Camera>(subtype).data();
    break;
  case ANARI_RENDERER: {
    std::string rendererDeviceName;
    if (auto *c = node.child("rendererDeviceName"); c != nullptr)
      rendererDeviceName = c->getValueAs<std::string>();
    if (!rendererDeviceName.empty())
      obj = scene.createRenderer(rendererDeviceName, subtype).get();
  } break;
  default:
    break;
  }

  if (!obj) {
    logError("[nodeToObject] unable to create object from DataNode");
    return;
  }

  if (obj->index() != index) {
    logError("[nodeToObject] object (%s) index mismatch on import: %zu | %zu",
        anari::toString(type),
        obj->index(),
        index);
  }

  obj->removeAllParameters(); // clear default parameters
  nodeToObject(node, *obj);
}

// Camera poses ///////////////////////////////////////////////////////////////

void cameraPoseToNode(const rendering::CameraPose &p, core::DataNode &node)
{
  node["name"] = p.name;
  node["lookat"] = p.lookat;
  node["azeldist"] = p.azeldist;
  node["fixedDist"] = p.fixedDist;
  node["upAxis"] = p.upAxis;
  node["mode"] = p.mode;
}

void nodeToCameraPose(core::DataNode &node, rendering::CameraPose &pose)
{
  node["name"].getValue(ANARI_STRING, &pose.name);
  node["lookat"].getValue(ANARI_FLOAT32_VEC3, &pose.lookat);
  node["azeldist"].getValue(ANARI_FLOAT32_VEC3, &pose.azeldist);
  node["fixedDist"].getValue(ANARI_FLOAT32, &pose.fixedDist);
  node["upAxis"].getValue(ANARI_INT32, &pose.upAxis);
  node["mode"].getValue(ANARI_INT32, &pose.mode);
}

// Layers /////////////////////////////////////////////////////////////////////

void layerNodeInstanceParametersToNode(
    const LayerNodeData &data, core::DataNode &node)
{
  const auto &instanceParameters = data.getInstanceParameters();
  if (instanceParameters.empty())
    return;

  auto &ipNode = node.append("instanceParameters");
  for (const auto &p : instanceParameters)
    ipNode.append(p.first) = p.second;
}

void nodeToLayerNodeInstanceParameters(
    core::DataNode &node, LayerNodeData &data)
{
  if (auto *ipNode = node.child("instanceParameters"); ipNode != nullptr) {
    ipNode->foreach_child([&](core::DataNode &p) {
      data.setInstanceParameter(p.name(), p.getValue());
    });
  }
}

void layerToNode(const Layer &layer, core::DataNode &node)
{
  layerSubtreeToNode(layer, layer.root(), node);
}

// Serialize a layer subtree, optionally pruning a set of excluded subtree roots
// (and everything beneath them). Excluded nodes are never emitted, so the
// emitted hierarchy stays self-consistent for the level-tracking bookkeeping.
static void layerSubtreeToNodeImpl(const Layer &layer,
    LayerNodeRef start,
    core::DataNode &node,
    const std::vector<LayerNodeRef> *excluded)
{
  auto isExcludedNode = [&](const LayerNode &tsdNode) {
    if (!excluded)
      return false;
    return std::any_of(excluded->begin(), excluded->end(), [&](LayerNodeRef r) {
      return r && &(*r) == &tsdNode;
    });
  };

  std::stack<core::DataNode *> nodes;
  core::DataNode *currentParentNode = nullptr;
  core::DataNode *currentNode = &node;
  int currentLevel = -1;
  layer.traverse_const(start, [&](const LayerNode &tsdNode, int level) {
    if (isExcludedNode(tsdNode))
      return false; // prune this subtree without emitting it

    if (currentLevel < level) {
      nodes.push(currentNode);
      currentParentNode = currentNode;
    } else if (currentLevel > level) {
      for (int i = 0; i < currentLevel - level; i++)
        nodes.pop();
      currentParentNode = nodes.top();
    }

    currentLevel = level;

    if (level == 0)
      currentNode = &node;
    else
      currentNode = &currentParentNode->child("children")->append();

    currentNode->append("name") = tsdNode->name();
    currentNode->append("value") = tsdNode->getValueRaw();
    if (tsdNode->isTransform())
      currentNode->append("transformSRT") = tsdNode->getTransformSRT();
    currentNode->append("enabled") = tsdNode->isEnabled();
    layerNodeInstanceParametersToNode(*tsdNode, *currentNode);
    currentNode->append("children");

    return true;
  });
}

void layerSubtreeToNode(
    const Layer &layer, LayerNodeRef start, core::DataNode &node)
{
  layerSubtreeToNodeImpl(layer, start, node, nullptr);
}

void nodeToLayer(core::DataNode &rootNode, Layer &layer, Scene &scene)
{
  layer.clear();

  std::stack<LayerNodeRef> tsdNodes;
  LayerNodeRef currentParentNode;
  LayerNodeRef currentNode = layer.root();
  int currentLevel = -1;
  rootNode.traverse([&](core::DataNode &node, int level) {
    if (level & 0x1 || !node.child("children"))
      return true;

    level /= 2;
    if (currentLevel < level) {
      tsdNodes.push(currentNode);
      currentParentNode = currentNode;
    } else if (currentLevel > level) {
      for (int i = 0; i < currentLevel - level; i++)
        tsdNodes.pop();
      currentParentNode = tsdNodes.top();
    }

    currentLevel = level;

    if (level == 0)
      currentNode = layer.root();
    else {
      currentNode = currentParentNode->insert_last_child({&layer});
      if (auto *c = node.child("transformSRT"); c != nullptr)
        (*currentNode)->setAsTransform(c->getValueAs<math::mat3>());
      else
        (*currentNode)->setValueRaw(node["value"].getValue());
      (*currentNode)->setEnabled(node["enabled"].getValueOr(true));
      (*currentNode)->name() = node["name"].getValueAs<std::string>();
      nodeToLayerNodeInstanceParameters(node, (*currentNode).value());
    }

    return true;
  });
}

// Animations /////////////////////////////////////////////////////////////////

void animationToNode(const animation::Animation &anim, core::DataNode &node)
{
  node["name"] = anim.name();

  auto &bindingsNode = node["objectBindings"];
  for (const auto &b : anim.objectParameterBindings())
    b.toDataNode(bindingsNode.append());

  auto &transformsNode = node["transformBindings"];
  for (const auto &tb : anim.transformBindings())
    tb.toDataNode(transformsNode.append());

  auto &fileBindingsNode = node["fileBindings"];
  for (const auto &fb : anim.fileBindings()) {
    auto &fbNode = fileBindingsNode.append();
    fbNode["kind"] = fb->kind();
    fb->toDataNode(fbNode);
  }
}

void nodeToAnimation(
    core::DataNode &node, animation::Animation &anim, Scene &scene)
{
  anim.editableName() = node["name"].getValueAs<std::string>();

  if (auto *bindingsNode = node.child("objectBindings")) {
    bindingsNode->foreach_child([&](core::DataNode &bn) {
      auto &b = anim.addEmptyObjectParameterBinding();
      b.fromDataNode(bn);
    });
  }

  if (auto *transformsNode = node.child("transformBindings")) {
    transformsNode->foreach_child([&](core::DataNode &tn) {
      auto &b = anim.addEmptyTransformBinding();
      b.fromDataNode(tn);
    });
  }

  if (auto *fileBindingsNode = node.child("fileBindings")) {
    fileBindingsNode->foreach_child([&](core::DataNode &fbNode) {
      auto kind = fbNode["kind"].getValueAs<std::string>();
      if (kind == "spatialField") {
        auto targetIndex = fbNode["targetIndex"].getValueAs<size_t>();
        auto *vol = static_cast<scene::Volume *>(
            scene.getObject(ANARI_VOLUME, targetIndex));
        if (!vol) {
          logWarning(
              "[nodeToAnimation] spatialField binding: volume index %zu not"
              " found; skipping",
              targetIndex);
          return;
        }

        std::vector<std::string> files;
        if (auto *filesNode = fbNode.child("files")) {
          filesNode->foreach_child([&](core::DataNode &fn) {
            files.push_back(fn.getValueAs<std::string>());
          });
        }

        // The volume's current "value" param is the initial field (frame 0)
        scene::SpatialFieldRef initialField;
        if (auto *sf =
                vol->parameterValueAsObject<scene::SpatialField>("value")) {
          initialField = sf->self();
        }

        anim.emplaceFileBinding<SpatialFieldFileBinding>(
            &scene, vol, initialField, std::move(files));
      } else if (kind == "ensight") {
        auto data = EnSightFileBinding::fromDataNode(scene, fbNode);
        if (!data)
          return;

        anim.emplaceFileBinding<EnSightFileBinding>(&scene,
            std::move(data->parts),
            std::move(data->geoFiles),
            std::move(data->fieldMappings));
      } else {
        logWarning(
            "[nodeToAnimation] unknown file binding kind '%s'; skipping",
            kind.c_str());
      }
    });
  }
}

void animationManagerToNode(
    const animation::AnimationManager &mgr, core::DataNode &node)
{
  node["time"] = mgr.getAnimationTime();
  node["increment"] = mgr.getAnimationIncrement();
  node["totalFrames"] = mgr.getAnimationTotalFrames();
  node["fps"] = mgr.getAnimationFPS();

  auto &animationsNode = node["objects"];
  for (const auto &anim : mgr.animations()) {
    animationToNode(anim, animationsNode.append());
  }
}

void nodeToAnimationManager(
    core::DataNode &node, animation::AnimationManager &mgr, Scene &scene)
{
  float time = mgr.getAnimationTime();
  float increment = mgr.getAnimationIncrement();
  float fps = mgr.getAnimationFPS();
  int totalFrames = mgr.getAnimationTotalFrames();

  time = node["time"].getValueOr<float>(time);
  increment = node["increment"].getValueOr<float>(increment);
  fps = node["fps"].getValueOr<float>(fps);
  totalFrames = node["totalFrames"].getValueOr<int>(totalFrames);

  mgr.setAnimationIncrement(increment);
  mgr.setAnimationFPS(fps);
  mgr.setAnimationTotalFrames(totalFrames);

  node["objects"].foreach_child([&](core::DataNode &animNode) {
    auto &anim = mgr.addAnimation();
    nodeToAnimation(animNode, anim, scene);
  });

  mgr.setAnimationTime(time);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Scene-save exclusion helpers ///////////////////////////////////////////////

// Visit every Object in the Scene's objectDB, pool by pool.
template <typename FCN>
static void forEachSceneObject(Scene &scene, FCN &&fn)
{
  auto &db = scene.objectDB();
  auto pool = [&](auto &p) {
    foreach_item_const(p, [&](const auto *o) {
      if (o)
        fn(*static_cast<const Object *>(o));
    });
  };
  pool(db.geometry);
  pool(db.sampler);
  pool(db.material);
  pool(db.surface);
  pool(db.field);
  pool(db.volume);
  pool(db.light);
  pool(db.camera);
  pool(db.renderer);
  pool(db.array);
}

// Visit every object reference held by an object's parameters and metadata.
template <typename FCN>
static void forEachObjectRef(const Object &obj, FCN &&fn)
{
  for (size_t p = 0; p < obj.numParameters(); p++) {
    const auto &param = obj.parameterAt(p);
    if (param.value().holdsObject())
      fn(param.value());
    if (param.hasMin() && param.min().holdsObject())
      fn(param.min());
    if (param.hasMax() && param.max().holdsObject())
      fn(param.max());
  }
  for (size_t m = 0; m < obj.numMetadata(); m++) {
    const auto *name = obj.getMetadataName(m);
    anari::DataType arrayType = ANARI_UNKNOWN;
    const void *arrayPtr = nullptr;
    size_t arraySize = 0;
    obj.getMetadataArray(name, &arrayType, &arrayPtr, &arraySize);
    if (arrayType != ANARI_UNKNOWN)
      continue;
    if (auto v = obj.getMetadataValue(name); v.holdsObject())
      fn(v);
  }
}

static bool keySetContains(
    const std::vector<detail::ObjectKey> &keys, const detail::ObjectKey &key)
{
  return std::any_of(keys.begin(), keys.end(), [&](const auto &k) {
    return detail::sameKey(k, key);
  });
}

// Gather the light objects (and instance-parameter objects) seeded by the
// excluded subtrees, then close over the arrays they reference (lightRigPolicy).
// Arrays still referenced by *retained* objects are rescued so the saved scene
// keeps a valid reference graph. Sets *ok=false if the closure could not be
// built (the caller then excludes nothing, keeping the payload self-consistent).
static std::vector<detail::ObjectKey> computeExcludedObjects(
    Scene &scene, const std::vector<LayerNodeRef> &excludedSubtrees, bool &ok)
{
  ok = true;
  std::vector<detail::ObjectKey> excluded;

  std::vector<Object *> seeds;
  for (auto root : excludedSubtrees) {
    if (!root)
      continue;
    auto *layer = (*root).value().layer();
    if (!layer)
      continue;
    layer->traverse_const(root, [&](const LayerNode &tsdNode, int) {
      if (tsdNode->isObject()) {
        if (auto *o = tsdNode->getObject())
          seeds.push_back(o);
      }
      for (const auto &p : tsdNode->getInstanceParameters()) {
        if (p.second.holdsObject()) {
          if (auto *o = scene.getObject(p.second))
            seeds.push_back(o);
        }
      }
      return true;
    });
  }

  if (seeds.empty())
    return excluded;

  std::vector<detail::ClosureEntry> entries;
  std::string err;
  if (!detail::buildClosure(scene,
          seeds,
          detail::lightRigPolicy(),
          detail::ObjectKey{},
          entries,
          err)) {
    tsd::core::logWarning(
        "[save_Scene] could not compute light-rig exclusion closure (%s); "
        "saving the whole scene inline",
        err.c_str());
    ok = false;
    return {};
  }

  for (const auto &e : entries)
    excluded.push_back(e.source);

  // Rescue any excluded array that a retained object still references. Repeat to
  // a fixed point so a rescued array's own references are honored too.
  bool changed = true;
  while (changed) {
    changed = false;
    forEachSceneObject(scene, [&](const Object &obj) {
      if (keySetContains(excluded, detail::makeKey(obj.type(), obj.index())))
        return; // an excluded object's references do not keep things alive
      forEachObjectRef(obj, [&](const Any &ref) {
        const auto key = detail::makeKey(ref);
        auto it = std::find_if(excluded.begin(),
            excluded.end(),
            [&](const auto &k) { return detail::sameKey(k, key); });
        if (it != excluded.end()) {
          excluded.erase(it);
          changed = true;
        }
      });
    });
  }

  return excluded;
}

// Dense per-pool index assigned to a retained object on reload.
struct ObjectIndexRemap
{
  detail::ObjectKey key;
  size_t newIndex{0};
};

// Mirror the objectPoolToNode write order to assign each retained object the
// dense per-pool index it will receive when load_Scene recreates the pool.
static std::vector<ObjectIndexRemap> buildObjectRemap(
    Scene &scene, const std::vector<detail::ObjectKey> &excluded)
{
  std::vector<ObjectIndexRemap> remap;
  auto addPool = [&](auto &pool) {
    size_t newIndex = 0;
    foreach_item_const(pool, [&](const auto *o) {
      if (!o)
        return;
      const Object &obj = *static_cast<const Object *>(o);
      const auto key = detail::makeKey(obj.type(), obj.index());
      if (keySetContains(excluded, key))
        return;
      remap.push_back({key, newIndex++});
    });
  };
  auto &db = scene.objectDB();
  addPool(db.geometry);
  addPool(db.sampler);
  addPool(db.material);
  addPool(db.surface);
  addPool(db.field);
  addPool(db.volume);
  addPool(db.light);
  addPool(db.camera);
  addPool(db.renderer);
  addPool(db.array);
  return remap;
}

// Rewrite every scalar object reference (and object self-id) in a serialized
// subtree from its current Scene index to the dense reload index.
static void remapObjectRefs(
    core::DataNode &root, const std::vector<ObjectIndexRemap> &remap)
{
  root.traverse([&](core::DataNode &node, int) {
    if (!node.holdsObjectIdx())
      return true;

    anari::DataType type = ANARI_UNKNOWN;
    size_t index = tsd::core::INVALID_INDEX;
    node.getValueAsObjectIdx(&type, &index);

    const auto key = detail::makeKey(type, index);
    auto it = std::find_if(remap.begin(), remap.end(), [&](const auto &e) {
      return detail::sameKey(e.key, key);
    });
    if (it == remap.end()) {
      tsd::core::logError(
          "[save_Scene] dropping serialized reference to excluded object %s @%zu",
          anari::toString(type),
          index);
      return true;
    }
    node.setValue(Any(type, it->newIndex));
    return true;
  });
}

// Look up the dense reload index for a (type, index) object; returns
// INVALID_INDEX if the object was excluded or has no mapping.
static size_t remappedObjectIndex(const std::vector<ObjectIndexRemap> &remap,
    anari::DataType type,
    size_t index)
{
  if (index == tsd::core::INVALID_INDEX || type == ANARI_UNKNOWN)
    return index;
  const auto key = detail::makeKey(type, index);
  auto it = std::find_if(remap.begin(), remap.end(), [&](const auto &e) {
    return detail::sameKey(e.key, key);
  });
  return it == remap.end() ? tsd::core::INVALID_INDEX : it->newIndex;
}

// Per-layer remap of live layer-node indices to the dense indices the nodes
// receive when nodeToLayer rebuilds the (pruned) layer in document order.
struct LayerNodeRemap
{
  std::string layerName;
  std::vector<std::pair<size_t, size_t>> indices; // live -> reload
};

// Mirror layerSubtreeToNodeImpl's emit order (pre-order, skipping excluded
// subtrees) to compute each retained node's reload index. The root is index 0,
// matching the freshly constructed layer root that nodeToLayer reuses.
static std::vector<LayerNodeRemap> buildLayerNodeRemaps(
    Scene &scene, const std::vector<LayerNodeRef> &excludedNodes)
{
  auto isExcluded = [&](const LayerNode &tsdNode) {
    return std::any_of(excludedNodes.begin(),
        excludedNodes.end(),
        [&](LayerNodeRef r) { return r && &(*r) == &tsdNode; });
  };

  std::vector<LayerNodeRemap> remaps;
  for (auto l : scene.layers()) {
    if (!l.second.ptr)
      continue;
    LayerNodeRemap remap;
    remap.layerName = l.first.c_str();
    size_t reloadIndex = 0;
    l.second.ptr->traverse_const(
        l.second.ptr->root(), [&](const LayerNode &tsdNode, int) {
          if (isExcluded(tsdNode))
            return false; // pruned: not emitted, not counted
          remap.indices.emplace_back(tsdNode.index(), reloadIndex++);
          return true;
        });
    remaps.push_back(std::move(remap));
  }
  return remaps;
}

static size_t remappedLayerNodeIndex(const std::vector<LayerNodeRemap> &remaps,
    const std::string &layerName,
    size_t index)
{
  for (const auto &r : remaps) {
    if (r.layerName != layerName)
      continue;
    for (const auto &pair : r.indices) {
      if (pair.first == index)
        return pair.second;
    }
    return tsd::core::INVALID_INDEX; // node was pruned
  }
  return index; // layer not remapped
}

// Animation bindings store absolute object/layer-node indices as plain scalars
// (see ObjectParameterBinding/TransformBinding/*FileBinding::toDataNode), so the
// generic ref remap cannot reach them. Rewrite them here to match the dense
// indices the excluded payload reloads with.
static void remapAnimationBindings(core::DataNode &animationsNode,
    const std::vector<ObjectIndexRemap> &objectRemap,
    const std::vector<LayerNodeRemap> &layerRemaps)
{
  auto *objects = animationsNode.child("objects");
  if (!objects)
    return;

  objects->foreach_child([&](core::DataNode &anim) {
    if (auto *bindings = anim.child("objectBindings")) {
      bindings->foreach_child([&](core::DataNode &b) {
        const auto type =
            static_cast<anari::DataType>(b["targetType"].getValueOr<int>(
                ANARI_UNKNOWN));
        const auto idx =
            b["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
        b["targetIndex"] = remappedObjectIndex(objectRemap, type, idx);
      });
    }

    if (auto *bindings = anim.child("fileBindings")) {
      bindings->foreach_child([&](core::DataNode &b) {
        // Currently only spatialField file bindings carry an object target.
        if (auto *idxNode = b.child("targetIndex")) {
          const auto idx = idxNode->getValueOr<size_t>(tsd::core::INVALID_INDEX);
          b["targetIndex"] =
              remappedObjectIndex(objectRemap, ANARI_VOLUME, idx);
        }
      });
    }

    if (auto *bindings = anim.child("transformBindings")) {
      bindings->foreach_child([&](core::DataNode &b) {
        auto *layerNode = b.child("layerName");
        auto *idxNode = b.child("nodeIndex");
        if (!layerNode || !idxNode)
          return;
        const auto layerName = layerNode->getValueAs<std::string>();
        const auto idx = idxNode->getValueOr<size_t>(tsd::core::INVALID_INDEX);
        b["nodeIndex"] = remappedLayerNodeIndex(layerRemaps, layerName, idx);
      });
    }
  });
}

void save_Scene(Scene &scene, const char *filename)
{
  tsd::core::logStatus("Saving context to file: %s", filename);
  tsd::core::logStatus("  ...serializing context");
  core::DataTree tree;
  save_Scene(scene, tree.root(), false);
  tsd::core::logStatus("  ...writing file");
  tree.save(filename);
  tsd::core::logStatus("  ...done!");
}

void save_Scene(Scene &scene,
    core::DataNode &root,
    bool forceProxyArrays,
    tsd::animation::AnimationManager *animMgr)
{
  SaveSceneOptions options;
  options.forceProxyArrays = forceProxyArrays;
  options.animMgr = animMgr;
  save_Scene(scene, root, options);
}

void save_Scene(Scene &scene, core::DataNode &root, const SaveSceneOptions &options)
{
  core::writeDataTreeMetadata(root,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene",
          std::string(schema::SCENE_FULL),
          1});

  scene.defragmentObjectStorage(); // ensure contiguous object indices

  // Compute the exclusion. If the closure cannot be built we exclude nothing
  // (objects AND subtrees both inline) so the payload stays self-consistent.
  bool exclusionOk = true;
  std::vector<detail::ObjectKey> excluded;
  if (!options.excludedSubtrees.empty())
    excluded =
        computeExcludedObjects(scene, options.excludedSubtrees, exclusionOk);
  const bool doExclude = !options.excludedSubtrees.empty() && exclusionOk;

  const std::vector<detail::ObjectKey> *excludedObjects =
      (doExclude && !excluded.empty()) ? &excluded : nullptr;
  const std::vector<LayerNodeRef> *excludedNodes =
      doExclude ? &options.excludedSubtrees : nullptr;

  // Layers //

  tsd::core::logStatus("    ...serializing %zu layers", scene.numberOfLayers());

  auto &layersRoot = root["layers"];
  for (auto l : scene.layers()) {
    if (l.second.ptr) {
      auto &layerRoot = layersRoot[l.first.c_str()];
      layerSubtreeToNodeImpl(
          *l.second.ptr, l.second.ptr->root(), layerRoot, excludedNodes);
      layerRoot["isActive"] = l.second.active;
    }
  }

  // ObjectDB //

  const bool fp = options.forceProxyArrays;
  auto &db = scene.objectDB();
  auto &objectDB = root["objectDB"];
  objectPoolToNode(objectDB, db.geometry, "geometry", fp, excludedObjects);
  objectPoolToNode(objectDB, db.sampler, "sampler", fp, excludedObjects);
  objectPoolToNode(objectDB, db.material, "material", fp, excludedObjects);
  objectPoolToNode(objectDB, db.surface, "surface", fp, excludedObjects);
  objectPoolToNode(objectDB, db.field, "spatialfield", fp, excludedObjects);
  objectPoolToNode(objectDB, db.volume, "volume", fp, excludedObjects);
  objectPoolToNode(objectDB, db.light, "light", fp, excludedObjects);
  objectPoolToNode(objectDB, db.camera, "camera", fp, excludedObjects);
  objectPoolToNode(objectDB, db.renderer, "renderer", fp, excludedObjects);
  objectPoolToNode(objectDB, db.array, "array", fp, excludedObjects);

  // Animations are serialized before the remap so the remap can rewrite the
  // absolute object/layer-node indices the bindings store.
  if (options.animMgr)
    animationManagerToNode(*options.animMgr, root["animations"]);

  // Excluding objects and pruning subtrees shifts retained per-pool object
  // indices and layer-node indices; rewrite every serialized reference, object
  // self-id, and animation binding to the dense index it reloads with.
  if (doExclude) {
    const auto objectRemap = buildObjectRemap(scene, excluded);
    remapObjectRefs(objectDB, objectRemap);
    remapObjectRefs(layersRoot, objectRemap);
    if (auto *animationsNode = root.child("animations")) {
      const auto layerRemaps =
          buildLayerNodeRemaps(scene, options.excludedSubtrees);
      remapAnimationBindings(*animationsNode, objectRemap, layerRemaps);
    }
  }
}

void save_SceneCamerasAndRenderers(Scene &scene, const char *filename)
{
  tsd::core::logStatus(
      "Saving scene cameras and renderers to file: %s", filename);
  core::DataTree tree;
  save_SceneCamerasAndRenderers(scene, tree.root());
  if (!tree.save(filename))
    tsd::core::logError(
        "[save_SceneCamerasAndRenderers] failed to write file '%s'", filename);
}

void save_SceneCamerasAndRenderers(Scene &scene, core::DataNode &root)
{
  root.reset();
  core::writeDataTreeMetadata(root,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene-subset",
          std::string(schema::SCENE_CAMERAS_AND_RENDERERS),
          1});
  scene.defragmentObjectStorage(); // ensure contiguous object indices

  auto &objectDB = root["objectDB"];
  objectPoolToNode(objectDB, scene.m_db.camera, "camera", false);
  objectPoolToNode(objectDB, scene.m_db.renderer, "renderer", false);
}

void load_Scene(Scene &scene,
    const char *filename,
    tsd::animation::AnimationManager *animMgr)
{
  tsd::core::logStatus("Loading context from file: %s", filename);
  tsd::core::logStatus("  ...loading file");
  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError("[load_Scene] failed to load file '%s'", filename);
    return;
  }
  load_Scene(scene, tree.root(), animMgr);
}

void load_Scene(Scene &scene,
    core::DataNode &root,
    tsd::animation::AnimationManager *animMgr)
{
  PayloadValidationResult result;
  tryLoad_Scene(scene, root, &result, animMgr);
  if (!result.accepted())
    logValidationFailure("load_Scene", result);
}

PayloadValidationResult validate_ScenePayload(core::DataNode &root)
{
  return validateScenePayloadImpl(
      root, {schema::SCENE_FULL}, {schema::SCENE_FULL,
                                      schema::SCENE_CAMERAS_AND_RENDERERS});
}

bool tryLoad_Scene(Scene &scene,
    core::DataNode &root,
    PayloadValidationResult *resultOut,
    tsd::animation::AnimationManager *animMgr)
{
  // Clear out any existing context contents //
  auto result = validate_ScenePayload(root);
  if (resultOut)
    *resultOut = result;
  if (!result.accepted())
    return false;

  auto &payloadRoot = resolveScenePayloadRoot(root);

  tsd::core::logStatus("  ...clearing old context");

  scene.removeAllObjects();

  // Load data from file (objects then layer) //

  // ObjectDB

  tsd::core::logStatus("  ...converting objects");

  auto &objectDB = payloadRoot["objectDB"];
  auto nodeToObjectPool =
      [](core::DataNode &node, Scene &scene, const char *childNodeName) {
        auto &objectsNode = node[childNodeName];
        objectsNode.foreach_child([&](auto &n) { nodeToNewObject(scene, n); });
      };

  nodeToObjectPool(objectDB, scene, "array");
  nodeToObjectPool(objectDB, scene, "sampler");
  nodeToObjectPool(objectDB, scene, "material");
  nodeToObjectPool(objectDB, scene, "geometry");
  nodeToObjectPool(objectDB, scene, "surface");
  nodeToObjectPool(objectDB, scene, "spatialfield");
  nodeToObjectPool(objectDB, scene, "volume");
  nodeToObjectPool(objectDB, scene, "light");
  nodeToObjectPool(objectDB, scene, "camera");
  nodeToObjectPool(objectDB, scene, "renderer");

  // Layers

  tsd::core::logStatus("  ...converting layers");

  auto &layerRoot = payloadRoot["layers"];
  layerRoot.foreach_child([&](auto &nLayer) {
    tsd::core::Token layerName = nLayer.name().c_str();
    auto &tLayer = *scene.addLayer(layerName);
    nodeToLayer(nLayer, tLayer, scene);
    bool active = true;
    nLayer["isActive"].getValue(ANARI_BOOL, &active);
    scene.setLayerActive(layerName, active);
    scene.signalLayerStructureChanged(&tLayer);
  });

  scene.m_numActiveLayers = 0;
  for (auto &ls : scene.layers()) {
    if (ls.second.active)
      scene.m_numActiveLayers++;
  }

  scene.signalActiveLayersChanged();

  // Animations

  if (animMgr)
    nodeToAnimationManager(payloadRoot["animations"], *animMgr, scene);

  tsd::core::logStatus("  ...done!");
  return true;
}

void load_SceneCamerasAndRenderers(Scene &scene, const char *filename)
{
  tsd::core::logStatus(
      "Loading scene cameras and renderers from file: %s", filename);
  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError(
        "[load_SceneCamerasAndRenderers] failed to load file '%s'", filename);
    return;
  }

  load_SceneCamerasAndRenderers(scene, tree.root());
}

void load_SceneCamerasAndRenderers(Scene &scene, core::DataNode &root)
{
  PayloadValidationResult result;
  tryLoad_SceneCamerasAndRenderers(scene, root, &result);
  if (!result.accepted())
    logValidationFailure("load_SceneCamerasAndRenderers", result);
}

PayloadValidationResult validate_SceneCamerasAndRenderersPayload(
    core::DataNode &root)
{
  return validateScenePayloadImpl(root,
      {schema::SCENE_CAMERAS_AND_RENDERERS, schema::SCENE_FULL},
      {schema::SCENE_FULL, schema::SCENE_CAMERAS_AND_RENDERERS});
}

bool tryLoad_SceneCamerasAndRenderers(Scene &scene,
    core::DataNode &root,
    PayloadValidationResult *resultOut)
{
  auto result = validate_SceneCamerasAndRenderersPayload(root);
  if (resultOut)
    *resultOut = result;
  if (!result.accepted())
    return false;

  auto &payloadRoot = resolveScenePayloadRoot(root);

  auto removeObjects = [&](auto &pool) {
    for (size_t i = pool.capacity(); i-- > 0;) {
      auto obj = pool.at(i);
      if (obj)
        scene.removeObject(obj.data());
    }
  };

  scene.m_defaultObjects.camera.reset();
  removeObjects(scene.m_db.renderer);
  removeObjects(scene.m_db.camera);

  auto &objectDB = payloadRoot["objectDB"];
  auto nodeToObjectPool = [](core::DataNode &node,
                              Scene &scene,
                              const char *childNodeName) {
    auto &objectsNode = node[childNodeName];
    objectsNode.foreach_child([&](auto &n) { nodeToNewObject(scene, n); });
  };

  nodeToObjectPool(objectDB, scene, "camera");
  nodeToObjectPool(objectDB, scene, "renderer");

  scene.m_defaultObjects.camera.reset();
  scene.defaultCamera();

  tsd::core::logStatus("  ...done!");
  return true;
}

} // namespace tsd::io
