// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/animation/EnSightFileBinding.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/animation/UsdGeometryFileBinding.hpp"
#include "tsd/io/animation/UsdInstancerFileBinding.hpp"
#include "tsd/io/archives/AnimationManagerArchive.hpp"
#include "tsd/io/archives/CameraArchive.hpp"
#include "tsd/io/archives/RendererArchive.hpp"
#include "tsd/io/archives/SceneArchive.hpp"
#include "tsd/io/archives/detail/AnimationRemap.hpp"
#include "tsd/io/archives/detail/ArchiveClosure.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
// std
#include <algorithm>
#include <vector>

namespace tsd::io {

static core::DataNode &resolveScenePayloadRoot(core::DataNode &root)
{
  if (auto *context = root.child("context"))
    return *context;
  return root;
}

static ArchiveValidationResult validateScenePayloadImpl(core::DataNode &root,
    const std::vector<std::string_view> &acceptedSchemas,
    const std::vector<std::string_view> &knownSchemas)
{
  auto &payloadRoot = resolveScenePayloadRoot(root);
  auto metadataResult = core::readDataTreeMetadata(payloadRoot);

  ArchiveValidationResult result;
  if (metadataResult.malformed()) {
    result.status = ArchiveValidationStatus::MalformedMetadata;
    result.message = metadataResult.message;
    return result;
  }

  if (metadataResult.found()) {
    const auto &metadata = *metadataResult.metadata;
    result.fileType = metadata.fileType;
    result.schema = metadata.schema;
    result.envelopeVersion = metadata.envelopeVersion;
    result.schemaVersion = metadata.schemaVersion;

    if (metadata.envelopeVersion != core::DATA_TREE_METADATA_ENVELOPE_VERSION) {
      result.status = ArchiveValidationStatus::UnsupportedEnvelopeVersion;
      result.message = "expected envelopeVersion 1, got "
          + std::to_string(metadata.envelopeVersion);
      return result;
    }

    const auto schemaMatches = [&](std::string_view schema) {
      return metadata.schema == schema;
    };

    if (std::none_of(
            acceptedSchemas.begin(), acceptedSchemas.end(), schemaMatches)) {
      result.status =
          std::any_of(knownSchemas.begin(), knownSchemas.end(), schemaMatches)
          ? ArchiveValidationStatus::IncompatibleSchema
          : ArchiveValidationStatus::UnknownSchema;
      result.message =
          "schema '" + metadata.schema + "' is not accepted by this loader";
      return result;
    }

    if (metadata.schemaVersion != 1) {
      result.status = ArchiveValidationStatus::UnsupportedSchemaVersion;
      result.message = "schema '" + metadata.schema
          + "' supports version 1..1, got "
          + std::to_string(metadata.schemaVersion);
      return result;
    }
  } else {
    result.status = ArchiveValidationStatus::MissingMetadataAccepted;
    result.message = "payload has no __tsd_metadata node; treating as legacy";
  }

  if (!payloadRoot.child("objectDB")) {
    result.status = ArchiveValidationStatus::MissingRequiredNode;
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
    serialize_Object(*obj, childNode->append(), forceProxyArrays);
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
      } else if (kind == "usdGeometry") {
        UsdGeometryFileBinding::addToAnimation(anim, scene, fbNode);
      } else if (kind == "usdInstancer") {
        UsdInstancerFileBinding::addToAnimation(anim, scene, fbNode);
      } else {
        logWarning("[nodeToAnimation] unknown file binding kind '%s'; skipping",
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

static void rescueRetainedObjectDependencies(
    Scene &scene, std::vector<detail::ObjectKey> &excluded)
{
  // Repeat to a fixed point so a rescued object's own references are retained.
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
}

struct SceneExclusionPlan
{
  bool valid{true};
  std::vector<detail::ObjectKey> objects;
  std::vector<size_t> animations;
  std::vector<ArchiveAnimationDependency> animationDependencies;
};

static SceneExclusionPlan planSceneExclusion(Scene &scene,
    const detail::LegacySceneExclusion &exclusion,
    const animation::AnimationManager *animationManager)
{
  SceneExclusionPlan combined;
  for (auto root : exclusion.roots) {
    ArchivePlanOptions options;
    options.objectPolicy = exclusion.objectPolicy;
    options.animationManager = animationManager;
    auto result = plan_SubtreeArchive(scene, root, options);
    if (!result.accepted()) {
      tsd::core::logWarning(
          "[serializeLegacyScenePayload] could not plan subtree exclusion "
          "(%s); saving the "
          "whole scene inline",
          result.message.c_str());
      combined.valid = false;
      combined.objects.clear();
      combined.animations.clear();
      return combined;
    }

    for (const auto &object : result.plan.objects) {
      const auto key = detail::makeKey(object.type, object.sourceIndex);
      if (!keySetContains(combined.objects, key))
        combined.objects.push_back(key);
    }
    for (const auto animationIndex : result.plan.ownedAnimations) {
      if (std::find(combined.animations.begin(),
              combined.animations.end(),
              animationIndex)
          == combined.animations.end())
        combined.animations.push_back(animationIndex);
    }
    for (const auto &dependency : result.plan.animationDependencies) {
      const auto found = std::find_if(combined.animationDependencies.begin(),
          combined.animationDependencies.end(),
          [&](const auto &existing) {
            return existing.animationIndex == dependency.animationIndex
                && existing.object == dependency.object;
          });
      if (found == combined.animationDependencies.end())
        combined.animationDependencies.push_back(dependency);
    }
  }

  if (exclusion.animations == detail::LegacyExcludedAnimationPolicy::Retain
      && !combined.animations.empty()) {
    tsd::core::logWarning(
        "[serializeLegacyScenePayload] retained animations target excluded "
        "subtrees; saving "
        "the whole scene inline");
    combined.valid = false;
    combined.objects.clear();
    combined.animations.clear();
    return combined;
  }

  // Dependencies of retained animations must remain in the full-scene
  // payload even when an excluded subtree also references them.
  for (const auto &dependency : combined.animationDependencies) {
    if (!dependency.object
        || std::find(combined.animations.begin(),
               combined.animations.end(),
               dependency.animationIndex)
            != combined.animations.end())
      continue;
    const auto key =
        detail::makeKey(dependency.object->type(), dependency.object->index());
    auto found = std::find_if(combined.objects.begin(),
        combined.objects.end(),
        [&](const auto &candidate) { return detail::sameKey(candidate, key); });
    if (found != combined.objects.end())
      combined.objects.erase(found);
  }

  rescueRetainedObjectDependencies(scene, combined.objects);
  return combined;
}

static void animationManagerToNodeExcluding(
    const animation::AnimationManager &manager,
    core::DataNode &node,
    const std::vector<size_t> &excludedAnimations)
{
  node["time"] = manager.getAnimationTime();
  node["increment"] = manager.getAnimationIncrement();
  node["totalFrames"] = manager.getAnimationTotalFrames();
  node["fps"] = manager.getAnimationFPS();
  auto &objects = node["objects"];
  for (size_t i = 0; i < manager.animations().size(); ++i) {
    if (std::find(excludedAnimations.begin(), excludedAnimations.end(), i)
        == excludedAnimations.end())
      animationToNode(manager.animations()[i], objects.append());
  }
}

// Dense per-pool index assigned to a retained object on reload.
struct ObjectIndexRemap
{
  detail::ObjectKey key;
  size_t newIndex{0};
};

// Mirror the objectPoolToNode write order to assign each retained object the
// dense per-pool index it will receive when the payload recreates the pool.
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
          "[serializeLegacyScenePayload] dropping serialized reference to "
          "excluded object %s @%zu",
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
// receive when deserialize_Layer rebuilds the (pruned) layer in document order.
struct LayerNodeRemap
{
  std::string layerName;
  std::vector<std::pair<size_t, size_t>> indices; // live -> reload
};

// Mirror detail::serialize_LayerSubtree's emit order (pre-order, skipping
// excluded subtrees) to compute each retained node's reload index. The root is
// index 0, matching the freshly constructed layer root that deserialize_Layer
// reuses.
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

void detail::serializeLegacyScenePayload(Scene &scene,
    core::DataNode &root,
    const LegacySceneSerializationOptions &options)
{
  core::writeDataTreeMetadata(root,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene",
          std::string(schema::SCENE_FULL),
          1});

  scene.defragmentObjectStorage(); // ensure contiguous object indices

  // Plan each root as an independent archive partition. A mixed animation or
  // any other ownership ambiguity cancels the entire exclusion so the emitted
  // scene remains self-consistent.
  SceneExclusionPlan exclusionPlan;
  if (!options.exclusion.roots.empty()) {
    exclusionPlan =
        planSceneExclusion(scene, options.exclusion, options.animationManager);
  }
  const bool doExclude =
      !options.exclusion.roots.empty() && exclusionPlan.valid;
  const auto &excluded = exclusionPlan.objects;

  const std::vector<detail::ObjectKey> *excludedObjects =
      (doExclude && !excluded.empty()) ? &excluded : nullptr;
  const std::vector<LayerNodeRef> *excludedNodes =
      doExclude ? &options.exclusion.roots : nullptr;

  // Layers //

  tsd::core::logStatus("    ...serializing %zu layers", scene.numberOfLayers());

  auto &layersRoot = root["layers"];
  for (auto l : scene.layers()) {
    if (l.second.ptr) {
      auto &layerRoot = layersRoot[l.first.c_str()];
      detail::serialize_LayerSubtree(
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
  if (options.animationManager) {
    if (doExclude
        && options.exclusion.animations
            == LegacyExcludedAnimationPolicy::OmitOwned) {
      animationManagerToNodeExcluding(*options.animationManager,
          root["animations"],
          exclusionPlan.animations);
    } else
      animationManagerToNode(*options.animationManager, root["animations"]);
  }

  // Excluding objects and pruning subtrees shifts retained per-pool object
  // indices and layer-node indices; rewrite every serialized reference, object
  // self-id, and animation binding to the dense index it reloads with.
  if (doExclude) {
    const auto objectRemap = buildObjectRemap(scene, excluded);
    remapObjectRefs(objectDB, objectRemap);
    remapObjectRefs(layersRoot, objectRemap);
    if (auto *animationsNode = root.child("animations")) {
      const auto layerRemaps =
          buildLayerNodeRemaps(scene, options.exclusion.roots);
      std::string errorMessage;
      if (!detail::remapSceneAnimations(
              *animationsNode,
              [&](anari::DataType type, size_t index) {
                return remappedObjectIndex(objectRemap, type, index);
              },
              [&](const std::string &layerName, size_t index) {
                return remappedLayerNodeIndex(layerRemaps, layerName, index);
              },
              errorMessage)) {
        tsd::core::logError(
            "[serializeLegacyScenePayload] could not remap animations: %s",
            errorMessage.c_str());
      }
    }
  }
}

void detail::serializeLegacyCameraRendererPayload(
    Scene &scene, core::DataNode &root)
{
  root.reset();
  core::writeDataTreeMetadata(root,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene-subset",
          std::string(schema::SCENE_CAMERAS_AND_RENDERERS),
          1});
  scene.defragmentObjectStorage(); // ensure contiguous object indices

  auto &objectDB = root["objectDB"];
  const auto &db = scene.objectDB();
  objectPoolToNode(objectDB, db.camera, "camera", false);
  objectPoolToNode(objectDB, db.renderer, "renderer", false);
}

ArchiveValidationResult detail::validateLegacyScenePayload(core::DataNode &root)
{
  return validate_SceneArchive(root);
}

bool detail::tryDeserializeLegacyScenePayload(Scene &scene,
    core::DataNode &root,
    ArchiveValidationResult *resultOut,
    tsd::animation::AnimationManager *animMgr)
{
  if (!deserialize_SceneArchive(scene, root, resultOut))
    return false;

  auto &payloadRoot = resolveScenePayloadRoot(root);
  if (animMgr) {
    if (auto *animations = payloadRoot.child("animations"))
      return deserialize_AnimationManagerArchive(*animMgr, *animations);
  }
  return true;
}

ArchiveValidationResult detail::validateLegacyCameraRendererPayload(
    core::DataNode &root)
{
  return validateScenePayloadImpl(root,
      {schema::SCENE_CAMERAS_AND_RENDERERS, schema::SCENE_FULL},
      {schema::SCENE_FULL, schema::SCENE_CAMERAS_AND_RENDERERS});
}

bool detail::tryDeserializeLegacyCameraRendererPayload(
    Scene &scene, core::DataNode &root, ArchiveValidationResult *resultOut)
{
  auto result = validateLegacyCameraRendererPayload(root);
  if (resultOut)
    *resultOut = result;
  if (!result.accepted())
    return false;

  auto &payloadRoot = resolveScenePayloadRoot(root);
  auto *objectDB = payloadRoot.child("objectDB");
  if (!objectDB)
    return false;

  core::DataTree cameraArchive;
  core::writeDataTreeMetadata(cameraArchive.root(),
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene-subset",
          std::string(schema::SCENE_CAMERAS),
          1});
  cameraArchive.root()["objectDB"];
  if (auto *camera = objectDB->child("camera"))
    cameraArchive.root()["objectDB"]["camera"] = *camera;

  core::DataTree rendererArchive;
  core::writeDataTreeMetadata(rendererArchive.root(),
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "scene-subset",
          std::string(schema::SCENE_RENDERERS),
          1});
  rendererArchive.root()["objectDB"];
  if (auto *renderer = objectDB->child("renderer"))
    rendererArchive.root()["objectDB"]["renderer"] = *renderer;

  if (!validate_CameraArchive(cameraArchive.root()).accepted()
      || !validate_RendererArchive(rendererArchive.root()).accepted()
      || !deserialize_CameraArchive(scene, cameraArchive.root())
      || !deserialize_RendererArchive(scene, rendererArchive.root())) {
    return false;
  }
  scene.defaultCamera();
  return true;
}

} // namespace tsd::io
