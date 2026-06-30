// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <limits>
// tsd_rendering
#include "tsd/rendering/view/Manipulator.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <string>
#include <string_view>
#include <vector>

namespace tsd::animation {
struct Animation;
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::io {

using namespace tsd::scene;

// NanoVDB quantization precision options
enum class VDBPrecision
{
  Float32, // No quantization (32-bit float)
  Fp4, // 4-bit fixed-point (~8:1 compression)
  Fp8, // 8-bit fixed-point (~4:1 compression)
  Fp16, // 16-bit fixed-point (~2:1 compression)
  FpN, // Variable bit fixed-point
  Half // IEEE 16-bit half float
};

namespace schema {

inline constexpr std::string_view SCENE_FULL = "tsd.scene.full";
inline constexpr std::string_view SCENE_CAMERAS_AND_RENDERERS =
    "tsd.scene.cameras-and-renderers";
inline constexpr std::string_view OBJECT_SURFACE = "tsd.object.surface";
inline constexpr std::string_view OBJECT_VOLUME = "tsd.object.volume";
inline constexpr std::string_view LAYER_SUBTREE = "tsd.layer.subtree";

} // namespace schema

enum class PayloadValidationStatus
{
  Valid,
  MissingMetadataAccepted,
  UnknownSchema,
  IncompatibleSchema,
  UnsupportedEnvelopeVersion,
  UnsupportedSchemaVersion,
  MalformedMetadata,
  MissingRequiredNode
};

struct PayloadValidationResult
{
  PayloadValidationStatus status{PayloadValidationStatus::Valid};
  std::string fileType;
  std::string schema;
  int envelopeVersion{0};
  int schemaVersion{0};
  std::string message;

  bool accepted() const;
};

inline bool PayloadValidationResult::accepted() const
{
  return status == PayloadValidationStatus::Valid
      || status == PayloadValidationStatus::MissingMetadataAccepted;
}

// clang-format off

// Parameters //

void parameterToNode(const Parameter &p, core::DataNode &node);
void nodeToParameter(core::DataNode &node, Parameter &p);
void nodeToObjectParameters(core::DataNode &node, Object &obj);

// Objects //

void objectToNode(const Object &obj, core::DataNode &node, bool forceArraysAsProxies = false);
void nodeToObject(core::DataNode &node, Object &obj);
void nodeToObjectMetadata(core::DataNode &node, Object &obj);
void nodeToNewObject(Scene &scene, core::DataNode &node);

// Camera poses //

void cameraPoseToNode(const rendering::CameraPose &pose, core::DataNode &node);
void nodeToCameraPose(core::DataNode &node, rendering::CameraPose &pose);

// Layers //

void layerToNode(const Layer &layer, core::DataNode &node);
void layerSubtreeToNode(const Layer &layer, LayerNodeRef start, core::DataNode &node);
void nodeToLayer(core::DataNode &rootNode, Layer &layer, Scene &scene);
void layerNodeInstanceParametersToNode(const LayerNodeData &data, core::DataNode &node);
void nodeToLayerNodeInstanceParameters(core::DataNode &node, LayerNodeData &data);

// Animations //

void animationToNode(const animation::Animation &anim, core::DataNode &node);
void nodeToAnimation(core::DataNode &node, animation::Animation &anim, Scene &scene);
void animationManagerToNode(const animation::AnimationManager &mgr, core::DataNode &node);
void nodeToAnimationManager(core::DataNode &node, animation::AnimationManager &mgr, Scene &scene);

// Scenes + Objects //

// Options controlling a full-scene save. excludedSubtrees names layer-subtree
// roots that should NOT appear in the saved payload: each excluded subtree's
// nodes are dropped from layer serialization, and the lights it contains (plus
// any arrays referenced *only* by those lights, per lightRigPolicy) are dropped
// from the objectDB. Arrays still referenced by retained objects are kept.
// Retained object indices are densely remapped so the payload loads correctly.
// load_Scene is unchanged and needs no knowledge of exclusions.
struct SaveSceneOptions
{
  bool forceProxyArrays{false};
  tsd::animation::AnimationManager *animMgr{nullptr};
  std::vector<LayerNodeRef> excludedSubtrees;
  // Broad closure is required when excluded subtrees may contain arbitrary
  // scene objects (for example standalone dataset assets), rather than only
  // light-rig objects.
  bool excludeFullObjectClosure{false};
  bool excludeAnimationsTargetingSubtrees{false};
};

void save_Scene(Scene &scene, const char *filename);
void save_Scene(Scene &scene, core::DataNode &root, const SaveSceneOptions &options);
void save_Scene(Scene &scene, core::DataNode &root, bool forceProxyArrays, tsd::animation::AnimationManager *animMgr = nullptr);
void load_Scene(Scene &scene, const char *filename, tsd::animation::AnimationManager *animMgr = nullptr);
void load_Scene(Scene &scene, core::DataNode &root, tsd::animation::AnimationManager *animMgr = nullptr);
PayloadValidationResult validate_ScenePayload(core::DataNode &root);
bool tryLoad_Scene(Scene &scene, core::DataNode &root, PayloadValidationResult *result = nullptr, tsd::animation::AnimationManager *animMgr = nullptr);

void save_SceneCamerasAndRenderers(Scene &scene, const char *filename);
void save_SceneCamerasAndRenderers(Scene &scene, core::DataNode &root);
void load_SceneCamerasAndRenderers(Scene &scene, const char *filename);
void load_SceneCamerasAndRenderers(Scene &scene, core::DataNode &root);
PayloadValidationResult validate_SceneCamerasAndRenderersPayload(core::DataNode &root);
bool tryLoad_SceneCamerasAndRenderers(Scene &scene, core::DataNode &root, PayloadValidationResult *result = nullptr);

bool export_Object(const char *filename, const Object &obj);
Object *import_Object(Scene &scene, const char *filename);
SurfaceRef import_Surface(Scene &scene, const char *filename);
VolumeRef import_Volume(Scene &scene, const char *filename);
PayloadValidationResult validate_ObjectPayload(core::DataNode &root);
PayloadValidationResult validate_SurfacePayload(core::DataNode &root);
PayloadValidationResult validate_VolumePayload(core::DataNode &root);

// Layer subtrees //

bool export_LayerSubtree(const char *filename, LayerNodeRef root);
LayerNodeRef import_LayerSubtree(Scene &scene, const char *filename, LayerNodeRef destinationParent = {});
PayloadValidationResult validate_LayerSubtreePayload(core::DataNode &root);

// Generalized subtree IO shared by layer-subtree export and app-specific rig
// files. The caller supplies the metadata envelope's fileType/schema and chooses
// the object closure policy (lightsOnly restricts it to lights + the arrays they
// reference). An optional displayName is stored alongside the payload and read
// back on import. Import splices the subtree under destinationParent and returns
// its new root.
struct SubtreeIODesc
{
  std::string_view fileType;
  std::string_view schema;
  bool lightsOnly{false};
};

// Optional animation participation for subtree files. When a manager is
// supplied, export carries every animation whose targets are wholly owned by
// the subtree and rejects animations that span the subtree and the surrounding
// scene. Import appends the serialized animations to the supplied manager.
// File-binding animations can be omitted when a higher-level format persists
// their authoritative definition separately and recreates them at runtime.
struct SubtreeIOOptions
{
  animation::AnimationManager *animationManager{nullptr};
  bool includeFileBindingAnimations{true};
};

bool export_Subtree(const char *filename,
    LayerNodeRef root,
    const SubtreeIODesc &desc,
    std::string_view displayName = {},
    const SubtreeIOOptions &options = {});
LayerNodeRef import_Subtree(Scene &scene,
    const char *filename,
    LayerNodeRef destinationParent,
    const SubtreeIODesc &desc,
    std::string *displayNameOut = nullptr,
    const SubtreeIOOptions &options = {});
PayloadValidationResult validate_SubtreePayload(core::DataNode &root, const SubtreeIODesc &desc);

void export_SceneToUSD(
    Scene &scene, const char *filename, int framesPerSecond = 30, tsd::animation::AnimationManager *animMgr = nullptr);
void export_StructuredVolumeToNanoVDB(
  const SpatialField* spatialField,
  std::string_view outputFilename,
  bool useUndefinedValue = false,
  float undefinedValue = std::numeric_limits<float>::quiet_NaN(),
  VDBPrecision precision = VDBPrecision::Fp16,
  bool enableDithering = false
);

// clang-format on

} // namespace tsd::io
