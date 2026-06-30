// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TypeMacros.hpp"
// tsd_rendering
#include "tsd/rendering/view/Manipulator.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <limits>
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

// Subtree archive planning ///////////////////////////////////////////////////

enum class ArchiveObjectPolicy
{
  All,
  LightsOnly
};

enum class FileBindingArchivePolicy
{
  Include,
  Omit
};

enum class ArchivePlanStatus
{
  Valid,
  InvalidSubtree,
  ObjectClosureFailure,
  InvalidAnimationTarget,
  MixedAnimationTargets,
  UnsupportedFileBinding
};

struct ArchiveNode
{
  LayerNodeRef source;
  size_t archiveIndex{0};
};

struct ArchiveObject
{
  Object *source{nullptr};
  anari::DataType type{ANARI_UNKNOWN};
  size_t sourceIndex{tsd::core::INVALID_INDEX};
  size_t archiveIndex{tsd::core::INVALID_INDEX};
  bool animationDependency{false};
};

struct ArchiveAnimationDependency
{
  size_t animationIndex{tsd::core::INVALID_INDEX};
  Object *object{nullptr};
};

struct ArchivePlan
{
  // This is a snapshot: source refs and indices remain valid only until the
  // Scene or animation manager is structurally mutated.
  LayerNodeRef root;
  std::vector<ArchiveNode> nodes;
  std::vector<ArchiveObject> objects;
  std::vector<ArchiveAnimationDependency> animationDependencies;
  std::vector<size_t> ownedAnimations;
  std::vector<size_t> archivedAnimations;

  bool containsObject(const Object *object) const;
};

struct ArchivePlanResult
{
  ArchivePlanStatus status{ArchivePlanStatus::Valid};
  std::string message;
  ArchivePlan plan;

  bool accepted() const;
};

struct ArchivePlanOptions
{
  ArchiveObjectPolicy objectPolicy{ArchiveObjectPolicy::All};
  const animation::AnimationManager *animationManager{nullptr};
  FileBindingArchivePolicy fileBindings{FileBindingArchivePolicy::Include};
};

ArchivePlanResult plan_SubtreeArchive(
    Scene &scene, LayerNodeRef root, const ArchivePlanOptions &options = {});

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

enum class ExcludedAnimationPolicy
{
  Retain,
  OmitOwned
};

struct SceneExclusion
{
  std::vector<LayerNodeRef> roots;
  ArchiveObjectPolicy objectPolicy{ArchiveObjectPolicy::LightsOnly};
  ExcludedAnimationPolicy animations{ExcludedAnimationPolicy::Retain};
};

// Options controlling a full-scene save. Exclusion roots and their complete
// ownership policy travel together, avoiding partially configured boolean
// combinations. Retained indices are densely remapped for reload.
struct SaveSceneOptions
{
  bool forceProxyArrays{false};
  tsd::animation::AnimationManager *animationManager{nullptr};
  SceneExclusion exclusion;
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
// the object closure policy (LightsOnly restricts it to lights + the arrays
// they reference). An optional displayName is stored alongside the payload and
// read back on import. Import splices the subtree under destinationParent and
// returns its new root.
struct SubtreeIODesc
{
  std::string_view fileType;
  std::string_view schema;
  ArchiveObjectPolicy objectPolicy{ArchiveObjectPolicy::All};
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
  FileBindingArchivePolicy fileBindings{FileBindingArchivePolicy::Include};
};

struct SubtreeImportResult
{
  SubtreeImportResult() = default;
  TSD_NOT_COPYABLE(SubtreeImportResult)
  TSD_DEFAULT_MOVEABLE(SubtreeImportResult)

  bool valid() const;

  LayerNodeRef root;
  // Exact resources created by this import. Animation indices assume the
  // manager is not structurally changed before rollback_SubtreeImport().
  std::vector<core::Any> createdObjects;
  std::vector<size_t> createdAnimations;

 private:
  bool m_succeeded{false};

  friend SubtreeImportResult import_SubtreeWithOwnership(Scene &,
      const char *,
      LayerNodeRef,
      const SubtreeIODesc &,
      std::string *,
      const SubtreeIOOptions &);
  friend void rollback_SubtreeImport(
      Scene &, animation::AnimationManager &, SubtreeImportResult &);
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
SubtreeImportResult import_SubtreeWithOwnership(Scene &scene,
    const char *filename,
    LayerNodeRef destinationParent,
    const SubtreeIODesc &desc,
    std::string *displayNameOut = nullptr,
    const SubtreeIOOptions &options = {});
void rollback_SubtreeImport(Scene &scene,
    animation::AnimationManager &animationManager,
    SubtreeImportResult &result);
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
