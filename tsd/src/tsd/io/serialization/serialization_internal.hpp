// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_io
#include "tsd/io/archives/ArchiveValidation.hpp"
#include "tsd/io/archives/detail/ArchivePlan.hpp"
#include "tsd/io/serialization.hpp"
// tsd_core
#include "tsd/core/TypeMacros.hpp"
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

namespace schema {

inline constexpr std::string_view SCENE_FULL = "tsd.scene.full";
inline constexpr std::string_view SCENE_CAMERAS_AND_RENDERERS =
    "tsd.scene.cameras-and-renderers";
inline constexpr std::string_view SCENE_CAMERAS = "tsd.scene.cameras";
inline constexpr std::string_view SCENE_RENDERERS = "tsd.scene.renderers";
inline constexpr std::string_view OBJECT_SURFACE = "tsd.object.surface";
inline constexpr std::string_view OBJECT_VOLUME = "tsd.object.volume";
inline constexpr std::string_view LAYER_SUBTREE = "tsd.layer.subtree";

} // namespace schema

using PayloadValidationStatus = ArchiveValidationStatus;

namespace detail {

void serialize_LayerSubtree(const scene::Layer &layer,
    scene::LayerNodeRef start,
    core::DataNode &node,
    const std::vector<scene::LayerNodeRef> *excluded);

} // namespace detail

// clang-format off

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
struct SubtreeArchiveContentDesc
{
  std::string_view fileType;
  std::string_view schema;
  ArchiveObjectPolicy objectPolicy{ArchiveObjectPolicy::All};
};
using SubtreeIODesc = SubtreeArchiveContentDesc;

// Optional animation participation for subtree files. When a manager is
// supplied, export carries every animation whose targets are wholly owned by
// the subtree and rejects animations that span the subtree and the surrounding
// scene. Import appends the serialized animations to the supplied manager.
// File-binding animations can be omitted when a higher-level format persists
// their authoritative definition separately and recreates them at runtime.
struct SubtreeArchiveContentOptions
{
  animation::AnimationManager *animationManager{nullptr};
  FileBindingArchivePolicy fileBindings{FileBindingArchivePolicy::Include};
};
using SubtreeIOOptions = SubtreeArchiveContentOptions;

struct SubtreeArchiveResult
{
  SubtreeArchiveResult() = default;
  TSD_NOT_COPYABLE(SubtreeArchiveResult)
  TSD_DEFAULT_MOVEABLE(SubtreeArchiveResult)

  bool valid() const;

  LayerNodeRef root;
  // Exact resources created by this import. Animation indices assume the
  // manager is not structurally changed before rollback_SubtreeImport().
  std::vector<core::Any> createdObjects;
  std::vector<size_t> createdAnimations;

 private:
  bool m_succeeded{false};

  friend SubtreeArchiveResult import_SubtreeWithOwnership(Scene &,
      const char *,
      LayerNodeRef,
      const SubtreeIODesc &,
      std::string *,
      const SubtreeIOOptions &);
  friend SubtreeArchiveResult deserialize_SubtreeArchiveContent(Scene &,
      core::DataNode &,
      LayerNodeRef,
      const SubtreeIODesc &,
      std::string *,
      const SubtreeIOOptions &);
  friend void rollback_SubtreeImport(
      Scene &, animation::AnimationManager &, SubtreeArchiveResult &);
};
using SubtreeImportResult = SubtreeArchiveResult;

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

// Reusable subtree-content composition for application-owned Archives. These
// operations use a caller-supplied envelope and do not perform file I/O.
bool serialize_SubtreeArchiveContent(LayerNodeRef root,
    core::DataNode &archive,
    const SubtreeArchiveContentDesc &desc,
    std::string_view displayName = {},
    const SubtreeArchiveContentOptions &options = {});
SubtreeArchiveResult deserialize_SubtreeArchiveContent(Scene &scene,
    core::DataNode &archive,
    LayerNodeRef destinationParent,
    const SubtreeArchiveContentDesc &desc,
    std::string *displayNameOut = nullptr,
    const SubtreeArchiveContentOptions &options = {});

// clang-format on

} // namespace tsd::io
