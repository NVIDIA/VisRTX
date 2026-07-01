// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_io
#include "tsd/io/archives/ArchiveValidation.hpp"
#include "tsd/io/archives/SubtreeArchiveContent.hpp"
#include "tsd/io/serialization.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
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

namespace detail {

void serialize_LayerSubtree(const scene::Layer &layer,
    scene::LayerNodeRef start,
    core::DataNode &node,
    const std::vector<scene::LayerNodeRef> *excluded);

enum class LegacyExcludedAnimationPolicy
{
  Retain,
  OmitOwned
};

struct LegacySceneExclusion
{
  std::vector<scene::LayerNodeRef> roots;
  ArchiveObjectPolicy objectPolicy{ArchiveObjectPolicy::LightsOnly};
  LegacyExcludedAnimationPolicy animations{
      LegacyExcludedAnimationPolicy::Retain};
};

struct LegacySceneSerializationOptions
{
  bool forceProxyArrays{false};
  animation::AnimationManager *animationManager{nullptr};
  LegacySceneExclusion exclusion;
};

void serializeLegacyScenePayload(scene::Scene &scene,
    core::DataNode &root,
    const LegacySceneSerializationOptions &options = {});
ArchiveValidationResult validateLegacyScenePayload(core::DataNode &root);
bool tryDeserializeLegacyScenePayload(scene::Scene &scene,
    core::DataNode &root,
    ArchiveValidationResult *result = nullptr,
    animation::AnimationManager *animationManager = nullptr);

void serializeLegacyCameraRendererPayload(
    scene::Scene &scene, core::DataNode &root);
ArchiveValidationResult validateLegacyCameraRendererPayload(
    core::DataNode &root);
bool tryDeserializeLegacyCameraRendererPayload(scene::Scene &scene,
    core::DataNode &root,
    ArchiveValidationResult *result = nullptr);

} // namespace detail

// clang-format off

// Animations //

void animationToNode(const animation::Animation &anim, core::DataNode &node);
void nodeToAnimation(core::DataNode &node, animation::Animation &anim, Scene &scene);
void animationManagerToNode(const animation::AnimationManager &mgr, core::DataNode &node);
void nodeToAnimationManager(core::DataNode &node, animation::AnimationManager &mgr, Scene &scene);

// clang-format on

} // namespace tsd::io
