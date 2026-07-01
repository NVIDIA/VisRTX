// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/ObjectPool.hpp"
// tsd_io
#include "tsd/io/archives/ArchivePolicies.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <string>
#include <vector>

namespace tsd::animation {
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::io {

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
  scene::LayerNodeRef source;
  size_t archiveIndex{0};
};

struct ArchiveObject
{
  scene::Object *source{nullptr};
  anari::DataType type{ANARI_UNKNOWN};
  size_t sourceIndex{core::INVALID_INDEX};
  size_t archiveIndex{core::INVALID_INDEX};
  bool animationDependency{false};
};

struct ArchiveAnimationDependency
{
  size_t animationIndex{core::INVALID_INDEX};
  scene::Object *object{nullptr};
};

struct ArchivePlan
{
  // This is a snapshot: source refs and indices remain valid only until the
  // Scene or animation manager is structurally mutated.
  scene::LayerNodeRef root;
  std::vector<ArchiveNode> nodes;
  std::vector<ArchiveObject> objects;
  std::vector<ArchiveAnimationDependency> animationDependencies;
  std::vector<size_t> ownedAnimations;
  std::vector<size_t> archivedAnimations;

  bool containsObject(const scene::Object *object) const;
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

ArchivePlanResult plan_SubtreeArchive(scene::Scene &scene,
    scene::LayerNodeRef root,
    const ArchivePlanOptions &options = {});

} // namespace tsd::io
