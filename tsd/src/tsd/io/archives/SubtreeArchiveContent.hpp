// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/Any.hpp"
#include "tsd/core/TypeMacros.hpp"
// tsd_io
#include "tsd/io/archives/ArchivePolicies.hpp"
#include "tsd/io/archives/ArchiveValidation.hpp"
// tsd_scene
#include "tsd/scene/Layer.hpp"
// std
#include <string>
#include <string_view>
#include <vector>

namespace tsd::animation {
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::io {

struct SubtreeArchiveContentDesc
{
  std::string_view fileType;
  std::string_view schema;
  ArchiveObjectPolicy objectPolicy{ArchiveObjectPolicy::All};
};

struct SubtreeArchiveContentOptions
{
  animation::AnimationManager *animationManager{nullptr};
  FileBindingArchivePolicy fileBindings{FileBindingArchivePolicy::Include};
};

struct SubtreeArchiveResult
{
  SubtreeArchiveResult() = default;
  TSD_NOT_COPYABLE(SubtreeArchiveResult)
  TSD_DEFAULT_MOVEABLE(SubtreeArchiveResult)

  bool valid() const;

  scene::LayerNodeRef root;
  // Exact resources created during deserialization. Animation indices remain
  // valid only while the manager is not structurally changed before rollback.
  std::vector<core::Any> createdObjects;
  std::vector<size_t> createdAnimations;

 private:
  bool m_succeeded{false};

  friend SubtreeArchiveResult deserialize_SubtreeArchiveContent(scene::Scene &,
      core::DataNode &,
      scene::LayerNodeRef,
      const SubtreeArchiveContentDesc &,
      std::string *,
      const SubtreeArchiveContentOptions &);
  friend void rollback_SubtreeArchiveContent(
      scene::Scene &, animation::AnimationManager &, SubtreeArchiveResult &);
};

bool serialize_SubtreeArchiveContent(scene::LayerNodeRef root,
    core::DataNode &archive,
    const SubtreeArchiveContentDesc &desc,
    std::string_view displayName = {},
    const SubtreeArchiveContentOptions &options = {});
ArchiveValidationResult validate_SubtreeArchiveContent(
    core::DataNode &archive, const SubtreeArchiveContentDesc &desc);
SubtreeArchiveResult deserialize_SubtreeArchiveContent(scene::Scene &scene,
    core::DataNode &archive,
    scene::LayerNodeRef destinationParent,
    const SubtreeArchiveContentDesc &desc,
    std::string *displayNameOut = nullptr,
    const SubtreeArchiveContentOptions &options = {});
void rollback_SubtreeArchiveContent(scene::Scene &scene,
    animation::AnimationManager &animationManager,
    SubtreeArchiveResult &result);

} // namespace tsd::io
