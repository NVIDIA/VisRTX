// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LightRigIO.h"

#include "ProjectSerialization.h"

#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/SubtreeArchiveContent.hpp"
#include "tsd/scene/Scene.hpp"

namespace tsd::scivis_studio {

namespace {

const tsd::io::SubtreeArchiveContentDesc LIGHT_RIG_ARCHIVE_DESC{
    LIGHT_RIG_FILE_TYPE,
    LIGHT_RIG_SCHEMA,
    tsd::io::ArchiveObjectPolicy::LightsOnly};

} // namespace

tsd::io::ArchiveValidationResult validateLightRigArchive(
    tsd::core::DataNode &archive)
{
  return tsd::io::validate_SubtreeArchiveContent(
      archive, LIGHT_RIG_ARCHIVE_DESC);
}

bool saveLightRigArchiveFile(tsd::scene::LayerNodeRef root,
    const std::filesystem::path &file,
    std::string_view displayName)
{
  tsd::core::DataTree tree;
  return tsd::io::serialize_SubtreeArchiveContent(
             root, tree.root(), LIGHT_RIG_ARCHIVE_DESC, displayName)
      && tree.save(file.string().c_str());
}

tsd::scene::LayerNodeRef deserializeLightRigArchive(tsd::scene::Scene &scene,
    tsd::core::DataNode &archive,
    tsd::scene::LayerNodeRef destination,
    std::string *displayName)
{
  return tsd::io::deserialize_SubtreeArchiveContent(
      scene, archive, destination, LIGHT_RIG_ARCHIVE_DESC, displayName)
      .root;
}

tsd::scene::LayerNodeRef loadLightRigArchiveFile(tsd::scene::Scene &scene,
    const std::filesystem::path &file,
    tsd::scene::LayerNodeRef destination,
    std::string *displayName)
{
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str()))
    return {};
  return deserializeLightRigArchive(
      scene, tree.root(), destination, displayName);
}

} // namespace tsd::scivis_studio
