// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/RendererArchive.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_io
#include "tsd/io/archives/detail/PoolArchive.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"

namespace tsd::io {

bool serialize_RendererArchive(
    const scene::Scene &scene, core::DataNode &archive)
{
  return detail::serializePoolArchive(
      scene, archive, ANARI_RENDERER, "renderer", schema::SCENE_RENDERERS);
}

ArchiveValidationResult validate_RendererArchive(core::DataNode &archive)
{
  return detail::validatePoolArchive(
      archive, ANARI_RENDERER, "renderer", schema::SCENE_RENDERERS);
}

bool deserialize_RendererArchive(scene::Scene &scene,
    core::DataNode &archive,
    ArchiveValidationResult *validation)
{
  return detail::deserializePoolArchive(scene,
      archive,
      ANARI_RENDERER,
      "renderer",
      schema::SCENE_RENDERERS,
      validation);
}

bool save_RendererArchive(const scene::Scene &scene, const char *filename)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return serialize_RendererArchive(scene, tree.root()) && tree.save(filename);
}

bool load_RendererArchive(scene::Scene &scene,
    const char *filename,
    ArchiveValidationResult *validation)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return tree.load(filename)
      && deserialize_RendererArchive(scene, tree.root(), validation);
}

} // namespace tsd::io
