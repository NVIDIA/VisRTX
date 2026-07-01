// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/CameraArchive.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_io
#include "tsd/io/archives/detail/PoolArchive.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"

namespace tsd::io {

bool serialize_CameraArchive(const scene::Scene &scene, core::DataNode &archive)
{
  return detail::serializePoolArchive(
      scene, archive, ANARI_CAMERA, "camera", schema::SCENE_CAMERAS);
}

ArchiveValidationResult validate_CameraArchive(core::DataNode &archive)
{
  return detail::validatePoolArchive(
      archive, ANARI_CAMERA, "camera", schema::SCENE_CAMERAS);
}

bool deserialize_CameraArchive(scene::Scene &scene,
    core::DataNode &archive,
    ArchiveValidationResult *validation)
{
  auto result = validate_CameraArchive(archive);
  if (validation)
    *validation = result;
  if (!result.accepted())
    return false;

  scene.m_defaultObjects.camera.reset();
  return detail::deserializePoolArchive(scene,
      archive,
      ANARI_CAMERA,
      "camera",
      schema::SCENE_CAMERAS,
      validation);
}

bool save_CameraArchive(const scene::Scene &scene, const char *filename)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return serialize_CameraArchive(scene, tree.root()) && tree.save(filename);
}

bool load_CameraArchive(scene::Scene &scene,
    const char *filename,
    ArchiveValidationResult *validation)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return tree.load(filename)
      && deserialize_CameraArchive(scene, tree.root(), validation);
}

} // namespace tsd::io
