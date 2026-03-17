// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/serialization.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"

namespace tsd::io {

void import_file(Scene &scene,
    tsd::animation::SceneAnimation &sceneAnim,
    const ImportFile &f,
    tsd::scene::LayerNodeRef root)
{
  tsd::core::TransferFunction tf;
  import_file(scene, sceneAnim, f, tf, root);
}

void import_file(Scene &scene,
    tsd::animation::SceneAnimation &sceneAnim,
    const ImportFile &f,
    tsd::core::TransferFunction &tf,
    tsd::scene::LayerNodeRef root)
{
  const bool customLocation = root;

  auto files = splitString(f.second, ';');
  std::string file = files[0];
  std::string layerName = files.size() > 1 ? files[1] : "";
  if (layerName.empty())
    layerName = "default";

  if (!customLocation) {
    tsd::core::logStatus(
        "...loading file '%s' in layer '%s'", file.c_str(), layerName.c_str());
    root = scene.addLayer(layerName)->root();
  } else {
    tsd::core::logStatus("...loading file '%s'", file.c_str());
  }

  if (f.first == ImporterType::TSD)
    tsd::io::load_Scene(scene, file.c_str(), &sceneAnim);
  else if (f.first == ImporterType::AGX)
    tsd::io::import_AGX(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::ASSIMP)
    tsd::io::import_ASSIMP(scene, sceneAnim, file.c_str(), root, false);
  else if (f.first == ImporterType::ASSIMP_FLAT)
    tsd::io::import_ASSIMP(scene, sceneAnim, file.c_str(), root, true);
  else if (f.first == ImporterType::AXYZ)
    tsd::io::import_AXYZ(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::DLAF)
    tsd::io::import_DLAF(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::E57XYZ)
    tsd::io::import_E57XYZ(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::ENSIGHT)
    tsd::io::import_ENSIGHT(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::GLTF)
    tsd::io::import_GLTF(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::HDRI)
    tsd::io::import_HDRI(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::HSMESH)
    tsd::io::import_HSMESH(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::NBODY)
    tsd::io::import_NBODY(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::OBJ)
    tsd::io::import_OBJ(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::PDB)
    tsd::io::import_PDB(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::PLY)
    tsd::io::import_PLY(scene, sceneAnim, file.c_str());
  else if (f.first == ImporterType::POINTSBIN_MULTIFILE)
    tsd::io::import_POINTSBIN(scene, sceneAnim, {file.c_str()}, root);
  else if (f.first == ImporterType::PT)
    tsd::io::import_PT(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::SILO)
    tsd::io::import_SILO(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::SMESH)
    tsd::io::import_SMESH(scene, sceneAnim, file.c_str(), root, false);
  else if (f.first == ImporterType::SMESH_ANIMATION)
    tsd::io::import_SMESH(scene, sceneAnim, file.c_str(), root, true);
  else if (f.first == ImporterType::SWC)
    tsd::io::import_SWC(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::TRK)
    tsd::io::import_TRK(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::USD)
    tsd::io::import_USD(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::USD2) {
    tsd::io::import_USD(scene, sceneAnim, file.c_str(), root);
    tsd::io::import_USD2(scene, sceneAnim, file.c_str(), root);
  } else if (f.first == ImporterType::VTP)
    tsd::io::import_VTP(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::VTU)
    tsd::io::import_VTU(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::XYZDP)
    tsd::io::import_XYZDP(scene, sceneAnim, file.c_str(), root);
  else if (f.first == ImporterType::VOLUME)
    tsd::io::import_volume(scene, file.c_str(), tf, root);
  else if (f.first == ImporterType::XF)
    tf = tsd::io::importTransferFunction(file);
  else if (f.first == ImporterType::BLANK) {
    // no-op
  } else {
    tsd::core::logWarning(
        "...skipping unknown file type for '%s'", file.c_str());
  }
}

void import_files(Scene &s,
    tsd::animation::SceneAnimation &sceneAnim,
    const std::vector<ImportFile> &files,
    tsd::scene::LayerNodeRef root)
{
  import_files(s, sceneAnim, files, {}, root);
}

void import_files(Scene &s,
    tsd::animation::SceneAnimation &sceneAnim,
    const std::vector<ImportFile> &files,
    tsd::core::TransferFunction tf,
    tsd::scene::LayerNodeRef root)
{
  if (tf.colorPoints.empty() && tf.opacityPoints.empty()) {
    for (const auto &c : core::colormap::viridis) {
      tf.colorPoints.push_back({float(tf.colorPoints.size())
              / float(core::colormap::viridis.size() - 1),
          c.x,
          c.y,
          c.z});
    }
    tf.opacityPoints = {{0.0f, 0.0f}, {1.0f, 1.0f}};
    tf.range = {};
  }

  for (const auto &f : files)
    import_file(s, sceneAnim, f, tf, root);
}

void import_animations(Scene &scene,
    tsd::animation::SceneAnimation &sceneAnim,
    const std::vector<ImportAnimationFiles> &files,
    tsd::scene::LayerNodeRef root)
{
  for (auto &anim : files) {
    if (anim.second.empty()) {
      tsd::core::logWarning("...skipping animation import for empty file list");
      continue;
    }

    if (anim.first == ImporterType::POINTSBIN_MULTIFILE)
      import_POINTSBIN(scene, sceneAnim, anim.second, root);
    else {
      tsd::core::logWarning("...skipping unknown animation file importer type");
    }
  }
}

} // namespace tsd::io
