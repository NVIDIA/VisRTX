// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/SpatialFieldFileBinding.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/serialization.hpp"
// tsd_animation
#include "tsd/animation/AnimationManager.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"

namespace tsd::io {

void import_file(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const ImportFile &f,
    tsd::scene::LayerNodeRef root)
{
  tsd::core::TransferFunction tf;
  import_file(scene, animMgr, f, tf, root);
}

void import_file(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
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
    tsd::io::load_Scene(scene, file.c_str(), &animMgr);
  else if (f.first == ImporterType::AGX)
    tsd::io::import_AGX(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::ASSIMP)
    tsd::io::import_ASSIMP(scene, animMgr, file.c_str(), root, false);
  else if (f.first == ImporterType::ASSIMP_FLAT)
    tsd::io::import_ASSIMP(scene, animMgr, file.c_str(), root, true);
  else if (f.first == ImporterType::AXYZ)
    tsd::io::import_AXYZ(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::DLAF)
    tsd::io::import_DLAF(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::E57XYZ)
    tsd::io::import_E57XYZ(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::ENSIGHT)
    tsd::io::import_ENSIGHT(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::GLTF)
    tsd::io::import_GLTF(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::HDRI)
    tsd::io::import_HDRI(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::HSMESH)
    tsd::io::import_HSMESH(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::NBODY)
    tsd::io::import_NBODY(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::OBJ)
    tsd::io::import_OBJ(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::PDB)
    tsd::io::import_PDB(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::PLY)
    tsd::io::import_PLY(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::POINTSBIN_MULTIFILE)
    tsd::io::import_POINTSBIN(scene, animMgr, {file.c_str()}, root);
  else if (f.first == ImporterType::PT)
    tsd::io::import_PT(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::SILO)
    tsd::io::import_SILO(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::SMESH)
    tsd::io::import_SMESH(scene, animMgr, file.c_str(), root, false);
  else if (f.first == ImporterType::SMESH_ANIMATION)
    tsd::io::import_SMESH(scene, animMgr, file.c_str(), root, true);
  else if (f.first == ImporterType::SWC)
    tsd::io::import_SWC(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::TRK)
    tsd::io::import_TRK(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::USD)
    tsd::io::import_USD(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::VTP)
    tsd::io::import_VTP(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::VTU)
    tsd::io::import_VTU(scene, animMgr, file.c_str(), root);
  else if (f.first == ImporterType::XYZDP)
    tsd::io::import_XYZDP(scene, animMgr, file.c_str(), root);
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
    tsd::animation::AnimationManager &animMgr,
    const std::vector<ImportFile> &files,
    tsd::scene::LayerNodeRef root)
{
  import_files(s, animMgr, files, {}, root);
}

void import_files(Scene &s,
    tsd::animation::AnimationManager &animMgr,
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
    import_file(s, animMgr, f, tf, root);
}

void import_animations(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<ImportAnimationFiles> &files,
    tsd::scene::LayerNodeRef root)
{
  for (auto &anim : files) {
    if (anim.second.empty()) {
      tsd::core::logWarning("...skipping animation import for empty file list");
      continue;
    }

    if (anim.first == ImporterType::POINTSBIN_MULTIFILE)
      import_POINTSBIN(scene, animMgr, anim.second, root);
    else if (anim.first == ImporterType::VOLUME_ANIMATION)
      import_volume_animation(scene, animMgr, anim.second, root);
    else {
      tsd::core::logWarning("...skipping unknown animation file importer type");
    }
  }
}

VolumeRef import_volume_animation(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<std::string> &files,
    LayerNodeRef location)
{
  if (files.empty()) {
    logError("[import_volume_animation] file list is empty");
    return {};
  }

  auto field = import_spatial_field(scene, files[0].c_str());
  if (!field) {
    logError("[import_volume_animation] failed to load first frame: '%s'",
        files[0].c_str());
    return {};
  }

  float2 valueRange = field->computeValueRange();

  auto tx = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
      tx, tokens::volume::transferFunction1D);
  volume->setName(fileOf(files[0]).c_str());
  volume->setParameterObject("value", *field);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  SpatialFieldFileBinding binding(&scene, volume.data(), field, files);
  auto &anim = animMgr.addAnimation(fileOf(files[0]).c_str());
  binding.addToAnimation(anim);

  return volume;
}

} // namespace tsd::io
