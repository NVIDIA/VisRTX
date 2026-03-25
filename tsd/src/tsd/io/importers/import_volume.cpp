// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// tsd_animation
#include "tsd/animation/AnimationManager.hpp"
// tsd_io
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <cstdio>

namespace tsd::io {

using namespace tsd::core;

// Helper functions ///////////////////////////////////////////////////////////

static void applyTransferFunction(Scene &scene,
    VolumeRef volume,
    const tsd::core::TransferFunction &transferFunction)
{
  // Build RGBA colors with evenly-spaced positions
  std::vector<tsd::math::float4> colormap;

  constexpr const size_t numRGBPoints = 256;

  for (size_t i = 0; i < numRGBPoints; ++i) {
    float x = (i / float(numRGBPoints - 1));

    auto color = detail::interpolateColor(transferFunction.colorPoints, x);
    auto opacty = detail::interpolateOpacity(transferFunction.opacityPoints, x);
    colormap.push_back({color.x, color.y, color.z, opacty});
  }

  auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, colormap.size());
  colorArray->setData(colormap);
  volume->setParameterObject("color", *colorArray);

  if (transferFunction.range.lower < transferFunction.range.upper)
    volume->setParameter(
        "valueRange", ANARI_FLOAT32_BOX1, &transferFunction.range);

  volume->setMetadataArray("opacityControlPoints",
      ANARI_FLOAT32_VEC2,
      transferFunction.opacityPoints.data(),
      transferFunction.opacityPoints.size());
}

// import_volume definitions //////////////////////////////////////////////////

SpatialFieldRef import_spatial_field(Scene &scene, const char *filepath)
{
  auto ext = extensionOf(filepath);
  if (ext == ".raw")
    return import_RAW(scene, filepath);
  else if (ext == ".flash" || ext == ".hdf5")
    return import_FLASH(scene, filepath);
  else if (ext == ".nvdb")
    return import_NVDB(scene, filepath);
  else if (ext == ".mhd")
    return import_MHD(scene, filepath);
  else if (ext == ".vtu")
    return import_VTU(scene, filepath);
  else if (ext == ".silo" || ext == ".sil")
    return import_SILO(scene, filepath);
  else {
    logError(
        "[import_spatial_field] no loader for file type '%s'", ext.c_str());
    return {};
  }
}

VolumeRef import_volume(
    Scene &scene, const char *filepath, LayerNodeRef location)
{
  auto file = fileOf(filepath);
  auto ext = extensionOf(filepath);

  SpatialFieldRef field;

  if (ext == ".vti") {
    std::vector<SpatialFieldRef> extraFields;
    field = import_VTI(scene, filepath, location, &extraFields);
    // Store extra fields as named object parameters on the Volume so they
    // survive scene GC and are selectable via the ObjectEditor "select" button.
    if (field && !extraFields.empty()) {
      float2 valueRange = field->computeValueRange();
      auto tx = scene.insertChildTransformNode(
          location ? location : scene.defaultLayer()->root());
      auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
          tx, tokens::volume::transferFunction1D);
      volume->setName(fileOf(filepath).c_str());
      volume->setParameterObject("value", *field);
      volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
      for (auto &extra : extraFields)
        volume->setParameterObject(extra->name().c_str(), *extra);
      return volume;
    }
  } else {
    field = import_spatial_field(scene, filepath);
  }
  if (!field) {
    logError(
        "[import_volume] unable to load field from file '%s'", file.c_str());
    return {};
  }

  float2 valueRange = field->computeValueRange();

  auto tx = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
      tx, tokens::volume::transferFunction1D);
  volume->setName(file.c_str());
  volume->setParameterObject("value", *field);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  return volume;
}

VolumeRef import_volume(Scene &scene,
    const char *filepath,
    const TransferFunction &transferFunction,
    LayerNodeRef location)
{
  auto volume = import_volume(scene, filepath, location);
  applyTransferFunction(scene, volume, transferFunction);
  return volume;
}

VolumeRef import_volume_animation(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<std::string> &files,
    const tsd::core::TransferFunction &transferFunction,
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
  applyTransferFunction(scene, volume, transferFunction);

  auto &anim = animMgr.addAnimation(fileOf(files[0]).c_str());
  auto &fb = anim.emplaceFileBinding<SpatialFieldFileBinding>(
      &scene, volume.data(), field, files);
  fb.addCallbackToAnimation(anim);

  return volume;
}

} // namespace tsd::io
