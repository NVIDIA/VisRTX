// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <anari/anari_cpp/ext/linalg.h>
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <cstdio>

namespace tsd::io {

using namespace tsd::core;

VolumeRef import_volume(
    Scene &scene, const char *filepath, LayerNodeRef location)
{
  SpatialFieldRef field;

  auto file = fileOf(filepath);
  auto ext = extensionOf(filepath);
  if (ext == ".raw")
    field = import_RAW(scene, filepath);
  else if (ext == ".flash" || ext == ".hdf5")
    field = import_FLASH(scene, filepath);
  else if (ext == ".nvdb")
    field = import_NVDB(scene, filepath);
  else if (ext == ".mhd")
    field = import_MHD(scene, filepath);
  else if (ext == ".vtu")
    field = import_VTU(scene, filepath);
  else if (ext == ".vti")
    field = import_VTI(scene, filepath);
  else if (ext == ".silo" || ext == ".sil")
    field = import_SILO(scene, filepath);
  else {
    logError("[import_volume] no loader for file type '%s'", ext.c_str());
    return {};
  }

  if (!field) {
    logError(
        "[import_volume] unable to load field from file '%s'", file.c_str());
    return {};
  }

  float2 valueRange{0.f, 1.f};
  if (field)
    valueRange = field->computeValueRange();

  auto tx = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
      tx, tokens::volume::transferFunction1D);
  volume->setName(fileOf(filepath).c_str());
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

  return volume;
}

} // namespace tsd::io
