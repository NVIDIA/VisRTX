// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <cstdio>

namespace tsd::io {

using namespace tsd::core;

VolumeRef import_volume(Context &ctx,
    const char *filepath,
    ArrayRef colorArray,
    ArrayRef opacityArray)
{
  SpatialFieldRef field;

  auto file = fileOf(filepath);
  auto ext = extensionOf(filepath);
  if (ext == ".raw")
    field = import_RAW(ctx, filepath);
  else if (ext == ".flash")
    field = import_FLASH(ctx, filepath);
  else if (ext == ".nvdb")
    field = import_NVDB(ctx, filepath);
  else if (ext == ".mhd")
    field = import_MHD(ctx, filepath);
  else if (ext == ".vtu")
    field = import_VTU(ctx, filepath);
  else if (ext == ".vti")
    field = import_VTI(ctx, filepath);
  else {
    logError("[import_volume] no loader for file type '%s'", ext.c_str());
    return {};
  }

  if (!field) {
    logError(
        "[import_volume] unable to load field from file '%s'", file.c_str());
    return {};
  }

  if (!colorArray) {
    colorArray = ctx.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(makeDefaultColorMap(colorArray->size()).data());
  }

  float2 valueRange{0.f, 1.f};
  if (field)
    valueRange = field->computeValueRange();

  auto tx = ctx.insertChildTransformNode(ctx.defaultLayer()->root());

  auto [inst, volume] = ctx.insertNewChildObjectNode<Volume>(
      tx, tokens::volume::transferFunction1D);
  volume->setName(fileOf(filepath).c_str());
  volume->setParameterObject("value", *field);
  volume->setParameterObject("color", *colorArray);
  if (opacityArray)
    volume->setParameterObject("opacity", *opacityArray);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  return volume;
}

} // namespace tsd
