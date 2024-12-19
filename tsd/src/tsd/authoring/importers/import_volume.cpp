// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <cstdio>

namespace tsd {

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

  auto tx = ctx.insertChildTransformNode(ctx.tree.root());

  auto [inst, volume] = ctx.insertNewChildObjectNode<tsd::Volume>(
      tx, tokens::volume::transferFunction1D);
  volume->setName(fileOf(filepath).c_str());
  volume->setParameterObject("value", *field);
  volume->setParameterObject("color", *colorArray);
  volume->setParameterObject("opacity", *opacityArray);
  volume->setParameter("densityScale", 0.1f);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  return volume;
}

} // namespace tsd
