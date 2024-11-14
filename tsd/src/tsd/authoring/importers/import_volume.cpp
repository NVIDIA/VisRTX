// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
// std
#include <cstdio>

namespace tsd {

IndexedVectorRef<Volume> import_volume(Context &ctx,
    const char *filepath,
    IndexedVectorRef<Array> colorArray,
    IndexedVectorRef<Array> opacityArray)
{
  IndexedVectorRef<SpatialField> field;

  auto ext = extensionOf(filepath);
  if (ext == ".raw")
    field = import_RAW(ctx, filepath);
  else if (ext == ".flash")
    field = import_FLASH(ctx, filepath);

  auto volume = ctx.createObject<Volume>(tokens::volume::transferFunction1D);
  volume->setName(fileOf(filepath).c_str());
  volume->setParameterObject("value", *field);
  volume->setParameterObject("color", *colorArray);
  volume->setParameterObject("opacity", *opacityArray);
  volume->setParameter("densityScale", 0.1f);

  ctx.tree.insert_last_child(
      ctx.tree.root(), utility::Any(ANARI_VOLUME, volume.index()));

  return volume;
}

} // namespace tsd
