// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_STAGE
#define TSD_USE_STAGE 1
#endif

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
#if TSD_USE_STAGE
// stage
#include <stage/stage.h>
#endif

namespace tsd {

#if TSD_USE_STAGE

static void populateSTAGELayer(Context &ctx, LayerNodeRef tsdLayerRef)
{
  // TODO
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void import_STAGE(Context &ctx, const char *filename, LayerNodeRef location)
{
  stage::Config config;
  stage::Scene scene(filename, config);
  if (!scene.isValid()) {
    logError("[import_STAGE] could not load scene from file %s", filename);
    return;
  }

  populateSTAGELayer(ctx, location ? location : ctx.defaultLayer()->root());
}
#else
void import_STAGE(
    Context &ctx, const char *filename, LayerNodeRef location, bool flatten)
{
  logError("[import_STAGE] STAGE not enabled in TSD build.");
}
#endif

} // namespace tsd
