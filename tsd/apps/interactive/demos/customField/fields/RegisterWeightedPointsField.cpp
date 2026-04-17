// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "WeightedPointsField.h"
#include "spatial_field/SpatialFieldRegistry.h"

namespace {

struct WeightedPointsFieldRegistrar
{
  WeightedPointsFieldRegistrar()
  {
    visrtx::registerCustomField("weightedPoints",
        [](visrtx::DeviceGlobalState *d) -> visrtx::SpatialField * {
          return new visrtx_custom::WeightedPointsField(d);
        });
  }
};

static WeightedPointsFieldRegistrar g_registrar;

} // namespace
