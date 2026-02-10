/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "SpatialFieldRegistry.h"

namespace visrtx {

// This function is called by external applications (like VolumetricPlanets)
// to register their custom spatial fields with the VisRTX device.
//
// Usage from VolumetricPlanets:
//   #include "spatial_field/SpatialFieldRegistry.h"
//   visrtx::registerCustomField("magnetic", [](DeviceGlobalState* d) {
//     return new MagneticField(d);
//   });

void registerCustomField(
    const std::string &typeName, SpatialFieldFactory factory)
{
  SpatialFieldRegistry::instance().registerType(typeName, factory);
}

} // namespace visrtx
