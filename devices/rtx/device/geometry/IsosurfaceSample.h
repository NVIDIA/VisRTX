/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Inline field sampling for the isosurface intersector. Built-in fields are
// sampled through their concrete inline samplers (no OptiX callable). The
// Init/Sample direct callables exist only for external custom fields, which
// are out of scope here.
//
// VISRTX_ISOSURFACE_FIELD_VARIANTS is the single source of truth for the
// supported built-in fields; the intersector expands it once to dispatch on
// the concrete sampler (no per-sample switch).

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/sbt.h"
#include "gpu/shadingState.h"
#include "spatial_field/NvdbRectilinearSamplerInline.h"
#include "spatial_field/NvdbRegularSamplerInline.h"
#include "spatial_field/StructuredRectilinearSamplerInline.h"
#include "spatial_field/StructuredRegularSamplerInline.h"

namespace visrtx {

// (sbt entry point, VolumeSamplingState union member, init fn). Sampling goes
// through the shared sampleValue/sampleNormal overloads, resolved by ADL on the
// concrete state type, so no per-variant sample fn is needed here.
#define VISRTX_ISOSURFACE_FIELD_VARIANTS                                       \
  X(SpatialFieldSamplerRegular, structuredRegular, initStructuredRegularSampler)\
  X(SpatialFieldSamplerRectilinear, structuredRectilinear,                     \
      initStructuredRectilinearSampler)                                        \
  X(SpatialFieldSamplerNvdbFp4, nvdbFp4, initNvdbSampler)                      \
  X(SpatialFieldSamplerNvdbFp8, nvdbFp8, initNvdbSampler)                      \
  X(SpatialFieldSamplerNvdbFp16, nvdbFp16, initNvdbSampler)                    \
  X(SpatialFieldSamplerNvdbFpN, nvdbFpN, initNvdbSampler)                      \
  X(SpatialFieldSamplerNvdbFloat, nvdbFloat, initNvdbSampler)                  \
  X(SpatialFieldSamplerNvdbRectilinearFp4, nvdbRectilinearFp4,                 \
      initNvdbRectilinearSampler)                                             \
  X(SpatialFieldSamplerNvdbRectilinearFp8, nvdbRectilinearFp8,                 \
      initNvdbRectilinearSampler)                                             \
  X(SpatialFieldSamplerNvdbRectilinearFp16, nvdbRectilinearFp16,               \
      initNvdbRectilinearSampler)                                             \
  X(SpatialFieldSamplerNvdbRectilinearFpN, nvdbRectilinearFpN,                 \
      initNvdbRectilinearSampler)                                             \
  X(SpatialFieldSamplerNvdbRectilinearFloat, nvdbRectilinearFloat,             \
      initNvdbRectilinearSampler)

} // namespace visrtx
