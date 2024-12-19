/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <nanovdb/NanoVDB.h>
#include "gpu/gpu_objects.h"
#include "nanovdb/math/Math.h"
#include "nanovdb/math/SampleFromVoxels.h"

namespace visrtx {

VISRTX_DEVICE const SpatialFieldGPUData &getSpatialFieldData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.fields[idx];
}

VISRTX_DEVICE float sampleSpatialField(
    const SpatialFieldGPUData &sf, const vec3 &location)
{
  float retval = 0.f;

  // TODO: runtime compile errors if these are in the switch()

  switch (sf.type) {
  case SpatialFieldType::STRUCTURED_REGULAR: {
    const auto &srf = sf.data.structuredRegular;
    const auto srfCoords =
        ((location - srf.origin) + 0.5f * srf.spacing) * srf.invSpacing;

    retval = tex3D<float>(srf.texObj, srfCoords.x, srfCoords.y, srfCoords.z);
    break;
  }
  case SpatialFieldType::NANOVDB_REGULAR: {
    const auto &metadata = sf.data.nvdbRegular;
    const auto nvdbLoc = nanovdb::Vec3d(location.x, location.y, location.z);

    switch (metadata.gridType) {
    case nanovdb::GridType::Fp4: {
      auto grid = reinterpret_cast<const nanovdb::Fp4Grid *>(metadata.gridData);
      auto acc = grid->getAccessor();
      auto sampler = nanovdb::math::createSampler<1>(acc);
      auto res = sampler(nanovdb::math::Vec3d(grid->worldToIndexF(nvdbLoc)));
      retval = res;
      break;
    }
    case nanovdb::GridType::Fp8: {
      auto grid = reinterpret_cast<const nanovdb::Fp8Grid *>(metadata.gridData);
      auto acc = grid->getAccessor();
      auto sampler = nanovdb::math::createSampler<1>(acc);
      auto res = sampler(nanovdb::math::Vec3d(grid->worldToIndexF(nvdbLoc)));
      retval = res;
      break;
    }
    case nanovdb::GridType::Fp16: {
      auto grid =
          reinterpret_cast<const nanovdb::Fp16Grid *>(metadata.gridData);
      auto acc = grid->getAccessor();
      auto sampler = nanovdb::math::createSampler<1>(acc);
      auto res = sampler(nanovdb::math::Vec3d(grid->worldToIndexF(nvdbLoc)));
      retval = res;
      break;
    }
    case nanovdb::GridType::FpN: {
      auto grid = reinterpret_cast<const nanovdb::FpNGrid *>(metadata.gridData);
      auto acc = grid->getAccessor();
      auto sampler = nanovdb::math::createSampler<1>(acc);
      auto res = sampler(nanovdb::math::Vec3d(grid->worldToIndexF(nvdbLoc)));
      retval = res;
      break;
    }

    case nanovdb::GridType::Float: {
      auto grid =
          reinterpret_cast<const nanovdb::FloatGrid *>(metadata.gridData);
      auto acc = grid->getAccessor();
      auto sampler = nanovdb::math::createSampler<1>(acc);
      auto res = sampler(grid->worldToIndexF(nvdbLoc));
      retval = res;
      break;
    }
    default:
      break;
    }
  }
  default:
    break;
  }
  return retval;
}

} // namespace visrtx
