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
#include <texture_types.h>
#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "nanovdb/math/Math.h"
#include "nanovdb/math/SampleFromVoxels.h"

namespace visrtx {

VISRTX_DEVICE const SpatialFieldGPUData &getSpatialFieldData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.fields[idx];
}

template<typename T>
class SpatialFieldSampler {
};

template<>
class SpatialFieldSampler<cudaTextureObject_t> {
public:
  VISRTX_DEVICE SpatialFieldSampler(const SpatialFieldGPUData& sf) {
    m_texObj = sf.data.structuredRegular.texObj;
    m_origin = sf.data.structuredRegular.origin;
    m_spacing = sf.data.structuredRegular.spacing;
    m_invSpacing = sf.data.structuredRegular.invSpacing;
  }

  VISRTX_DEVICE float operator()(const vec3 &location) {
    const auto srfCoords =
        ((location - m_origin) + 0.5f * m_spacing) * m_invSpacing;
    return tex3D<float>(m_texObj, srfCoords.x, srfCoords.y, srfCoords.z);
  }

private:
  cudaTextureObject_t m_texObj;
  vec3 m_origin;
  vec3 m_spacing;
  vec3 m_invSpacing;
};

template<typename T>
class SpatialFieldSampler<nanovdb::Grid<nanovdb::NanoTree<T>>> {
  using GridType = typename nanovdb::Grid<nanovdb::NanoTree<T>>;
  using AccessorType = typename GridType::AccessorType;
  using SamplerType = nanovdb::math::SampleFromVoxels<AccessorType, 1>;

public:
  VISRTX_DEVICE SpatialFieldSampler(const SpatialFieldGPUData& sf) :
    m_grid(reinterpret_cast<const GridType *>(sf.data.nvdbRegular.gridData)),
    m_accessor(m_grid->getAccessor()),
    m_sampler(nanovdb::math::createSampler<1>(m_accessor))
  {
  }

  VISRTX_DEVICE float operator()(const vec3 &location) {
    const auto nvdbLoc = nanovdb::Vec3d(location.x, location.y, location.z);
    return m_sampler(nanovdb::math::Vec3d(m_grid->worldToIndexF(nvdbLoc)));
  }

private:
  const GridType* m_grid;
  const AccessorType m_accessor;
  SamplerType m_sampler;
};

template<typename T>
using NvdbSpatialFieldSampler = SpatialFieldSampler<nanovdb::Grid<nanovdb::NanoTree<T>>>;


// FIXME: To remove once done with replace sampleSpatialField with SpatialFieldSamplers
#if 1
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
#if 0
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
#endif
   case nanovdb::GridType::Float: {
      NvdbSpatialFieldSampler<float> sampler(sf);
      retval = sampler(location);
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
#endif

} // namespace visrtx
