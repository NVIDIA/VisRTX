/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"

// cuda
#include <texture_types.h>

// nanovdb
#include <nanovdb/NanoVDB.h>
#include "nanovdb/math/Math.h"
#include "nanovdb/math/SampleFromVoxels.h"

namespace visrtx {

// Template sampler classes for UniformGrid preprocessing (CPU-side kernel launches)
// These are only used during space-skipping grid construction, not for runtime sampling

template <typename T>
class SpatialFieldSampler
{
};

template <>
class SpatialFieldSampler<cudaTextureObject_t>
{
 public:
  VISRTX_DEVICE SpatialFieldSampler(const SpatialFieldGPUData &sf)
  {
    m_texObj = sf.data.structuredRegular.texObj;
    m_origin = sf.data.structuredRegular.origin;
    m_invDims = sf.data.structuredRegular.invDims;
    m_invSpacing = sf.data.structuredRegular.invSpacing;
  }

  VISRTX_DEVICE float operator()(const vec3 &location)
  {
    const auto texelCoords = (location - m_origin) * m_invSpacing + 0.5f;
    const auto coords = texelCoords * m_invDims;
    return tex3D<float>(m_texObj, coords.x, coords.y, coords.z);
  }

 private:
  cudaTextureObject_t m_texObj;
  vec3 m_origin;
  vec3 m_invDims;
  vec3 m_invSpacing;
};

template <typename T>
class SpatialFieldSampler<nanovdb::Grid<nanovdb::NanoTree<T>>>
{
  using GridType = typename nanovdb::Grid<nanovdb::NanoTree<T>>;
  using AccessorType = typename GridType::AccessorType;
  using SamplerType = nanovdb::math::SampleFromVoxels<AccessorType, 1>;

 public:
  VISRTX_DEVICE SpatialFieldSampler(const SpatialFieldGPUData &sf)
      : m_grid(
            reinterpret_cast<const GridType *>(sf.data.nvdbRegular.gridData)),
        m_accessor(m_grid->getAccessor()),
        m_sampler(nanovdb::math::createSampler<1>(m_accessor)),
        m_scale(
            (nanovdb::Vec3f(m_grid->indexBBox().dim()) - nanovdb::Vec3f(1.0f))
            / nanovdb::Vec3f(m_grid->indexBBox().dim()))
  {}

  VISRTX_DEVICE float operator()(const vec3 &location)
  {
    return m_sampler(m_grid->worldToIndexF(
                         nanovdb::Vec3f(location.x, location.y, location.z))
        * m_scale);
  }

 private:
  const GridType *m_grid;
  const AccessorType m_accessor;
  SamplerType m_sampler;
  nanovdb::Vec3f m_scale;
};

template <typename T>
using NvdbSpatialFieldSampler =
    SpatialFieldSampler<nanovdb::Grid<nanovdb::NanoTree<T>>>;

} // namespace visrtx
