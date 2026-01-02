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
    m_cellCentered = sf.data.structuredRegular.cellCentered;
  }

  VISRTX_DEVICE float operator()(const vec3 &location)
  {
    const float offset = m_cellCentered ? 0.0f : 0.5f;
    const auto texelCoords = (location - m_origin) * m_invSpacing + offset;
    const auto coords = texelCoords * m_invDims;
    return tex3D<float>(m_texObj, coords.x, coords.y, coords.z);
  }

 private:
  cudaTextureObject_t m_texObj;
  vec3 m_origin;
  vec3 m_invDims;
  vec3 m_invSpacing;
  bool m_cellCentered;
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
         m_sampler(nanovdb::math::createSampler<1>(m_accessor))
    {
      const bool cellCentered = sf.data.nvdbRegular.cellCentered;
      const nanovdb::Vec3f dims = nanovdb::Vec3f(m_grid->indexBBox().dim());
      m_offset = cellCentered ? nanovdb::Vec3f(0.0f) : nanovdb::Vec3f(0.5f);
      const float scaleAdjust = cellCentered ? 1.5f : 1.0f;
      m_scale = (dims - nanovdb::Vec3f(scaleAdjust)) / dims;
    }

  VISRTX_DEVICE float operator()(const vec3 &location)
  {
    auto indexPos = m_grid->worldToIndexF(
        nanovdb::Vec3f(location.x, location.y, location.z));
    indexPos += m_offset;
    return m_sampler(indexPos * m_scale);
  }

 private:
  const GridType *m_grid;
  const AccessorType m_accessor;
  SamplerType m_sampler;
  nanovdb::Vec3f m_scale;
  nanovdb::Vec3f m_offset;
};

template <typename T>
using NvdbSpatialFieldSampler =
    SpatialFieldSampler<nanovdb::Grid<nanovdb::NanoTree<T>>>;

} // namespace visrtx
