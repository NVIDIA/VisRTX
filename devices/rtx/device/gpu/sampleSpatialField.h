/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"

// cuda
#include <texture_types.h>

// nanovdb
#include <nanovdb/NanoVDB.h>
#include "nanovdb/math/Math.h"
#include "nanovdb/math/SampleFromVoxels.h"

namespace visrtx {

VISRTX_DEVICE const SpatialFieldGPUData &getSpatialFieldData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.fields[idx];
}

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
