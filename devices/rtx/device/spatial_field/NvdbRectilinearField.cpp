/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvdbRectilinearField.h"
#include "RectilinearLUT.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/shadingState.h"
#include "optix_visrtx.h"

// anari
#include <anari/anari_cpp/Traits.h>
#include <anari/frontend/anari_enums.h>

// nanovdb
#include <nanovdb/NanoVDB.h>

// glm
#include <glm/ext/vector_float3.hpp>
#include <glm/gtx/component_wise.hpp>

namespace visrtx {

// NvdbRectilinearField definitions ////////////////////////////////////

NvdbRectilinearField::NvdbRectilinearField(DeviceGlobalState *d)
    : SpatialField(d),
      m_data(this),
      m_axisArrayX(this),
      m_axisArrayY(this),
      m_axisArrayZ(this)
{}

NvdbRectilinearField::~NvdbRectilinearField()
{
  cleanup();
}

void NvdbRectilinearField::commitParameters()
{
  m_filter = getParamString("filter", "linear");
  m_data = getParamObject<Array1D>("data");

  auto dataCentering = getParamString("dataCentering", "cell");
  m_cellCentered = (dataCentering == "cell");

  // ROI in object space (default: full range)
  m_roi = getParam<box3>("roi",
      box3(vec3(std::numeric_limits<float>::lowest()),
          vec3(std::numeric_limits<float>::max())));

  // Rectilinear axis arrays (all 3 required for this type)
  m_axisArrayX = getParamObject<Array1D>("coordsX");
  m_axisArrayY = getParamObject<Array1D>("coordsY");
  m_axisArrayZ = getParamObject<Array1D>("coordsZ");
}

void NvdbRectilinearField::finalize()
{
  cleanup();

  if (!m_data) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on nanovdbRectilinear spatial field");
    return;
  }

  if (!m_axisArrayX || !m_axisArrayY || !m_axisArrayZ) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'coordsX/y/z' on nanovdbRectilinear spatial field");
    return;
  }

  ANARIDataType format = m_data->elementType();
  if (format != ANARI_UINT8) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid data array type encountered "
        "in nanovdbRectilinear spatial field(%s)",
        anari::toString(format));
    return;
  }

  const void *dataPtr = m_data->data(AddressSpace::HOST);
  const auto *gridData = static_cast<const nanovdb::GridData *>(dataPtr);

  if (!gridData->isValid()) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "invalid NanoVDB grid data in spatial field");
    return;
  }

  m_gridMetadata = nanovdb::GridMetaData(gridData);

  if (m_gridMetadata->gridCount() != 1) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "VisRTX NanoVDB support's a single grid per file");
    return;
  }

  m_deviceBuffer.upload(
      static_cast<const std::byte *>(dataPtr), m_data->size());

  auto boundsMin = m_gridMetadata->worldBBox().min();
  auto boundsMax = m_gridMetadata->worldBBox().max();
  m_bounds = box3(glm::vec3(boundsMin[0], boundsMin[1], boundsMin[2]),
      glm::vec3(boundsMax[0], boundsMax[1], boundsMax[2]));
  auto voxelSize = m_gridMetadata->voxelSize();
  m_voxelSize = glm::vec3(voxelSize[0], voxelSize[1], voxelSize[2]);

  // Build rectilinear LUTs for all three required axes
  std::array<helium::ChangeObserverPtr<Array1D> *, 3> axisArrays = {
      &m_axisArrayX, &m_axisArrayY, &m_axisArrayZ};

  for (int axis = 0; axis < 3; ++axis) {
    if ((*axisArrays[axis])->elementType() != ANARI_FLOAT32) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "axis array %d has invalid type (expected ANARI_FLOAT32)",
          axis);
      cleanup();
      return;
    }

    const auto *axisData = static_cast<const float *>(
        (*axisArrays[axis])->data(AddressSpace::HOST));
    size_t numCoords = (*axisArrays[axis])->size();

    if (!RectilinearLUT::buildAxisLUT(axisData,
            numCoords,
            m_axisLutTextures[axis],
            m_axisLutArrays[axis])) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "failed to build rectilinear LUT for axis %d (insufficient coordinates or non-monotonic ordering).",
          axis);
      cleanup();
      return;
    }
  }

  // Compute invAvgVoxelSize from axis extents and grid dims
  {
    const auto gridDims = m_gridMetadata->indexBBox().dim();
    std::array<helium::ChangeObserverPtr<Array1D> *, 3> axes = {
        &m_axisArrayX, &m_axisArrayY, &m_axisArrayZ};
    for (int axis = 0; axis < 3; ++axis) {
      const auto *axisData = static_cast<const float *>(
          (*axes[axis])->data(AddressSpace::HOST));
      size_t n = (*axes[axis])->size();
      float extent = axisData[n - 1] - axisData[0];
      m_invAvgVoxelSize[axis] = m_cellCentered
          ? static_cast<float>(gridDims[axis]) / extent
          : static_cast<float>(gridDims[axis] - 1) / extent;
    }
  }

  buildGrid();
  upload();
}

bool NvdbRectilinearField::isValid() const
{
  return m_data && m_data->elementType() == ANARI_UINT8 && m_axisArrayX
      && m_axisArrayY && m_axisArrayZ;
}

box3 NvdbRectilinearField::bounds() const
{
  if (!isValid())
    return {box3(vec3(0.f), vec3(1.f))};

  // Use rectilinear bounds
  return m_bounds;
}

float NvdbRectilinearField::stepSize() const
{
  if (!isValid())
    return 1.0f;

  // Use minimum spacing between axis coordinates
  vec3 spacing(std::numeric_limits<float>::max());

  std::array<const helium::ChangeObserverPtr<Array1D> *, 3> axisArrays = {
      &m_axisArrayX, &m_axisArrayY, &m_axisArrayZ};

  for (int axis = 0; axis < 3; ++axis) {
    const auto *axisData = static_cast<const float *>(
        (*axisArrays[axis])->data(AddressSpace::HOST));
    size_t numCoords = (*axisArrays[axis])->size();

    if (numCoords > 1) {
      float minDelta = std::numeric_limits<float>::max();
      for (size_t i = 1; i < numCoords; ++i) {
        minDelta = std::min(minDelta, axisData[i] - axisData[i - 1]);
      }
      spacing[axis] = minDelta;
    }
  }

  return glm::compMin(spacing) / 2.f;
}

SpatialFieldGPUData NvdbRectilinearField::gpuData() const
{
  SpatialFieldGPUData sf;

  // Map grid type to rectilinear callable index
  auto gridType = m_gridMetadata->gridType();
  switch (gridType) {
  case nanovdb::GridType::Fp4:
    sf.samplerCallableIndex =
        SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp4;
    break;
  case nanovdb::GridType::Fp8:
    sf.samplerCallableIndex =
        SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp8;
    break;
  case nanovdb::GridType::Fp16:
    sf.samplerCallableIndex =
        SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFp16;
    break;
  case nanovdb::GridType::FpN:
    sf.samplerCallableIndex =
        SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFpN;
    break;
  case nanovdb::GridType::Float:
    sf.samplerCallableIndex =
        SbtCallableEntryPoints::SpatialFieldSamplerNvdbRectilinearFloat;
    break;
  default:
    sf.samplerCallableIndex = SbtCallableEntryPoints::Invalid;
    break;
  }

  sf.data.nvdbRectilinear.gridData = m_deviceBuffer.ptr();
  sf.data.nvdbRectilinear.gridType = gridType;
  sf.data.nvdbRectilinear.cellCentered = m_cellCentered;
  sf.data.nvdbRectilinear.filter = (m_filter == "nearest")
      ? SpatialFieldFilter::Nearest
      : SpatialFieldFilter::Linear;

  sf.data.nvdbRectilinear.axisLUT[0] = m_axisLutTextures[0];
  sf.data.nvdbRectilinear.axisLUT[1] = m_axisLutTextures[1];
  sf.data.nvdbRectilinear.axisLUT[2] = m_axisLutTextures[2];
  sf.data.nvdbRectilinear.invAvgVoxelSize = m_invAvgVoxelSize;

  sf.grid = m_uniformGrid.gpuData();

  sf.roi.lower = m_roi.lower;
  sf.roi.upper = m_roi.upper;

  return sf;
}

void NvdbRectilinearField::cleanup()
{
  for (int i = 0; i < 3; ++i) {
    if (m_axisLutTextures[i])
      cudaDestroyTextureObject(m_axisLutTextures[i]);
    if (m_axisLutArrays[i])
      cudaFreeArray(m_axisLutArrays[i]);
    m_axisLutTextures[i] = {};
    m_axisLutArrays[i] = {};
  }
  m_bounds = box3(vec3(std::numeric_limits<float>::max()),
      vec3(std::numeric_limits<float>::lowest()));

  m_uniformGrid.cleanup();
}

void NvdbRectilinearField::buildGrid()
{
  auto gridSize = m_gridMetadata->indexBBox().dim();
  m_uniformGrid.init(ivec3(gridSize[0], gridSize[1], gridSize[2]), m_bounds);
  m_uniformGrid.buildGrid(gpuData());
}

} // namespace visrtx
