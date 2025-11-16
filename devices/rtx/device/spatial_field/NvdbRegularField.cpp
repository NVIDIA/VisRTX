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

#include "NvdbRegularField.h"

// anari
#include <anari/frontend/anari_enums.h>

// nanovdb
#include <nanovdb/NanoVDB.h>

// glm
#include <glm/ext/vector_float3.hpp>
#include <glm/gtx/component_wise.hpp>

namespace visrtx {

// NvdbRegularField definitions /////////////////////////////////////////

NvdbRegularField::NvdbRegularField(DeviceGlobalState *d)
    : SpatialField(d), m_data(this)
{}

NvdbRegularField::~NvdbRegularField()
{
  cleanup();
}

void NvdbRegularField::commitParameters()
{
  m_filter = getParamString("filter", "linear");
  m_data = getParamObject<Array1D>("data");
}

void NvdbRegularField::finalize()
{
  cleanup();

  if (!m_data) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on NanoVDB regular spatial field");
    return;
  }

  ANARIDataType format = m_data->elementType();
  if (format != ANARI_UINT8) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid data array type encountered "
        "in NanoVDB spatial field(%s)",
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

  buildGrid();
  upload();
}

bool NvdbRegularField::isValid() const
{
  return m_data && m_data->elementType() == ANARI_UINT8;
}

box3 NvdbRegularField::bounds() const
{
  return m_bounds;
}

float NvdbRegularField::stepSize() const
{
  return glm::compMin(m_voxelSize) / 2.0f;
}

SpatialFieldGPUData NvdbRegularField::gpuData() const
{
  SpatialFieldGPUData sf;
  sf.type = SpatialFieldType::NANOVDB_REGULAR;
  sf.data.nvdbRegular.voxelSize = m_voxelSize;
  sf.data.nvdbRegular.origin = m_bounds.lower;
  sf.data.nvdbRegular.gridData = m_deviceBuffer.ptr();
  sf.data.nvdbRegular.gridType = m_gridMetadata->gridType();

  sf.grid = m_uniformGrid.gpuData();

  return sf;
}

void NvdbRegularField::cleanup()
{
  m_uniformGrid.cleanup();
}

void NvdbRegularField::buildGrid()
{
  auto gridSize = m_gridMetadata->indexBBox().dim();
  m_uniformGrid.init(ivec3(gridSize[0], gridSize[1], gridSize[2]), m_bounds);

  size_t numVoxels =
      (gridSize[0] - 1) * size_t(gridSize[1] - 1) * (gridSize[2] - 1);
  m_uniformGrid.buildGrid(gpuData());
}

} // namespace visrtx
