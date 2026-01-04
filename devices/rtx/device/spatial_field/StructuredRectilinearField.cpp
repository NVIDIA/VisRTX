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

#include "StructuredRectilinearField.h"

#include "RectilinearLUT.h"
#include "gpu/gpu_decl.h"
#include "gpu/shadingState.h"
#include "utility/AnariTypeHelpers.h"

// anari
#include <anari/anari_cpp/Traits.h>
#include <anari/frontend/anari_enums.h>
// std
#include <algorithm>
#include <limits>
#include <vector>
// glm
#include <glm/gtx/component_wise.hpp>

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

static bool validFieldDataType(anari::DataType format)
{
  switch (format) {
  case ANARI_FIXED8:
  case ANARI_UFIXED8:
  case ANARI_FIXED16:
  case ANARI_UFIXED16:
  case ANARI_FLOAT32:
  case ANARI_FLOAT64:
    return true;
  default:
    break;
  }
  return false;
}

static cudaChannelFormatKind getCudaChannelFormatKind(anari::DataType format)
{
  switch (format) {
  case ANARI_UFIXED8:
  case ANARI_UFIXED16:
    return cudaChannelFormatKindUnsigned;
  case ANARI_FIXED16:
  case ANARI_FIXED8:
    return cudaChannelFormatKindSigned;
  case ANARI_FLOAT32:
  case ANARI_FLOAT64:
  default:
    return cudaChannelFormatKindFloat;
    break;
  }
  return cudaChannelFormatKindFloat;
}

// StructuredRectilinearField definitions ////////////////////////////////////

StructuredRectilinearField::StructuredRectilinearField(DeviceGlobalState *d)
    : SpatialField(d),
      m_data(this),
      m_axisArrayX(this),
      m_axisArrayY(this),
      m_axisArrayZ(this)
{}

StructuredRectilinearField::~StructuredRectilinearField()
{
  cleanup();
}

void StructuredRectilinearField::commitParameters()
{
  auto dataCentering = getParamString("dataCentering", "node");
  m_cellCentered = (dataCentering == "cell");
  m_filter = getParamString("filter", "linear");
  m_data = getParamObject<Array3D>("data");

  // ROI in object space (default: full range)
  m_roi = getParam<box3>("roi",
      box3(vec3(std::numeric_limits<float>::lowest()),
          vec3(std::numeric_limits<float>::max())));

  // Rectilinear axis arrays (all 3 required for this type)
  m_axisArrayX = getParamObject<Array1D>("coordsX");
  m_axisArrayY = getParamObject<Array1D>("coordsY");
  m_axisArrayZ = getParamObject<Array1D>("coordsZ");
}

void StructuredRectilinearField::finalize()
{
  cleanup();

  if (!m_data) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on structuredRectilinear spatial field");
    return;
  }

  if (!m_axisArrayX || !m_axisArrayY || !m_axisArrayZ) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'coordsX/coordsY/coordsZ' on structuredRectilinear spatial field");
    return;
  }

  ANARIDataType format = m_data->elementType();

  if (!validFieldDataType(format)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid data array type encountered "
        "in structuredRectilinear spatial field(%s)",
        anari::toString(format));
    return;
  }

  const auto dims = m_data->size();

  auto desc = cudaCreateChannelDesc(
      anari::sizeOf(format) * 8, 0, 0, 0, getCudaChannelFormatKind(format));
  cudaMalloc3DArray(
      &m_cudaArray, &desc, make_cudaExtent(dims.x, dims.y, dims.z));

  cudaMemcpy3DParms copyParams;
  std::memset(&copyParams, 0, sizeof(copyParams));
  copyParams.srcPtr = make_cudaPitchedPtr(const_cast<void *>(m_data->dataGPU()),
      dims.x * anari::sizeOf(format),
      dims.x,
      dims.y);
  copyParams.dstArray = m_cudaArray;
  copyParams.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  copyParams.kind = cudaMemcpyDeviceToDevice;

  cudaMemcpy3D(&copyParams);

  m_data->evictGPU();

  cudaResourceDesc resDesc;
  std::memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_cudaArray;

  cudaTextureDesc texDesc;
  std::memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.filterMode =
      m_filter == "nearest" ? cudaFilterModePoint : cudaFilterModeLinear;
  texDesc.readMode =
      isFloat(format) ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&m_textureObject, &resDesc, &texDesc, nullptr);

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

    // Set bounds from axis coordinate endpoints
    m_bounds.lower[axis] = axisData[0];
    m_bounds.upper[axis] = axisData[numCoords - 1];
  }

  buildGrid();
  upload();
}

bool StructuredRectilinearField::isValid() const
{
  return m_data && validFieldDataType(m_data->elementType()) && m_axisArrayX
      && m_axisArrayY && m_axisArrayZ;
}

box3 StructuredRectilinearField::bounds() const
{
  if (!isValid())
    return {box3(vec3(0.f), vec3(1.f))};

  // Use rectilinear bounds
  return m_bounds;
}

float StructuredRectilinearField::stepSize() const
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

SpatialFieldGPUData StructuredRectilinearField::gpuData() const
{
  SpatialFieldGPUData sf;
  auto dims = m_data->size();
  sf.samplerCallableIndex =
      SbtCallableEntryPoints::SpatialFieldSamplerRectilinear;
  sf.data.structuredRectilinear.texObj = m_textureObject;
  sf.data.structuredRectilinear.dims = vec3(dims.x, dims.y, dims.z);
  sf.data.structuredRectilinear.cellCentered = m_cellCentered;
  sf.data.structuredRectilinear.axisLUT[0] = m_axisLutTextures[0];
  sf.data.structuredRectilinear.axisLUT[1] = m_axisLutTextures[1];
  sf.data.structuredRectilinear.axisLUT[2] = m_axisLutTextures[2];
  sf.data.structuredRectilinear.axisBoundsMin = m_bounds.lower;
  sf.data.structuredRectilinear.axisBoundsMax = m_bounds.upper;

  sf.grid = m_uniformGrid.gpuData();

  sf.roi.lower = m_roi.lower;
  sf.roi.upper = m_roi.upper;

  return sf;
}

void StructuredRectilinearField::cleanup()
{
  if (m_textureObject)
    cudaDestroyTextureObject(m_textureObject);
  if (m_cudaArray)
    cudaFreeArray(m_cudaArray);

  for (int i = 0; i < 3; ++i) {
    if (m_axisLutTextures[i])
      cudaDestroyTextureObject(m_axisLutTextures[i]);
    if (m_axisLutArrays[i])
      cudaFreeArray(m_axisLutArrays[i]);
    m_axisLutTextures[i] = {};
    m_axisLutArrays[i] = {};
  }

  m_textureObject = {};
  m_cudaArray = {};
  m_bounds = box3(vec3(std::numeric_limits<float>::max()),
      vec3(std::numeric_limits<float>::lowest()));
  m_uniformGrid.cleanup();
}

void StructuredRectilinearField::buildGrid()
{
  auto dims = m_data->size();
  m_uniformGrid.init(ivec3(dims.x, dims.y, dims.z), bounds());

  size_t numVoxels = (dims.x - 1) * size_t(dims.y - 1) * (dims.z - 1);
  m_uniformGrid.buildGrid(gpuData());
}

} // namespace visrtx
