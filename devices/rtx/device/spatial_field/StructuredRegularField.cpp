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

#include "StructuredRegularField.h"

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

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::box3, ANARI_FLOAT32_BOX3);

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

// StructuredRegularField definitions /////////////////////////////////////////

StructuredRegularField::StructuredRegularField(DeviceGlobalState *d)
    : SpatialField(d), m_data(this)
{}

StructuredRegularField::~StructuredRegularField()
{
  cleanup();
}

void StructuredRegularField::commitParameters()
{
  m_origin = getParam<vec3>("origin", vec3(0.f));
  m_spacing = getParam<vec3>("spacing", vec3(1.f));

  auto dataCentering = getParamString("dataCentering", "node");
  m_cellCentered = (dataCentering == "cell");
  m_filter = getParamString("filter", "linear");
  m_data = getParamObject<Array3D>("data");
  
  // ROI in object space (default: full range)
  m_roi = getParam<box3>("roi",
      box3(vec3(std::numeric_limits<float>::lowest()),
          vec3(std::numeric_limits<float>::max())));
}

void StructuredRegularField::finalize()
{
  cleanup();

  if (!m_data) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on structuredRegular spatial field");
    return;
  }

  ANARIDataType format = m_data->elementType();

  if (!validFieldDataType(format)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid data array type encountered "
        "in structuredRegular spatial field(%s)",
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
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(&m_textureObject, &resDesc, &texDesc, nullptr);

  buildGrid();
  upload();
}

bool StructuredRegularField::isValid() const
{
  return m_data && validFieldDataType(m_data->elementType());
}

box3 StructuredRegularField::bounds() const
{
  if (!isValid())
    return {box3(vec3(0.f), vec3(1.f))};
  auto dims = m_data->size();
  // Node-centered: index space [0, dims-1], cell-centered: index space [0, dims]
  float extentX = m_cellCentered ? float(dims.x) : float(dims.x - 1);
  float extentY = m_cellCentered ? float(dims.y) : float(dims.y - 1);
  float extentZ = m_cellCentered ? float(dims.z) : float(dims.z - 1);
  return box3(m_origin, m_origin + (vec3(extentX, extentY, extentZ) * m_spacing));
}

float StructuredRegularField::stepSize() const
{
  return glm::compMin(m_spacing / 2.f);
}

SpatialFieldGPUData StructuredRegularField::gpuData() const
{
  SpatialFieldGPUData sf;
  auto dims = m_data->size();
  sf.samplerCallableIndex = SbtCallableEntryPoints::SpatialFieldSamplerRegular;
  sf.data.structuredRegular.texObj = m_textureObject;
  sf.data.structuredRegular.origin = m_origin;
  sf.data.structuredRegular.invDims = 1.0f / vec3(dims.x, dims.y, dims.z);
  sf.data.structuredRegular.invSpacing = 1.0f / m_spacing;
  sf.data.structuredRegular.cellCentered = m_cellCentered;
  sf.grid = m_uniformGrid.gpuData();
  
  sf.roi.lower = m_roi.lower;
  sf.roi.upper = m_roi.upper;
  
  return sf;
}

void StructuredRegularField::cleanup()
{
  if (m_textureObject)
    cudaDestroyTextureObject(m_textureObject);
  if (m_cudaArray)
    cudaFreeArray(m_cudaArray);
  m_textureObject = {};
  m_cudaArray = {};
  m_uniformGrid.cleanup();
}

void StructuredRegularField::buildGrid()
{
  auto dims = m_data->size();
  m_uniformGrid.init(ivec3(dims.x, dims.y, dims.z), bounds());

  size_t numVoxels = (dims.x - 1) * size_t(dims.y - 1) * (dims.z - 1);
  m_uniformGrid.buildGrid(gpuData());
}

} // namespace visrtx
