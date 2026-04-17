// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "WeightedPointsField.h"
#include "WeightedPointsFieldData.h"
#include "gpu/gpu_decl.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

namespace visrtx_custom {

WeightedPointsField::WeightedPointsField(visrtx::DeviceGlobalState *d)
    : CustomField(d)
    , m_valuesArray(this)
    , m_indicesArray(this)
{
}

WeightedPointsField::~WeightedPointsField()
{
  freeDeviceArrays();
}

void WeightedPointsField::freeDeviceArrays()
{
  if (m_dValues) { cudaFree(m_dValues); m_dValues = nullptr; }
  if (m_dIndices) { cudaFree(m_dIndices); m_dIndices = nullptr; }
  m_dValuesSize = 0;
  m_dIndicesSize = 0;
}

void WeightedPointsField::commitParameters()
{
  m_sigma = getParam<float>("sigma", 0.f);
  m_cutoff = getParam<float>("cutoff", 0.f);
  m_domainMin = getParam<visrtx::vec3>(
      "domainMin", visrtx::vec3(0.f, 0.f, 0.f));
  m_domainMax = getParam<visrtx::vec3>(
      "domainMax", visrtx::vec3(1.f, 1.f, 1.f));

  m_valuesArray = getParamObject<visrtx::Array1D>("values");
  m_indicesArray = getParamObject<visrtx::Array1D>("indices");
}

void WeightedPointsField::finalize()
{
  freeDeviceArrays();

  if (m_valuesArray) {
    size_t numElements = m_valuesArray->totalSize();
    m_numNodes = (int32_t)(numElements / 4);
    m_dValuesSize = numElements * sizeof(float);
    const float *hostPtr = m_valuesArray->beginAs<float>();

    auto err1 = cudaMalloc(&m_dValues, m_dValuesSize);
    auto err2 = cudaMemcpy(
        m_dValues, hostPtr, m_dValuesSize, cudaMemcpyHostToDevice);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
      m_dValues = nullptr;
      m_numNodes = 0;
    }
  }

  if (m_indicesArray) {
    size_t numElements = m_indicesArray->totalSize();
    m_dIndicesSize = numElements * sizeof(int32_t);
    const int32_t *hostPtr = m_indicesArray->beginAs<int32_t>();

    auto err1 = cudaMalloc(&m_dIndices, m_dIndicesSize);
    auto err2 = cudaMemcpy(
        m_dIndices, hostPtr, m_dIndicesSize, cudaMemcpyHostToDevice);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
      m_dIndices = nullptr;
    }
  }

  float sigma = m_sigma;
  if (sigma <= 0.f) {
    visrtx::vec3 diag = m_domainMax - m_domainMin;
    float vol = diag.x * diag.y * diag.z;
    int N = std::max(m_numNodes, 1);
    sigma = cbrtf(vol / (float)N);
  }

  float cutoff = m_cutoff;
  if (cutoff <= 0.f)
    cutoff = 5.f * sigma;

  m_deviceData.values = m_dValues;
  m_deviceData.indices = m_dIndices;
  m_deviceData.numNodes = m_numNodes;
  m_deviceData.sigma = sigma;
  m_deviceData.inv2SigmaSq = 1.f / (2.f * sigma * sigma);
  m_deviceData.cutoff = cutoff;
  m_deviceData.domainMin = m_domainMin;
  m_deviceData.domainMax = m_domainMax;

  m_uniformGrid.init(visrtx::ivec3(16, 16, 16), bounds());
  CustomField::finalize();
}

bool WeightedPointsField::isValid() const
{
  return m_valuesArray && m_indicesArray && m_dValues && m_dIndices
      && m_numNodes > 0
      && m_domainMin.x < m_domainMax.x
      && m_domainMin.y < m_domainMax.y
      && m_domainMin.z < m_domainMax.z;
}

visrtx::box3 WeightedPointsField::bounds() const
{
  return visrtx::box3(m_domainMin, m_domainMax);
}

float WeightedPointsField::stepSize() const
{
  visrtx::vec3 extent = m_domainMax - m_domainMin;
  return std::min({extent.x, extent.y, extent.z}) / 32.f;
}

visrtx::SpatialFieldGPUData WeightedPointsField::gpuData() const
{
  visrtx::SpatialFieldGPUData sf;
  sf.samplerCallableIndex =
      visrtx::SbtCallableEntryPoints::SpatialFieldSamplerCustom;
  sf.data.custom.subType = visrtx::DEMO_WEIGHTED_POINTS_FIELD_TYPE;
  std::memcpy(sf.data.custom.fieldData, &m_deviceData,
      sizeof(WeightedPointsFieldData));
  sf.roi = bounds();
  sf.grid = m_uniformGrid.gpuData();
  return sf;
}

} // namespace visrtx_custom
