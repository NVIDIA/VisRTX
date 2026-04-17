// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "spatial_field/CustomField.h"
#include "array/Array1D.h"
#include "WeightedPointsFieldData.h"

namespace visrtx_custom {

class WeightedPointsField : public visrtx::CustomField
{
 public:
  WeightedPointsField(visrtx::DeviceGlobalState *d);
  ~WeightedPointsField() override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  visrtx::box3 bounds() const override;
  float stepSize() const override;
  visrtx::SpatialFieldGPUData gpuData() const override;

 private:
  void freeDeviceArrays();

  float m_sigma{0.f};
  float m_cutoff{0.f};
  visrtx::vec3 m_domainMin{0.f, 0.f, 0.f};
  visrtx::vec3 m_domainMax{1.f, 1.f, 1.f};

  helium::ChangeObserverPtr<visrtx::Array1D> m_valuesArray;
  helium::ChangeObserverPtr<visrtx::Array1D> m_indicesArray;
  int32_t m_numNodes{0};

  float *m_dValues{nullptr};
  int32_t *m_dIndices{nullptr};
  size_t m_dValuesSize{0};
  size_t m_dIndicesSize{0};

  WeightedPointsFieldData m_deviceData{};
};

} // namespace visrtx_custom
