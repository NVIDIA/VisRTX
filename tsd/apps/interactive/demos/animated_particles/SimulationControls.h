// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"
// std
#include <functional>
#include <utility>

#include "particle_system.h"

namespace tsd_viewer {

struct SimulationControls : public anari_viewer::windows::Window
{
  SimulationControls(AppCore *state, const char *name = "Simulation Controls");

  void buildUI() override;
  void setGeometry(tsd::GeometryRef particles,
      tsd::GeometryRef blackHoles,
      tsd::SamplerRef particleColorSampler);

 private:
  void remakeDataArrays();
  void resetSimulation();
  void updateColorMapScale();
  std::pair<tsd::float3, tsd::float3> updateBhPoints();
  void iterateSimulation();

  AppCore *m_core{nullptr};
  tsd::GeometryRef m_particleGeom;
  tsd::GeometryRef m_bhGeom;
  tsd::SamplerRef m_particleColorSampler;
  tsd::ArrayRef m_dataPoints;
  tsd::ArrayRef m_dataPointsCUDA;
  tsd::ArrayRef m_dataDistances;
  tsd::ArrayRef m_dataDistancesCUDA;
  tsd::ArrayRef m_dataVelocities;
  tsd::ArrayRef m_dataVelocitiesCUDA;
  tsd::ArrayRef m_dataBhPoints;
  int m_particlesPerSide{100};
  tsd::ParticleSystemParameters m_params;
  float m_angle{0.f};
  float m_rotationSpeed{35.f};
  float m_colorMapScaleFactor{3.f};
  bool m_playing{false};
  bool m_useGPUInterop{true};
  bool m_randomizeInitialVelocities{true};
};

} // namespace tsd_viewer
