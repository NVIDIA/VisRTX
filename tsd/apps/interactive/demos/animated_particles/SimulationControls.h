// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_app
#include <tsd/app/Core.h>
// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// std
#include <functional>
#include <utility>

#include "particle_system.h"

namespace tsd::demo {

struct SimulationControls : public tsd::ui::imgui::Window
{
  SimulationControls(tsd::ui::imgui::Application *app,
      const char *name = "Simulation Controls");

  void buildUI() override;
  void setGeometry(tsd::core::GeometryRef particles,
      tsd::core::GeometryRef blackHoles,
      tsd::core::SamplerRef particleColorSampler);

 private:
  void remakeDataArrays();
  void resetSimulation();
  void updateColorMapScale();
  std::pair<tsd::math::float3, tsd::math::float3> updateBhPoints();
  void iterateSimulation();

  tsd::core::GeometryRef m_particleGeom;
  tsd::core::GeometryRef m_bhGeom;
  tsd::core::SamplerRef m_particleColorSampler;
  tsd::core::ArrayRef m_dataPoints;
  tsd::core::ArrayRef m_dataPointsCUDA;
  tsd::core::ArrayRef m_dataDistances;
  tsd::core::ArrayRef m_dataDistancesCUDA;
  tsd::core::ArrayRef m_dataVelocities;
  tsd::core::ArrayRef m_dataVelocitiesCUDA;
  tsd::core::ArrayRef m_dataBhPoints;
  int m_particlesPerSide{100};
  tsd::demo::ParticleSystemParameters m_params;
  float m_angle{0.f};
  float m_rotationSpeed{35.f};
  float m_colorMapScaleFactor{3.f};
  bool m_playing{false};
  bool m_useGPUInterop{true};
  bool m_randomizeInitialVelocities{true};
};

} // namespace tsd::demo
