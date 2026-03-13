// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_app
#include <tsd/app/Context.h>
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
  void setGeometry(tsd::scene::GeometryRef particles,
      tsd::scene::GeometryRef blackHoles,
      tsd::scene::SamplerRef particleColorSampler);

 private:
  void remakeDataArrays();
  void resetSimulation();
  void updateColorMapScale();
  std::pair<tsd::math::float3, tsd::math::float3> updateBhPoints();
  void iterateSimulation();

  tsd::scene::ObjectUsePtr<tsd::scene::Geometry> m_particleGeom;
  tsd::scene::ObjectUsePtr<tsd::scene::Geometry> m_bhGeom;
  tsd::scene::ObjectUsePtr<tsd::scene::Sampler> m_particleColorSampler;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataPoints;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataPointsCUDA;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataDistances;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataDistancesCUDA;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataVelocities;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataVelocitiesCUDA;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataBhPoints;
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
