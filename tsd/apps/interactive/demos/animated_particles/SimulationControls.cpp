// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "SimulationControls.h"
// tsd_viewer
#include "tsd_ui.h"
// tsd
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
// cuda
#include <cuda_runtime.h>

namespace tsd_viewer {

SimulationControls::SimulationControls(AppCore *core, const char *name)
    : anari_viewer::windows::Window(name, true), m_core(core)
{}

void SimulationControls::buildUI()
{
  if (!m_particleGeom) {
    ImGui::Text("{simulation window not setup correctly}");
    return;
  }

  if (ImGui::Button("reset"))
    resetSimulation();

  ImGui::SameLine();

  ImGui::Text(" | ");

  ImGui::SameLine();

  if (ImGui::Button(m_playing ? "stop" : "play"))
    m_playing = !m_playing;

  ImGui::SameLine();

  ImGui::Text(" | ");

  ImGui::SameLine();

  ImGui::BeginDisabled(m_playing);
  if (ImGui::Button("iterate") || m_playing)
    iterateSimulation();
  ImGui::EndDisabled();

  ImGui::BeginDisabled(m_playing);

  ImGui::DragInt("particles-per-side", &m_particlesPerSide, 1, 1024);

  if (ImGui::Checkbox("randomize velocities", &m_randomizeInitialVelocities))
    resetSimulation();

  if (ImGui::Checkbox("use GPU array interop", &m_useGPUInterop)) {
    if (m_useGPUInterop)
      m_particleGeom->setParameterObject("vertex.position", *m_dataPointsCUDA);
    else
      m_particleGeom->setParameterObject("vertex.position", *m_dataPoints);
  }

  ImGui::EndDisabled();

  ImGui::Separator();

  ImGui::DragFloat("rotation speed", &m_rotationSpeed, 1.f);
  ImGui::DragFloat("gravity", &m_params.gravity, 1.f);
  ImGui::DragFloat("particle mass", &m_params.particleMass, 0.1f);
  if (ImGui::DragFloat("max distance", &m_params.maxDistance, 1.f))
    updateColorMapScale();
  if (ImGui::DragFloat("color map scale", &m_colorMapScaleFactor, 1.f))
    updateColorMapScale();
  ImGui::InputFloat("delta T", &m_params.deltaT);
}

void SimulationControls::setGeometry(tsd::GeometryRef particles,
  tsd::GeometryRef blackHoles,
  tsd::SamplerRef sampler)
{
  m_particleGeom = particles;
  m_bhGeom = blackHoles;
  m_particleColorSampler = sampler;
  updateColorMapScale();
  resetSimulation();
}

void SimulationControls::remakeDataArrays()
{
  auto &ctx = m_core->tsd.ctx;
  auto resetArrayRef = [&](auto &ref) {
    if (ref)
      ctx.removeObject(*ref);
    ref = {};
  };

  resetArrayRef(m_dataPoints);
  resetArrayRef(m_dataPointsCUDA);
  resetArrayRef(m_dataDistances);
  resetArrayRef(m_dataDistancesCUDA);
  resetArrayRef(m_dataVelocities);
  resetArrayRef(m_dataVelocitiesCUDA);
  resetArrayRef(m_dataBhPoints);

  const int numParticles =
      m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;

  m_dataPointsCUDA = ctx.createArrayCUDA(ANARI_FLOAT32_VEC3, numParticles);
  m_dataDistancesCUDA = ctx.createArrayCUDA(ANARI_FLOAT32, numParticles);
  m_dataVelocitiesCUDA = ctx.createArrayCUDA(ANARI_FLOAT32_VEC3, numParticles);
  m_dataPoints = ctx.createArray(ANARI_FLOAT32_VEC3, numParticles);
  m_dataDistances = ctx.createArray(ANARI_FLOAT32, numParticles);
  m_dataVelocities = ctx.createArray(ANARI_FLOAT32_VEC3, numParticles);
  m_dataBhPoints = ctx.createArray(ANARI_FLOAT32_VEC3, 2);
}

void SimulationControls::resetSimulation()
{
  m_playing = false;

  if (!m_particleGeom) {
    tsd::logWarning("[particle system] no geometry set, unable to initialize");
    return;
  }

  tsd::logStatus("[particle system] resetting simulation data");

  m_angle = 0.f;

  remakeDataArrays();
  updateBhPoints();

  // Generate new initial data //

  const int numParticles =
      m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;

  auto *velocities = m_dataVelocities->mapAs<tsd::math::float3>();
  if (m_randomizeInitialVelocities) {
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(-0.1f, 0.1f);
    std::for_each((float*) velocities,
      (float*)(velocities + numParticles),
      [&](auto &v) { v = dist(rng) * 255; });
  } else {
    std::fill(velocities, velocities + numParticles, tsd::float3(0.f));
  }

  auto *distances = m_dataDistances->mapAs<float>();

  auto *positions = m_dataPoints->mapAs<tsd::math::float3>();
  const float d = 2.0f / m_particlesPerSide;
  size_t i = 0;
  for (int x = 0; x < m_particlesPerSide; x++) {
    for (int y = 0; y < m_particlesPerSide; y++) {
      for (int z = 0; z < m_particlesPerSide; z++) {
        auto p = tsd::float3(d * x - 1.f, d * y - 1.f, d * z - 1.f);
        positions[i] = p;
        distances[i++] = tsd::math::length(p);
      }
    }
  }

  cudaMemcpy(m_dataVelocitiesCUDA->map(),
      velocities,
      m_dataVelocitiesCUDA->size() * m_dataVelocitiesCUDA->elementSize(),
      cudaMemcpyHostToDevice);
  m_dataVelocitiesCUDA->unmap();

  cudaMemcpy(m_dataPointsCUDA->map(),
      positions,
      m_dataPointsCUDA->size() * m_dataPointsCUDA->elementSize(),
      cudaMemcpyHostToDevice);
  m_dataPointsCUDA->unmap();

  cudaMemcpy(m_dataDistancesCUDA->map(),
      distances,
      m_dataDistancesCUDA->size() * m_dataDistancesCUDA->elementSize(),
      cudaMemcpyHostToDevice);
  m_dataDistancesCUDA->unmap();

  m_dataVelocities->unmap();
  m_dataPoints->unmap();
  m_dataDistances->unmap();

  if (m_useGPUInterop) {
    m_particleGeom->setParameterObject("vertex.position", *m_dataPointsCUDA);
    m_particleGeom->setParameterObject("vertex.attribute0",
      *m_dataDistancesCUDA);
  } else {
    m_particleGeom->setParameterObject("vertex.position", *m_dataPoints);
    m_particleGeom->setParameterObject("vertex.attribute0",
      *m_dataDistances);
  }

  m_bhGeom->setParameterObject("vertex.position", *m_dataBhPoints);
}

void SimulationControls::updateColorMapScale()
{
  m_particleColorSampler->setParameter(
      "inTransform",
      tsd::makeColorMapTransform(0.f,
                                 m_params.maxDistance / m_colorMapScaleFactor)
  );
}

std::pair<tsd::float3, tsd::float3> SimulationControls::updateBhPoints()
{
  const auto rot = tsd::math::rotation_matrix(
      tsd::math::rotation_quat(tsd::math::float3(0, 0, 1), m_angle));
  tsd::float4 bh1_ = tsd::math::mul(rot, tsd::float4(5.f, 0.f, 0.f, 1.f));
  tsd::float4 bh2_ = tsd::math::mul(rot,tsd::float4(-5.f, 0.f, 0.f, 1.f));

  tsd::float3 bh1(bh1_.x, bh1_.y, bh1_.z);
  tsd::float3 bh2(bh2_.x, bh2_.y, bh2_.z);

  auto *bhPoints = m_dataBhPoints->mapAs<tsd::float3>();
  bhPoints[0] = bh1;
  bhPoints[1] = bh2;
  m_dataBhPoints->unmap();

  return std::make_pair(bh1, bh2);
}

void SimulationControls::iterateSimulation()
{
  if (!m_particleGeom) {
    tsd::logWarning("[particle system] no geometry set, unable to run");
    return;
  }

  auto start = std::chrono::steady_clock::now();

  m_angle += m_rotationSpeed * 1e-4f;
  if (m_angle > 360.f)
    m_angle -= 360.f;

  auto [bh1, bh2] = updateBhPoints();

  const int numParticles =
      m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;

  tsd::logStatus("[particle system] running time step");
  tsd::logStatus("    angle: %f", m_angle);

  auto *pointsCUDA = m_dataPointsCUDA->mapAs<tsd::float3>();
  auto *velocitiesCUDA = m_dataVelocitiesCUDA->mapAs<tsd::float3>();
  auto *distancesCUDA = m_dataDistancesCUDA->mapAs<float>();

  tsd::particlesComputeTimestep(
    numParticles,
    pointsCUDA,
    velocitiesCUDA,
    distancesCUDA,
    tsd::float3(bh1.x, bh1.y, bh1.z),
    tsd::float3(bh2.x, bh2.y, bh2.z),
    m_params
  );

  if (!m_useGPUInterop)
  {
    auto *points = m_dataPoints->mapAs<tsd::float3>();
    auto *velocities = m_dataVelocities->mapAs<tsd::float3>();
    auto *distances = m_dataDistances->mapAs<float>();

    cudaMemcpy(velocities,
        velocitiesCUDA,
        m_dataVelocitiesCUDA->size() * m_dataVelocitiesCUDA->elementSize(),
        cudaMemcpyDeviceToHost);
    m_dataVelocities->unmap();

    cudaMemcpy(points,
        pointsCUDA,
        m_dataPointsCUDA->size() * m_dataPointsCUDA->elementSize(),
        cudaMemcpyDeviceToHost);
    m_dataPoints->unmap();

    cudaMemcpy(distances,
        distancesCUDA,
        m_dataDistancesCUDA->size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    m_dataDistances->unmap();
  }

  m_dataPointsCUDA->unmap();
  m_dataVelocitiesCUDA->unmap();
  m_dataDistancesCUDA->unmap();
}

} // namespace tsd_viewer