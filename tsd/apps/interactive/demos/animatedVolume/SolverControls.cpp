// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "SolverControls.h"
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
// tsd_core
#include <tsd/core/Logging.hpp>
// std
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
// thrust
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include "jacobi3D.h"

namespace tsd::demo {

SolverControls::SolverControls(
    tsd::ui::imgui::Application *app, const char *name)
    : tsd::ui::imgui::Window(app, name)
{}

void SolverControls::buildUI()
{
  if (!m_field) {
    ImGui::Text("{solver window not setup correctly}");
    return;
  }

  if (ImGui::Button("reset"))
    resetSolver();

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
    iterateSolver();
  ImGui::EndDisabled();

  static bool squareDims = true;
  ImGui::Checkbox("cubic dimensions", &squareDims);

  if (squareDims) {
    if (ImGui::DragInt("dim", &m_dims.x, 1, 1024))
      m_dims.z = m_dims.y = m_dims.x;
  } else
    ImGui::DragInt3("dims", &m_dims.x, 1, 1024);

  ImGui::SliderInt("iterations-per-cycle", &m_iterationsPerCycle, 1, 100);

  ImGui::BeginDisabled(m_totalIterations > 0);
  if (ImGui::Checkbox("use GPU array interop", &m_useGPUInterop))
    resetSolver();
  ImGui::EndDisabled();
  ImGui::Checkbox("auto update transfer function", &m_updateTF);

  ImGui::Separator();

  if (ImGui::Button("export .raw"))
    exportRAW();
}

void SolverControls::setField(tsd::core::SpatialFieldRef f)
{
  m_field = f;
  resetSolver();
}

void SolverControls::setUpdateCallback(JacobiUpdateCallback cb)
{
  m_cb = cb;
}

void SolverControls::remakeDataArray()
{
  if (!m_field) {
    tsd::core::logWarning("[jacobi solver] no spatial field set");
    return;
  }

  auto &scene = appCore()->tsd.scene;
  if (m_dataHost)
    scene.removeObject(m_dataHost.get());
  if (m_dataCUDA_1)
    scene.removeObject(m_dataCUDA_1.get());
  if (m_dataCUDA_2)
    scene.removeObject(m_dataCUDA_2.get());

  m_dataHost = {};
  m_dataCUDA_1 = {};
  m_dataCUDA_2 = {};

  if (m_useGPUInterop) {
    m_dataCUDA_1 =
        scene.createArrayCUDA(ANARI_FLOAT32, m_dims.x, m_dims.y, m_dims.z);
    m_dataCUDA_2 =
        scene.createArrayCUDA(ANARI_FLOAT32, m_dims.x, m_dims.y, m_dims.z);
  } else {
    m_dataHost = scene.createArray(ANARI_FLOAT32, m_dims.x, m_dims.y, m_dims.z);
  }
  m_field->setParameter("spacing", 2.f / tsd::math::float3(m_dims.x));
}

void SolverControls::resetSolver()
{
  m_playing = false;

  if (!m_field) {
    tsd::core::logWarning("[jacobi solver] no spatial field set");
    return;
  }

  tsd::core::logStatus("[jacobi solver] resetting solver data");

  remakeDataArray();

  // Generate new initial data //

  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<float> dist(-100.f, 100.0f);

  const auto nx = m_dims.x;
  const auto ny = m_dims.y;
  const auto nz = m_dims.z;
  std::vector<float> h_grid(nx * ny * nz);

  // Example: Set initial values and boundary conditions
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const auto idx = size_t(x) + size_t(nx) * (y + size_t(ny) * z);
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0
            || z == nz - 1)
          h_grid[idx] = 1.0; // Boundary condition
        else
          h_grid[idx] = dist(rng);
      }
    }
  }

  if (m_useGPUInterop) {
    cudaMemcpy(m_dataCUDA_2->map(),
        h_grid.data(),
        m_dataCUDA_2->size() * anari::sizeOf(m_dataCUDA_2->elementSize()),
        cudaMemcpyHostToDevice);
    m_dataCUDA_2->unmap();
    m_field->setParameterObject("data", *m_dataCUDA_2);
  } else {
    std::memcpy(m_dataHost->map(),
        h_grid.data(),
        m_dataHost->size() * anari::sizeOf(m_dataHost->elementSize()));
    m_dataHost->unmap();
    m_field->setParameterObject("data", *m_dataHost);
  }

  if (m_cb)
    m_cb();

  m_totalIterations = 0;
}

void SolverControls::iterateSolver()
{
  if (!m_field) {
    tsd::core::logWarning("[jacobi solver] no spatial field set");
    return;
  }

  tsd::core::logStatus(
      "[jacobi solver] running %i iterations...", m_iterationsPerCycle);

  auto start = std::chrono::steady_clock::now();

  const auto nx = m_dims.x;
  const auto ny = m_dims.y;
  const auto nz = m_dims.z;
  if (m_useGPUInterop) {
    auto *d_grid_1 = m_dataCUDA_1->mapAs<float>();
    auto *d_grid_2 = m_dataCUDA_2->mapAs<float>();
    if (m_totalIterations % 2)
      std::swap(d_grid_1, d_grid_2);
    tsd::demo::jacobi3D(nx, ny, nz, d_grid_1, d_grid_2, m_iterationsPerCycle);
    m_dataCUDA_1->unmap();
    m_dataCUDA_2->unmap();
    if (m_iterationsPerCycle % 2) {
      m_field->setParameterObject(
          "data", m_totalIterations % 2 ? *m_dataCUDA_1 : *m_dataCUDA_2);
    }
  } else {
    auto *h_grid = m_dataHost->mapAs<float>();
    tsd::demo::jacobi3D(nx, ny, nz, h_grid, m_iterationsPerCycle);
    m_dataHost->unmap();
  }

  m_totalIterations += m_iterationsPerCycle;

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<float>(end - start).count() * 1000;

  tsd::core::logStatus("[jacobi solver] ...solver done (%i|%.2fms)",
      m_totalIterations,
      duration);

  start = std::chrono::steady_clock::now();

  if (m_updateTF && m_cb)
    m_cb();

  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration<float>(end - start).count() * 1000;

  tsd::core::logStatus("[jacobi solver] ...TSD update  (%i|%.2fms)",
      m_totalIterations,
      duration);
}

void SolverControls::exportRAW()
{
  const auto nx = size_t(m_dims.x);
  const auto ny = size_t(m_dims.y);
  const auto nz = size_t(m_dims.z);

  std::string filename;
  filename.resize(100);
  filename.resize(std::snprintf(filename.data(),
      filename.size(),
      "jacobi_%i_%zux%zux%zu_float32.raw",
      m_totalIterations,
      nx,
      ny,
      nz));

  if (m_useGPUInterop) {
    cudaMemcpy(m_dataHost->map(),
        m_dataCUDA_2->map(),
        m_dataHost->size() * anari::sizeOf(m_dataHost->elementSize()),
        cudaMemcpyDeviceToHost);
    m_dataCUDA_2->unmap();
    m_dataHost->unmap();
  }

  auto *fp = std::fopen(filename.c_str(), "wb");
  std::fwrite(
      m_dataHost->dataAs<float>(), sizeof(float), m_dataHost->size(), fp);
  std::fclose(fp);

  tsd::core::logStatus("[jacobi solver] exported data to %s", filename.c_str());
}

} // namespace tsd::demo