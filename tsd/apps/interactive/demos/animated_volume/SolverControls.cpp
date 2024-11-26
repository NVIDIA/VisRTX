// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "SolverControls.h"
// tsd_viewer
#include "tsd_ui.h"
// tsd
#include "tsd/core/Logging.hpp"
// std
#include <cstdio>
#include <random>
#include <vector>

#include "jacobi3D.h"

namespace tsd_viewer {

SolverControls::SolverControls(AppCore *core, const char *name)
    : anari_viewer::windows::Window(name, true), m_core(core)
{}

void SolverControls::buildUI()
{
  if (m_data == nullptr) {
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

  ImGui::SliderInt("iterations-per-cycle", &m_iterationsPerCycle, 1, 100);

  ImGui::Separator();

  if (ImGui::Button("export"))
    exportRAW();
}

void SolverControls::setDataArray(tsd::Array *a)
{
  m_data = a;
  resetSolver();
}

void SolverControls::setUpdateCallback(JacobiUpdateCallback cb)
{
  m_cb = cb;
}

void SolverControls::resetSolver()
{
  if (!m_data) {
    tsd::logWarning("[jacobi solver] no data array set");
    return;
  }

  tsd::logStatus("[jacobi solver] resetting solver data");

  // Generate new initial data //

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> dist(0.f, 100.0f);

  const auto nx = m_data->dim(0);
  const auto ny = m_data->dim(1);
  const auto nz = m_data->dim(2);
  auto *h_grid = m_data->mapAs<float>();

  // Example: Set initial values and boundary conditions
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const auto idx = x + nx * (y + ny * z);
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0
            || z == nz - 1)
          h_grid[idx] = 1.0; // Boundary condition
        else
          h_grid[idx] = dist(rng);
      }
    }
  }

  m_data->unmap();

  if (m_cb)
    m_cb();

  m_totalIterations = 0;
}

void SolverControls::iterateSolver()
{
  if (!m_data) {
    tsd::logWarning("[jacobi solver] no data array set");
    return;
  }

  tsd::logStatus(
      "[jacobi solver] running %i iterations...", m_iterationsPerCycle);

  const auto nx = m_data->dim(0);
  const auto ny = m_data->dim(1);
  const auto nz = m_data->dim(2);
  auto *h_grid = m_data->mapAs<float>();
  tsd::jacobi3D(nx, ny, nz, h_grid, m_iterationsPerCycle);
  m_data->unmap();

  m_totalIterations += m_iterationsPerCycle;

  if (m_cb)
    m_cb();

  tsd::logStatus("[jacobi solver] ...done (%i)", m_totalIterations);
}

void SolverControls::exportRAW()
{
  const auto nx = m_data->dim(0);
  const auto ny = m_data->dim(1);
  const auto nz = m_data->dim(2);

  std::string filename;
  filename.resize(100);
  filename.resize(std::snprintf(filename.data(),
      filename.size(),
      "jacobi_%i_%zux%zux%zu_float32.raw",
      m_totalIterations,
      nx,
      ny,
      nz));
  auto *fp = std::fopen(filename.c_str(), "wb");
  std::fwrite(m_data->dataAs<float>(), sizeof(float), m_data->size(), fp);
  std::fclose(fp);

  tsd::logStatus("[jacobi solver] exported data to %s", filename.c_str());
}

} // namespace tsd_viewer