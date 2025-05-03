// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"
// std
#include <functional>

namespace tsd_viewer {

using JacobiUpdateCallback = std::function<void()>;

struct SolverControls : public anari_viewer::windows::Window
{
  SolverControls(AppCore *core, const char *name = "Solver Controls");

  void buildUI() override;
  void setField(tsd::SpatialFieldRef f);
  void setUpdateCallback(JacobiUpdateCallback cb);

 private:
  void remakeDataArray();
  void resetSolver();
  void iterateSolver();
  void exportRAW();

  AppCore *m_core{nullptr};
  tsd::SpatialFieldRef m_field;
  tsd::ArrayRef m_dataHost;
  tsd::ArrayRef m_dataCUDA_1;
  tsd::ArrayRef m_dataCUDA_2;
  int m_iterationsPerCycle{2};
  tsd::int3 m_dims{256, 256, 256};
  int m_totalIterations{0};
  JacobiUpdateCallback m_cb;
  bool m_playing{false};
  bool m_useGPUInterop{false};
  bool m_updateTF{true};
};

} // namespace tsd_viewer
