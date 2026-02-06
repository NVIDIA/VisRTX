// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// tsd_core
#include <tsd/core/scene/objects/Array.hpp>
#include <tsd/core/scene/objects/SpatialField.hpp>
// std
#include <functional>

namespace tsd::demo {

using JacobiUpdateCallback = std::function<void()>;

struct SolverControls : public tsd::ui::imgui::Window
{
  SolverControls(
      tsd::ui::imgui::Application *app, const char *name = "Solver Controls");

  void buildUI() override;
  void setField(tsd::core::SpatialFieldRef f);
  void setUpdateCallback(JacobiUpdateCallback cb);

 private:
  void remakeDataArray();
  void resetSolver();
  void iterateSolver();
  void exportRAW();

  tsd::core::ObjectUsePtr<tsd::core::SpatialField> m_field;
  tsd::core::ObjectUsePtr<tsd::core::Array> m_dataHost;
  tsd::core::ObjectUsePtr<tsd::core::Array> m_dataCUDA_1;
  tsd::core::ObjectUsePtr<tsd::core::Array> m_dataCUDA_2;
  int m_iterationsPerCycle{2};
  tsd::math::int3 m_dims{256, 256, 256};
  int m_totalIterations{0};
  JacobiUpdateCallback m_cb;
  bool m_playing{false};
  bool m_useGPUInterop{false};
  bool m_updateTF{true};
};

} // namespace tsd::demo
