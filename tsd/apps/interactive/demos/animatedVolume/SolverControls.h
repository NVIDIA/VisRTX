// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// tsd_core
#include <tsd/scene/objects/Array.hpp>
#include <tsd/scene/objects/SpatialField.hpp>
// std
#include <functional>

namespace tsd::demo {

using JacobiUpdateCallback = std::function<void()>;

struct SolverControls : public tsd::ui::imgui::Window
{
  SolverControls(
      tsd::ui::imgui::Application *app, const char *name = "Solver Controls");

  void buildUI() override;
  void setField(tsd::scene::SpatialFieldRef f);
  void setUpdateCallback(JacobiUpdateCallback cb);

 private:
  void remakeDataArray();
  void resetSolver();
  void iterateSolver();
  void exportRAW();

  tsd::scene::ObjectUsePtr<tsd::scene::SpatialField> m_field;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataHost;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataCUDA_1;
  tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataCUDA_2;
  int m_iterationsPerCycle{2};
  tsd::math::int3 m_dims{256, 256, 256};
  int m_totalIterations{0};
  JacobiUpdateCallback m_cb;
  bool m_playing{false};
  bool m_useGPUInterop{false};
  bool m_updateTF{true};
  bool m_dumpVolumes{false};
  std::string m_exportRoot{"./"}; // root filename for .raw exports
};

} // namespace tsd::demo
