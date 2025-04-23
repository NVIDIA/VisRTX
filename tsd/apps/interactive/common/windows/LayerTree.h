// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct ImportFileDialog;

struct LayerTree : public anari_viewer::windows::Window
{
  LayerTree(AppCore *state, const char *name = "Layers");
  void buildUI() override;

 private:
  void buildUI_objectContextMenu();

  AppCore *m_core{nullptr};
  size_t m_hoveredNode{tsd::INVALID_INDEX};
  size_t m_menuNode{tsd::INVALID_INDEX};
  bool m_menuVisible{false};
  std::vector<int> m_needToTreePop;
};

} // namespace tsd_viewer
