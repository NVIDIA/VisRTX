// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct ImportFileDialog;

struct ObjectTree : public anari_viewer::windows::Window
{
  ObjectTree(AppCore *state, const char *name = "Object Tree");
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
