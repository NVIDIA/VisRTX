// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
// tsd
#include "tsd/core/IndexedVector.hpp"

namespace tsd_viewer {

struct ImportFileDialog;

struct LayerTree : public Window
{
  LayerTree(AppCore *state, const char *name = "Layers");
  void buildUI() override;

 private:
  void buildUI_layerHeader();
  void buildUI_tree();
  void buildUI_activateObjectContextMenu();
  void buildUI_buildObjectContextMenu();
  void buildUI_buildNewLayerContextMenu();

  // Data //

  size_t m_hoveredNode{tsd::INVALID_INDEX};
  size_t m_menuNode{tsd::INVALID_INDEX};
  bool m_editingNodeName{false};
  bool m_menuVisible{false};
  std::vector<int> m_needToTreePop;
  int m_layerIdx{0};
};

} // namespace tsd_viewer
