// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../AppCore.h"
// anari
#include "anari_viewer/windows/Window.h"
// tsd
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/ColorMapUtil.hpp"
// std
#include <functional>
#include <string>
#include <vector>
// SDL
#include <SDL3/SDL.h>

namespace tsd_viewer {

class TransferFunctionEditor : public anari_viewer::windows::Window
{
 public:
  TransferFunctionEditor(AppCore *ctx, const char *name = "TF Editor");
  ~TransferFunctionEditor() override;

  void buildUI() override;

 private:
  void buildUI_selectColorMap();
  void buildUI_drawEditor();
  void buildUI_opacityScale();
  void buildUI_valueRange();

  std::vector<tsd::float4> getSampledColorsAndOpacities(int numSamples = 256);

  void setMap(int which = 0);
  void setObjectPtrsFromSelectedObject();
  void loadDefaultMaps();
  void updateVolume();
  void updateTfnPaletteTexture();
  void resizeTfnPaletteTexture(size_t width);

  // Data //

  AppCore *m_core{nullptr};

  tsd::Volume *m_volume{nullptr};
  tsd::Array *m_colorMapArray{nullptr};

  // all available transfer functions
  std::vector<std::string> m_tfnsNames;
  std::vector<std::vector<tsd::ColorPoint>> m_tfnsColorPoints;
  std::vector<tsd::OpacityPoint> m_tfnOpacityPoints;

  // parameters of currently selected transfer function
  int m_currentMap{-1};
  int m_nextMap{0};
  std::vector<tsd::ColorPoint> *m_tfnColorPoints{nullptr};

  // flag indicating transfer function has changed in UI
  bool m_tfnChanged{true};

  // scaling factor for generated opacities
  float m_globalOpacityScale{1.f};

  // domain (value range) of transfer function
  tsd::float2 m_valueRange{0.f, 1.f};
  tsd::float2 m_defaultValueRange{0.f, 1.f};

  // texture for displaying transfer function color palette
  SDL_Texture *m_tfnPaletteTexture{nullptr};
  size_t m_tfnPaletteWidth{0};
};

} // namespace tsd_viewer
