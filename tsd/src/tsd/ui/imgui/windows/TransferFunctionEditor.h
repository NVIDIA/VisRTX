// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Volume.hpp"
// std
#include <string>
#include <vector>
// SDL
#include <SDL3/SDL.h>

namespace tsd::ui::imgui {

class TransferFunctionEditor : public Window
{
 public:
  TransferFunctionEditor(Application *app, const char *name = "TF Editor");
  ~TransferFunctionEditor() override;

  void buildUI() override;

 private:
  void buildUI_selectColorMap();
  void buildUI_drawEditor();
  void buildUI_opacityScale();
  void buildUI_valueRange();

  std::vector<tsd::math::float4> getSampledColorsAndOpacities(
      int numSamples = 256);

  void setMap(int which = 0);
  void setObjectPtrsFromSelectedObject();
  void loadDefaultMaps();
  void loadColormap(
      const std::string &filepath, const std::string &name);

  void saveColormapTo1dt(const std::string &filepath);
  void saveColormapToParaview(const std::string &filepath);
  void getTransferFunctionFilenameFromDialog(
      std::string &filenameOut, bool save = false);
  void updateVolume();
  void updateTfnPaletteTexture();
  void resizeTfnPaletteTexture(size_t width);

  // Data //

  tsd::core::Volume *m_volume{nullptr};
  tsd::core::Array *m_colorMapArray{nullptr};

  // all available transfer functions
  std::vector<std::string> m_tfnsNames;
  std::vector<std::vector<tsd::core::ColorPoint>> m_tfnsColorPoints;
  std::vector<tsd::core::OpacityPoint> m_tfnOpacityPoints;

  // parameters of currently selected transfer function
  int m_currentMap{-1};
  int m_nextMap{0};
  std::vector<tsd::core::ColorPoint> *m_tfnColorPoints{nullptr};

  // domain (value range) of transfer function
  tsd::math::float2 m_valueRange{0.f, 1.f};
  tsd::math::float2 m_defaultValueRange{0.f, 1.f};

  // texture for displaying transfer function color palette
  SDL_Texture *m_tfnPaletteTexture{nullptr};
  size_t m_tfnPaletteWidth{0};

  // New member for storing the filename of the currently loaded colormap
  std::string m_currentColormapFilename;
  std::string m_saveColormapFilename;
};

} // namespace tsd::ui::imgui
