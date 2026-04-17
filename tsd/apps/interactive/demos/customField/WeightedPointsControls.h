// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tsd/app/Context.h>
#include <tsd/ui/imgui/windows/Window.h>
#include <string>
#include <tsd/scene/Object.hpp>
#include <vector>

namespace tsd::demo {

struct WeightedPointsControls : public tsd::ui::imgui::Window
{
  WeightedPointsControls(tsd::ui::imgui::Application *app,
      const char *name = "Weighted Points Controls",
      const std::string &pdbPath = {});

  void buildUI() override;

 private:
  void createScene();
  void generatePoints();
  void rebuildField();

  std::vector<float> generateRandomUniform();
  std::vector<float> loadPDB(const std::string &path);

  bool m_usePDB{false};
  int m_numPoints{2000};
  float m_sigmaOverride{0.f};
  float m_cutoffOverride{0.f};

  std::string m_pdbPath;
  bool m_sceneCreated{false};

  tsd::scene::SpatialFieldRef m_field;
  tsd::scene::VolumeRef m_volume;
  tsd::scene::Object *m_light{nullptr};

  std::vector<float> m_rawPoints;
};

} // namespace tsd::demo
