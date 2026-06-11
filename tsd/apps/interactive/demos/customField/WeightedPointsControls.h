// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tsd/app/Context.h>
#include <tsd/ui/imgui/windows/Window.h>
#include <string>
#include <tsd/scene/Object.hpp>
#include <tsd/scene/objects/Array.hpp>
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
  void rebuildFieldFast();
  void setupAnimation();
  void perturbPoints(float t);

  // Swap the field's values/indices arrays, releasing the previous pair.
  void swapFieldArrays(
      const std::vector<float> &values, const std::vector<int32_t> &indices);

  std::vector<float> generateRandomUniform();
  std::vector<float> loadPDB(const std::string &path);

  bool m_usePDB{false};
  int m_numPoints{2000};
  float m_sigmaOverride{0.f};
  float m_cutoffOverride{0.f};
  float m_perturbAmplitude{1.f}; // in units of median nearest-neighbor distance
  float m_perturbFrequency{2.f};

  std::string m_pdbPath;
  bool m_sceneCreated{false};
  bool m_animationSetup{false};

  tsd::scene::SpatialFieldRef m_field;
  tsd::scene::VolumeRef m_volume;
  tsd::scene::Object *m_light{nullptr};

  // Current field data arrays, tracked so the previous ones can be released
  // each rebuild (otherwise every animation frame leaks a new pair).
  tsd::scene::ArrayRef m_valuesArrayRef;
  tsd::scene::ArrayRef m_indicesArrayRef;
  tsd::scene::ArrayRef m_colorArrayRef;
  // Gaussian width chosen by the last full rebuild; reused (not recomputed) by
  // the per-frame fast path so blob size stays stable across the animation.
  float m_effectiveSigma{1.f};
  // Median nearest-neighbor distance from the last full rebuild; the animation
  // amplitude is expressed in multiples of this so motion scales with the data.
  float m_medianNN{1.f};

  std::vector<float> m_rawPoints;
  std::vector<float> m_originalPoints;
};

} // namespace tsd::demo
