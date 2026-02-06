// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Scene.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/view/Manipulator.hpp"

#include "tsd/app/renderAnimationSequence.h"

namespace tsd::app {

using CameraPose = tsd::rendering::CameraPose;

struct CoreController
{
  virtual void parseCommandLine(int argc, const char **argv) = 0;
  virtual void setupSceneFromCommandLine(bool hdriOnly = false) = 0;

  // Offline rendering //

  virtual void setOfflineRenderingLibrary(const std::string &libName) = 0;

  // Selection //

  virtual tsd::core::LayerNodeRef getFirstSelected() const = 0;
  virtual const std::vector<tsd::core::LayerNodeRef> &getSelectedNodes()
      const = 0;
  virtual void setSelected(tsd::core::LayerNodeRef node) = 0;
  virtual void setSelected(
      const std::vector<tsd::core::LayerNodeRef> &nodes) = 0;
  virtual void setSelected(const tsd::core::Object *obj) = 0;
  virtual void addToSelection(tsd::core::LayerNodeRef node) = 0;
  virtual void removeFromSelection(tsd::core::LayerNodeRef node) = 0;
  virtual bool isSelected(tsd::core::LayerNodeRef node) const = 0;
  virtual void clearSelected() = 0;

  // Returns only parent nodes from selection (filters out children of selected
  // nodes)
  virtual std::vector<tsd::core::LayerNodeRef> getParentOnlySelectedNodes()
      const = 0;

  // Camera poses //

  virtual void addCurrentViewToCameraPoses(const char *name = "") = 0;
  virtual void addTurntableCameraPoses(
      const tsd::math::float3 &azimuths, // begin, end, step
      const tsd::math::float3 &elevations, // begin, end, step
      const tsd::math::float3 &center,
      float distance,
      const char *name = "") = 0;
  virtual void updateExistingCameraPoseFromView(CameraPose &p) = 0;
  virtual void setCameraPose(const CameraPose &pose) = 0;
  virtual void removeAllPoses() = 0;
  virtual bool updateCameraPathAnimation() = 0;
};

} // namespace tsd::app
