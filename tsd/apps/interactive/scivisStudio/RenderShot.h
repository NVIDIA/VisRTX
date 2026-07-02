// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"

#include <functional>

namespace tsd::scivis_studio {

struct RenderShotProgress
{
  std::function<bool(int frame, int totalFrames)> onFrame;
};

// What renderActiveShotToFrames loaded to materialize the shot, so the prior
// residency (and what a save would persist) can be restored afterward.
struct ShotDatasetResidencyRestore
{
  std::vector<DatasetID> loadedForRender;
  bool projectWasDirty{false};
};

// Bring every bound, enabled dataset of the shot fully resident regardless of
// its stored residency. Fails up front — restoring anything it already
// loaded — when a dataset cannot be made resident.
bool makeShotDatasetsResident(ProjectContext &projectContext,
    const Shot &shot,
    ShotDatasetResidencyRestore &restore,
    std::string *error = nullptr);

// Unload the datasets that were loaded only for rendering and restore the
// project dirty flag captured when materialization began.
void restoreShotDatasetResidency(ProjectContext &projectContext,
    const ShotDatasetResidencyRestore &restore);

bool renderActiveShotToFrames(
    ProjectContext &projectContext, RenderShotProgress *progress = nullptr);

} // namespace tsd::scivis_studio
