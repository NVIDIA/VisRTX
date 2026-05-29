// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Shot.h"

#include <algorithm>

namespace tsd::scivis_studio::shot {

DatasetBinding *findDatasetBinding(Shot &shot, const DatasetID &id)
{
  auto itr = std::find_if(shot.datasetBindings.begin(),
      shot.datasetBindings.end(),
      [&](const DatasetBinding &b) { return b.datasetId == id; });
  return itr == shot.datasetBindings.end() ? nullptr : &*itr;
}

const DatasetBinding *findDatasetBinding(const Shot &shot, const DatasetID &id)
{
  auto itr = std::find_if(shot.datasetBindings.begin(),
      shot.datasetBindings.end(),
      [&](const DatasetBinding &b) { return b.datasetId == id; });
  return itr == shot.datasetBindings.end() ? nullptr : &*itr;
}

void setDatasetBinding(Shot &shot, const DatasetID &id, bool enabled)
{
  if (auto *binding = findDatasetBinding(shot, id)) {
    binding->enabled = enabled;
    return;
  }

  shot.datasetBindings.push_back({id, enabled});
}

} // namespace tsd::scivis_studio::shot
