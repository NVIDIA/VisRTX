// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Project.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace tsd::scivis_studio {

bool Project::isSaved() const
{
  return !projectDirectory.empty();
}

void Project::markDirty()
{
  dirty = true;
}

void Project::markClean()
{
  dirty = false;
}

namespace project {

std::string makeGeneratedId(const char *prefix, size_t ordinal)
{
  std::ostringstream ss;
  ss << prefix << '_' << std::setfill('0') << std::setw(4) << ordinal;
  return ss.str();
}

DatasetID nextDatasetId(const Project &project)
{
  return makeGeneratedId("dataset", project.datasets.size() + 1);
}

ShotID nextShotId(const Project &project)
{
  return makeGeneratedId("shot", project.shots.size() + 1);
}

ColorMapID nextColorMapId(const Project &project)
{
  return makeGeneratedId("colorMap", project.colorMaps.size() + 1);
}

Dataset *findDataset(Project &project, const DatasetID &id)
{
  auto itr = std::find_if(project.datasets.begin(),
      project.datasets.end(),
      [&](const Dataset &d) { return d.id == id; });
  return itr == project.datasets.end() ? nullptr : &*itr;
}

const Dataset *findDataset(const Project &project, const DatasetID &id)
{
  auto itr = std::find_if(project.datasets.begin(),
      project.datasets.end(),
      [&](const Dataset &d) { return d.id == id; });
  return itr == project.datasets.end() ? nullptr : &*itr;
}

Shot *findShot(Project &project, const ShotID &id)
{
  auto itr = std::find_if(project.shots.begin(),
      project.shots.end(),
      [&](const Shot &s) { return s.id == id; });
  return itr == project.shots.end() ? nullptr : &*itr;
}

const Shot *findShot(const Project &project, const ShotID &id)
{
  auto itr = std::find_if(project.shots.begin(),
      project.shots.end(),
      [&](const Shot &s) { return s.id == id; });
  return itr == project.shots.end() ? nullptr : &*itr;
}

Shot *activeShot(Project &project)
{
  if (auto *shot = findShot(project, project.activeShotId))
    return shot;
  return project.shots.empty() ? nullptr : &project.shots.front();
}

const Shot *activeShot(const Project &project)
{
  if (auto *shot = findShot(project, project.activeShotId))
    return shot;
  return project.shots.empty() ? nullptr : &project.shots.front();
}

} // namespace project

} // namespace tsd::scivis_studio
