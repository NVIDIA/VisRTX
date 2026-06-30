// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Dataset.h"

#include <string>

namespace tsd::scivis_studio {

struct Project;

struct LightRig
{
  LightRigID id;
  std::string name;
  SceneNodeRef rootNode;

  // Runtime-only name of the asset path owned by this rig.
  std::string persistedName;
};

namespace light_rig {

// Collection lookups within a Project. (Light rig value data lives in the scene
// graph as a node subtree, so its file IO is a scene-aware ProjectContext
// method rather than a free function here.)
LightRigID nextLightRigId(const Project &project);
LightRig *findLightRig(Project &project, const LightRigID &id);
const LightRig *findLightRig(const Project &project, const LightRigID &id);

} // namespace light_rig

} // namespace tsd::scivis_studio
