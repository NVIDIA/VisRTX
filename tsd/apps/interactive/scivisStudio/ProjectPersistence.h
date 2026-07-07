// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Project.h"
#include "ProjectAssetTransaction.h"

#include "tsd/core/DataTree.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tsd::animation {
struct AnimationManager;
}

namespace tsd::core {
struct DataNode;
}

namespace tsd::scene {
struct Scene;
}

namespace tsd::scivis_studio {

namespace detail {
struct ProjectOpenState;
}

struct ProjectSaveRequest
{
  ProjectSaveRequest(const Project &project,
      const tsd::scene::Scene &scene,
      tsd::animation::AnimationManager &animationManager,
      std::filesystem::path directory);

  const Project &project;
  const tsd::scene::Scene &scene;
  tsd::animation::AnimationManager &animationManager;
  std::filesystem::path directory;
  std::vector<std::filesystem::path> pendingAssetRemovals;
  const tsd::core::DataNode *windows{nullptr};
  std::string layout;
  const tsd::core::DataNode *settings{nullptr};
};

struct ProjectSaveResult
{
  Project project;
  ProjectSavePlan plan;
};

bool buildProjectSavePlan(const ProjectSaveRequest &request,
    ProjectSaveResult &result,
    std::string *error = nullptr);

// Overrides applied while staging a project open. openUnloaded changes each
// dataset's initial residency; the project opens dirty when that override
// diverges from the manifest, so a subsequent save persists actual residency.
// bookkeeping instead opens without building any dataset runtime
// representation while leaving recorded residency untouched and the project
// clean: residency records intent, not process state, so a bookkeeping open
// must round-trip it unchanged. bookkeeping wins when both are set.
struct ProjectOpenOptions
{
  bool openUnloaded{false};
  bool bookkeeping{false};
};

struct ProjectOpenStage
{
  Project project;
  tsd::core::DataTree ui;

 private:
  std::shared_ptr<detail::ProjectOpenState> m_state;

  friend bool stageProjectOpen(const std::filesystem::path &,
      ProjectOpenStage &,
      const ProjectOpenOptions &,
      std::string *);
  friend bool applyProjectOpen(ProjectOpenStage &,
      tsd::scene::Scene &,
      tsd::animation::AnimationManager &,
      std::string *);
};

bool stageProjectOpen(const std::filesystem::path &directory,
    ProjectOpenStage &stage,
    const ProjectOpenOptions &options = {},
    std::string *error = nullptr);
bool applyProjectOpen(ProjectOpenStage &stage,
    tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    std::string *error = nullptr);

} // namespace tsd::scivis_studio
