// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Project.h"

#include "tsd/app/Context.h"
#include "tsd/io/importers.hpp"

#include <filesystem>
#include <string>

namespace tsd::scivis_studio {

struct ProjectContext
{
  ProjectContext() = default;
  explicit ProjectContext(tsd::app::Context *ctx);

  void setAppContext(tsd::app::Context *ctx);
  tsd::app::Context *appContext() const;

  Project &project();
  const Project &project() const;

  void createUnsavedProject();
  bool addShot(const std::string &name = "");
  Dataset *addStaticDataset(const std::string &name,
      const std::filesystem::path &sourcePath,
      tsd::io::ImporterType importerType);
  void applyActiveShot();
  void syncAnimationManagerToActiveShot();

  bool saveProject(const std::filesystem::path &directory,
      tsd::core::DataNode *windows = nullptr,
      const std::string &layout = "",
      tsd::core::DataNode *settings = nullptr,
      std::string *error = nullptr);
  bool openProject(const std::filesystem::path &directory,
      tsd::core::DataNode *windowsOut = nullptr,
      std::string *layoutOut = nullptr,
      tsd::core::DataNode *settingsOut = nullptr,
      std::string *error = nullptr);

  tsd::scene::LayerNodeRef resolve(const SceneNodeRef &ref) const;
  tsd::scene::Object *resolve(const SceneObjectRef &ref) const;
  SceneNodeRef refFor(
      const std::string &layerName, tsd::scene::LayerNodeRef ref) const;

 private:
  tsd::scene::LayerNodeRef ensureChild(
      tsd::scene::LayerNodeRef parent, const char *name);
  tsd::scene::LayerNodeRef ensureStudioRoot();
  tsd::scene::LayerNodeRef ensureDatasetsRoot();
  tsd::scene::LayerNodeRef ensureShotsRoot();
  void resetScene();
  void ensureRendererDefaults(Shot &shot);
  void markMissingDatasets();
  void installAnimationManagerCallback();
  void updateActiveShotFromAnimationTime();

  tsd::app::Context *m_ctx{nullptr};
  Project m_project;
  bool m_syncingAnimationManager{false};
};

const char *toString(tsd::io::ImporterType importerType);
tsd::io::ImporterType importerTypeFromString(const std::string &s);

} // namespace tsd::scivis_studio
