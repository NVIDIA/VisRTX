// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"

#include "tsd/ui/imgui/Application.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tsd::ui::imgui {
struct LayerTree;
struct Log;
struct ObjectEditor;
struct TransferFunctionEditor;
struct Viewport;
} // namespace tsd::ui::imgui

namespace tsd::scivis_studio {

struct AddStaticDatasetDialog;
struct AddFileAnimationDatasetDialog;
struct CameraRigEditor;
struct DatasetEditor;
struct LightRigEditor;
struct ProjectLocationDialog;
struct ProjectWindow;
struct ShotEditor;

struct ConfirmationModalState
{
  bool visible{false};
  std::string title;
  std::string message;
  std::string cancelLabel;
  std::string confirmLabel;
  float minWidth{0.f};
  std::function<void()> onCancel;
  std::function<void()> onConfirm;
};

class Application : public tsd::ui::imgui::Application
{
 public:
  Application(int argc = 0, const char **argv = nullptr);
  ~Application() override;

  ProjectContext &projectContext();
  const ProjectContext &projectContext() const;

  void showAddStaticDatasetDialog();
  void showAddFileAnimationDatasetDialog();
  void showProjectLocationDialogForOpen();
  void showProjectLocationDialogForSaveAs();
  void renderActiveShot();

 protected:
  tsd::ui::imgui::WindowArray setupWindows() override;
  void uiFrameStart() override;
  void teardown() override;
  void uiMainMenuBar() override;
  const char *getDefaultLayout() const override;

 private:
  enum class PendingDirtyAction
  {
    None,
    NewProject,
    OpenProject,
    OpenRecentProject
  };

  bool saveProject();
  bool saveProjectAs(const std::filesystem::path &directory);
  bool openProject(const std::filesystem::path &directory,
      const ProjectOpenOptions &options = {});
  void newProject();
  void saveDefaultLayoutFile() const;
  void saveWindowSettings(tsd::core::DataNode &node);
  void loadWindowSettings(tsd::core::DataNode &node);
  std::string saveLayout() const;
  void loadLayout(const std::string &layout);
  void requestDirtyAction(PendingDirtyAction action);
  void requestOpenRecentProject(const std::filesystem::path &directory);
  void continueDirtyAction();
  void loadRecentProjects();
  void saveRecentProjects() const;
  void addRecentProject(const std::filesystem::path &directory);
  void removeRecentProject(const std::filesystem::path &directory);
  void clearRecentProjects();
  void uiRecentProjectsMenu();
  std::filesystem::path recentProjectsFile() const;

  ProjectContext m_projectContext;
  std::filesystem::path m_initialProjectDirectory;
  std::filesystem::path m_pendingProjectDirectory;
  std::vector<std::filesystem::path> m_recentProjects;
  PendingDirtyAction m_pendingDirtyAction{PendingDirtyAction::None};
  bool m_viewportRenderingDisabledForShotRender{false};
  bool m_keepBlankProjectCleanAfterViewportSetup{false};

  tsd::ui::imgui::Viewport *m_viewport{nullptr};
  tsd::ui::imgui::LayerTree *m_layerTree{nullptr};
  tsd::ui::imgui::TransferFunctionEditor *m_transferFunctionEditor{nullptr};

  std::unique_ptr<ProjectLocationDialog> m_projectLocationDialog;
  std::unique_ptr<AddStaticDatasetDialog> m_addStaticDatasetDialog;
  std::unique_ptr<AddFileAnimationDatasetDialog> m_addFileAnimationDatasetDialog;
  ConfirmationModalState m_confirmationModal;
};

} // namespace tsd::scivis_studio
