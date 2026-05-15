// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"

#include "tsd/ui/imgui/Application.h"

#include <filesystem>
#include <memory>
#include <string>

namespace tsd::ui::imgui {
struct LayerTree;
struct Log;
struct ObjectEditor;
struct TransferFunctionEditor;
struct Viewport;
} // namespace tsd::ui::imgui

namespace tsd::scivis_studio {

struct AddDatasetDialog;
struct CameraRigEditor;
struct ConfirmDiscardDialog;
struct DatasetEditor;
struct ProjectLocationDialog;
struct ProjectWindow;
struct ShotEditor;

class Application : public tsd::ui::imgui::Application
{
 public:
  Application(int argc = 0, const char **argv = nullptr);
  ~Application() override;

  ProjectContext &projectContext();
  const ProjectContext &projectContext() const;

  void showAddDatasetDialog();
  void showProjectLocationDialogForNew();
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
    OpenProject
  };

  bool saveProject();
  bool saveProjectAs(const std::filesystem::path &directory);
  bool openProject(const std::filesystem::path &directory);
  void newProject();
  void closeProject();
  void saveDefaultLayoutFile() const;
  void saveWindowSettings(tsd::core::DataNode &node);
  void loadWindowSettings(tsd::core::DataNode &node);
  std::string saveLayout() const;
  void loadLayout(const std::string &layout);
  void requestDirtyAction(PendingDirtyAction action);
  void continueDirtyAction();

  ProjectContext m_projectContext;
  std::filesystem::path m_initialProjectDirectory;
  PendingDirtyAction m_pendingDirtyAction{PendingDirtyAction::None};

  tsd::ui::imgui::Viewport *m_viewport{nullptr};
  tsd::ui::imgui::LayerTree *m_layerTree{nullptr};
  tsd::ui::imgui::TransferFunctionEditor *m_transferFunctionEditor{nullptr};

  std::unique_ptr<ProjectLocationDialog> m_projectLocationDialog;
  std::unique_ptr<ConfirmDiscardDialog> m_confirmDiscardDialog;
  std::unique_ptr<AddDatasetDialog> m_addDatasetDialog;
};

} // namespace tsd::scivis_studio
