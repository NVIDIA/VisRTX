// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Application.h"

#include "DefaultLayout.h"
#include "RenderShot.h"
#include "modals/AddDatasetDialog.h"
#include "modals/ConfirmDiscardDialog.h"
#include "modals/ProjectLocationDialog.h"
#include "windows/CameraRigEditor.h"
#include "windows/DatasetEditor.h"
#include "windows/ProjectWindow.h"
#include "windows/ShotEditor.h"

#include "tsd/core/Logging.hpp"
#include "tsd/ui/imgui/windows/LayerTree.h"
#include "tsd/ui/imgui/windows/Log.h"
#include "tsd/ui/imgui/windows/ObjectEditor.h"
#include "tsd/ui/imgui/windows/TransferFunctionEditor.h"
#include "tsd/ui/imgui/windows/Viewport.h"

#include "imgui.h"

#include <cstdlib>
#include <cstdio>

namespace tsd::scivis_studio {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

Application::Application(int argc, const char **argv)
    : TSDApplication(argc, argv), m_projectContext(appContext())
{
  if (!appContext()->commandLine.stateFile.empty())
    m_initialProjectDirectory = appContext()->commandLine.stateFile;
  appContext()->commandLine.stateFile.clear();
  appContext()->commandLine.loadedFromStateFile = false;
}

Application::~Application() = default;

ProjectContext &Application::projectContext()
{
  return m_projectContext;
}

const ProjectContext &Application::projectContext() const
{
  return m_projectContext;
}

anari_viewer::WindowArray Application::setupWindows()
{
  auto windows = TSDApplication::setupWindows();

  auto *ctx = appContext();
  m_viewport = new tsd_ui::Viewport(this, &ctx->view.manipulator, "Viewport");
  auto *projectWindow = new ProjectWindow(this, &m_projectContext);
  auto *datasetEditor = new DatasetEditor(this, &m_projectContext);
  auto *shotEditor = new ShotEditor(
      this, &m_projectContext, [this]() { renderActiveShot(); });
  auto *cameraRigEditor = new CameraRigEditor(this, &m_projectContext);
  auto *objectEditor = new tsd_ui::ObjectEditor(this);
  m_layerTree = new tsd_ui::LayerTree(this);
  m_transferFunctionEditor = new tsd_ui::TransferFunctionEditor(this);
  auto *log = new tsd_ui::Log(this);

  windows.emplace_back(projectWindow);
  windows.emplace_back(datasetEditor);
  windows.emplace_back(shotEditor);
  windows.emplace_back(cameraRigEditor);
  windows.emplace_back(m_viewport);
  windows.emplace_back(objectEditor);
  windows.emplace_back(m_layerTree);
  windows.emplace_back(m_transferFunctionEditor);
  windows.emplace_back(log);

  setWindowArray(windows);

  m_layerTree->hide();
  m_transferFunctionEditor->hide();

  m_projectLocationDialog = std::make_unique<ProjectLocationDialog>(this);
  m_confirmDiscardDialog = std::make_unique<ConfirmDiscardDialog>(this);
  m_addDatasetDialog =
      std::make_unique<AddDatasetDialog>(this, &m_projectContext);

  if (!m_initialProjectDirectory.empty()) {
    if (!openProject(m_initialProjectDirectory))
      m_projectContext.createUnsavedProject();
  } else
    m_projectContext.createUnsavedProject();

  if (m_viewport)
    m_viewport->setLibraryToDefault();

  return windows;
}

void Application::teardown()
{
  TSDApplication::teardown();
}

void Application::saveWindowSettings(tsd::core::DataNode &node)
{
  node.reset();
  for (auto *w : m_windows)
    w->saveSettings(node[w->name()]);
}

void Application::loadWindowSettings(tsd::core::DataNode &node)
{
  for (auto *w : m_windows)
    w->loadSettings(node[w->name()]);
}

std::string Application::saveLayout() const
{
  return ImGui::SaveIniSettingsToMemory();
}

void Application::loadLayout(const std::string &layout)
{
  if (!layout.empty())
    ImGui::LoadIniSettingsFromMemory(layout.c_str());
}

bool Application::saveProject()
{
  auto &project = m_projectContext.project();
  if (!project.isSaved()) {
    showProjectLocationDialogForSaveAs();
    return false;
  }

  return saveProjectAs(project.projectDirectory);
}

bool Application::saveProjectAs(const std::filesystem::path &directory)
{
  tsd::core::DataTree scratch;
  auto &root = scratch.root();
  saveWindowSettings(root["windows"]);
  saveApplicationSettings(root);

  std::string error;
  const bool ok = m_projectContext.saveProject(
      directory, root.child("windows"), saveLayout(), root.child("settings"), &error);
  if (!ok)
    tsd::core::logError("[SciVisStudio] Save failed: %s", error.c_str());
  return ok;
}

bool Application::openProject(const std::filesystem::path &directory)
{
  tsd::core::DataTree scratch;
  std::string layout;
  std::string error;
  const bool ok = m_projectContext.openProject(
      directory, &scratch.root()["windows"], &layout, &scratch.root()["settings"], &error);
  if (!ok) {
    tsd::core::logError("[SciVisStudio] Open failed: %s", error.c_str());
    return false;
  }

  loadWindowSettings(scratch.root()["windows"]);
  loadLayout(layout);
  loadApplicationSettings(scratch.root());
  return true;
}

void Application::newProject()
{
  m_projectContext.createUnsavedProject();
}

void Application::closeProject()
{
  m_projectContext.createUnsavedProject();
}

void Application::requestDirtyAction(PendingDirtyAction action)
{
  if (!m_projectContext.project().dirty) {
    m_pendingDirtyAction = action;
    continueDirtyAction();
    return;
  }

  m_pendingDirtyAction = action;
  m_confirmDiscardDialog->configure(
      [this]() { continueDirtyAction(); },
      [this]() { m_pendingDirtyAction = PendingDirtyAction::None; });
  m_confirmDiscardDialog->show();
}

void Application::continueDirtyAction()
{
  const auto action = m_pendingDirtyAction;
  m_pendingDirtyAction = PendingDirtyAction::None;

  if (action == PendingDirtyAction::NewProject)
    showProjectLocationDialogForNew();
  else if (action == PendingDirtyAction::OpenProject)
    showProjectLocationDialogForOpen();
}

void Application::showAddDatasetDialog()
{
  m_addDatasetDialog->show();
}

void Application::showProjectLocationDialogForNew()
{
  m_projectLocationDialog->configure(ProjectLocationMode::NewProject,
      [this](const std::filesystem::path &directory) {
        newProject();
        saveProjectAs(directory);
      });
  m_projectLocationDialog->show();
}

void Application::showProjectLocationDialogForOpen()
{
  m_projectLocationDialog->configure(ProjectLocationMode::OpenProject,
      [this](const std::filesystem::path &directory) { openProject(directory); });
  m_projectLocationDialog->show();
}

void Application::showProjectLocationDialogForSaveAs()
{
  m_projectLocationDialog->configure(ProjectLocationMode::SaveProjectAs,
      [this](const std::filesystem::path &directory) { saveProjectAs(directory); });
  m_projectLocationDialog->show();
}

void Application::renderActiveShot()
{
  if (!m_projectContext.project().isSaved()) {
    tsd::core::logWarning(
        "[SciVisStudio] Save the project before rendering a shot");
    showProjectLocationDialogForSaveAs();
    return;
  }

  showTaskModal(
      [this]() {
        RenderShotProgress progress;
        renderActiveShotToFrames(m_projectContext, &progress);
      },
      "Rendering Active Shot...");
}

void Application::tickShotPlayback(float deltaTime)
{
  auto *shot = activeShot(m_projectContext.project());
  if (!shot || !shot->playing || shot->fps <= 0.f)
    return;

  m_playbackAccumulator += deltaTime;
  const float frameDuration = 1.f / shot->fps;
  if (m_playbackAccumulator < frameDuration)
    return;

  int steps = static_cast<int>(m_playbackAccumulator / frameDuration);
  m_playbackAccumulator -= steps * frameDuration;
  while (steps-- > 0 && shot->playing) {
    ++shot->currentFrame;
    if (shot->currentFrame >= shot->frameCount) {
      if (shot->loop)
        shot->currentFrame = 0;
      else {
        shot->currentFrame = std::max(0, shot->frameCount - 1);
        shot->playing = false;
      }
    }
  }

  m_projectContext.applyActiveShot();
}

void Application::uiFrameStart()
{
  const ImGuiIO &io = ImGui::GetIO();
  tickShotPlayback(io.DeltaTime);

  if (ImGui::BeginMainMenuBar()) {
    uiMainMenuBar();
    ImGui::EndMainMenuBar();
  }

  bool modalActive = false;
  if (m_taskModal && m_taskModal->visible()) {
    m_taskModal->renderUI();
    modalActive = true;
  }

  if (m_projectLocationDialog && m_projectLocationDialog->visible()) {
    m_projectLocationDialog->renderUI();
    modalActive = true;
  }

  if (m_confirmDiscardDialog && m_confirmDiscardDialog->visible()) {
    m_confirmDiscardDialog->renderUI();
    modalActive = true;
  }

  if (m_addDatasetDialog && m_addDatasetDialog->visible()) {
    m_addDatasetDialog->renderUI();
    modalActive = true;
  }

  if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space)) {
    if (auto *shot = activeShot(m_projectContext.project())) {
      shot->playing = !shot->playing;
      m_playbackAccumulator = 0.f;
    }
  }

  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    saveProject();

  if (!modalActive && ImGui::IsKeyChordPressed(ImGuiKey_Escape))
    appContext()->clearSelected();
}

void Application::uiMainMenuBar()
{
  if (ImGui::BeginMenu("Project")) {
    if (ImGui::MenuItem("New Project..."))
      requestDirtyAction(PendingDirtyAction::NewProject);
    if (ImGui::MenuItem("Open Project..."))
      requestDirtyAction(PendingDirtyAction::OpenProject);
    if (ImGui::MenuItem("Save Project", "Ctrl+S"))
      saveProject();
    if (ImGui::MenuItem("Save Project As..."))
      showProjectLocationDialogForSaveAs();
    if (ImGui::MenuItem("Close Project"))
      requestDirtyAction(PendingDirtyAction::NewProject);
    ImGui::Separator();
    if (ImGui::MenuItem("Quit"))
      std::exit(0);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Studio")) {
    if (ImGui::MenuItem("Add Dataset..."))
      showAddDatasetDialog();
    if (ImGui::MenuItem("Add Shot"))
      m_projectContext.addShot();
    if (ImGui::MenuItem("Render Active Shot..."))
      renderActiveShot();
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("View")) {
    for (auto *w : m_windows) {
      ImGui::PushID(w);
      ImGui::Checkbox(w->name(), w->visiblePtr());
      ImGui::PopID();
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Print Layout"))
      std::printf("%s\n", ImGui::SaveIniSettingsToMemory());
    if (ImGui::MenuItem("Reset Layout"))
      ImGui::LoadIniSettingsFromMemory(getDefaultLayout());
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Tools")) {
    ImGui::TextDisabled("No phase-one tools");
    ImGui::EndMenu();
  }
}

const char *Application::getDefaultLayout() const
{
  return DEFAULT_LAYOUT;
}

} // namespace tsd::scivis_studio
