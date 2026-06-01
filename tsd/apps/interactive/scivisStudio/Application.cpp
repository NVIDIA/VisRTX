// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Application.h"

#include "DefaultLayout.h"
#include "RenderShot.h"
#include "modals/AddFileAnimationDatasetDialog.h"
#include "modals/AddStaticDatasetDialog.h"
#include "modals/ProjectLocationDialog.h"
#include "windows/CameraRigEditor.h"
#include "windows/DatasetEditor.h"
#include "windows/LightRigEditor.h"
#include "windows/ProjectWindow.h"
#include "windows/ShotEditor.h"

#include "tsd/core/Logging.hpp"
#include "tsd/ui/imgui/windows/LayerTree.h"
#include "tsd/ui/imgui/windows/Log.h"
#include "tsd/ui/imgui/windows/ObjectEditor.h"
#include "tsd/ui/imgui/windows/TransferFunctionEditor.h"
#include "tsd/ui/imgui/windows/Viewport.h"

#include "imgui.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <system_error>

namespace tsd::scivis_studio {


using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

namespace {

constexpr std::size_t MAX_RECENT_PROJECTS = 10;

std::filesystem::path studioConfigDirectory()
{
#ifdef _WIN32
  if (const char *appData = std::getenv("APPDATA"); appData != nullptr)
    return std::filesystem::path(appData) / "tsd" / "studio";
#else
  if (const char *home = std::getenv("HOME"); home != nullptr)
    return std::filesystem::path(home) / ".config" / "tsd" / "studio";
#endif

  return std::filesystem::path("studio");
}

std::filesystem::path normalizedAbsolutePath(const std::filesystem::path &path)
{
  std::error_code ec;
  auto absolute = std::filesystem::absolute(path, ec);
  if (ec)
    absolute = path;
  return absolute.lexically_normal();
}

bool pathsReferToSameProject(
    const std::filesystem::path &a, const std::filesystem::path &b)
{
  std::error_code ec;
  if (std::filesystem::exists(a, ec) && !ec && std::filesystem::exists(b, ec)
      && !ec) {
    const bool same = std::filesystem::equivalent(a, b, ec);
    if (!ec && same)
      return true;
  }

  return normalizedAbsolutePath(a) == normalizedAbsolutePath(b);
}

bool renderConfirmationModal(ConfirmationModalState &modal)
{
  if (!modal.visible)
    return false;

  ImGuiIO &io = ImGui::GetIO();
  ImGui::SetNextWindowPos(
      ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
      ImGuiCond_Always,
      ImVec2(0.5f, 0.5f));

  ImGui::OpenPopup(modal.title.c_str());
  if (ImGui::BeginPopupModal(modal.title.c_str(),
          &modal.visible,
          ImGuiWindowFlags_AlwaysAutoResize)) {
    if (modal.minWidth > 0.f)
      ImGui::Dummy(ImVec2(modal.minWidth, 0.f));
    if (!modal.message.empty())
      ImGui::TextUnformatted(modal.message.c_str());
    ImGui::Spacing();

    if (ImGui::Button(modal.cancelLabel.c_str())) {
      modal.visible = false;
      if (modal.onCancel)
        modal.onCancel();
    }

    ImGui::SameLine();

    if (ImGui::Button(modal.confirmLabel.c_str())) {
      modal.visible = false;
      if (modal.onConfirm)
        modal.onConfirm();
    }

    ImGui::EndPopup();
  }

  return modal.visible;
}

} // namespace

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

tsd::ui::imgui::WindowArray Application::setupWindows()
{
  auto windows = TSDApplication::setupWindows();
  loadRecentProjects();

  auto *ctx = appContext();
  m_viewport = new tsd_ui::Viewport(this, &ctx->view.manipulator, "Viewport");
  auto *projectWindow = new ProjectWindow(this, &m_projectContext);
  auto *datasetEditor = new DatasetEditor(this, &m_projectContext);
  auto *lightRigEditor = new LightRigEditor(this, &m_projectContext);
  auto *shotEditor =
      new ShotEditor(this, &m_projectContext, [this]() { renderActiveShot(); });
  auto *cameraRigEditor = new CameraRigEditor(this, &m_projectContext);
  auto *objectEditor = new tsd_ui::ObjectEditor(this);
  m_layerTree = new tsd_ui::LayerTree(this);
  m_transferFunctionEditor = new tsd_ui::TransferFunctionEditor(this);
  auto *log = new tsd_ui::Log(this);

  windows.emplace_back(projectWindow);
  windows.emplace_back(datasetEditor);
  windows.emplace_back(lightRigEditor);
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
  m_addStaticDatasetDialog =
      std::make_unique<AddStaticDatasetDialog>(this, &m_projectContext);
  m_addFileAnimationDatasetDialog =
      std::make_unique<AddFileAnimationDatasetDialog>(this, &m_projectContext);

  if (!m_initialProjectDirectory.empty()) {
    if (!openProject(m_initialProjectDirectory)) {
      m_projectContext.createUnsavedProject();
      m_keepBlankProjectCleanAfterViewportSetup = true;
      m_viewport->setLibraryToDefault();
    }
  } else {
    m_projectContext.createUnsavedProject();
    m_keepBlankProjectCleanAfterViewportSetup = true;
    m_viewport->setLibraryToDefault();
  }

  ctx->tsd.sceneLoadComplete = true;

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
  const bool ok = m_projectContext.saveProject(directory,
      root.child("windows"),
      saveLayout(),
      root.child("settings"),
      &error);
  if (!ok)
    tsd::core::logError("[SciVisStudio] Save failed: %s", error.c_str());
  else
    addRecentProject(directory);
  return ok;
}

bool Application::openProject(const std::filesystem::path &directory)
{
  if (m_viewport)
    m_viewport->releaseSceneReferences();

  tsd::core::DataTree scratch;
  std::string layout;
  std::string error;
  const bool ok = m_projectContext.openProject(directory,
      &scratch.root()["windows"],
      &layout,
      &scratch.root()["settings"],
      &error);
  if (!ok) {
    tsd::core::logError("[SciVisStudio] Open failed: %s", error.c_str());
    if (m_viewport)
      m_viewport->setLibraryToDefault();
    return false;
  }

  loadWindowSettings(scratch.root()["windows"]);
  loadLayout(layout);
  loadApplicationSettings(scratch.root());
  addRecentProject(directory);
  return true;
}

void Application::newProject()
{
  if (m_viewport)
    m_viewport->releaseSceneReferences();

  m_projectContext.createUnsavedProject();
  m_keepBlankProjectCleanAfterViewportSetup = true;
  if (m_viewport)
    m_viewport->setLibraryToDefault();
}

void Application::requestDirtyAction(PendingDirtyAction action)
{
  if (!m_projectContext.project().dirty) {
    m_pendingDirtyAction = action;
    continueDirtyAction();
    return;
  }

  m_pendingDirtyAction = action;
  m_confirmationModal.visible = true;
  m_confirmationModal.title = "Discard Unsaved Changes";
  m_confirmationModal.message = "The current project has unsaved changes.";
  m_confirmationModal.cancelLabel = "Cancel";
  m_confirmationModal.confirmLabel = "Discard and Continue";
  m_confirmationModal.minWidth = 0.f;
  m_confirmationModal.onCancel = [this]() {
    m_pendingDirtyAction = PendingDirtyAction::None;
    m_pendingProjectDirectory.clear();
  };
  m_confirmationModal.onConfirm = [this]() { continueDirtyAction(); };
}

void Application::requestOpenRecentProject(
    const std::filesystem::path &directory)
{
  m_pendingProjectDirectory = directory;
  requestDirtyAction(PendingDirtyAction::OpenRecentProject);
}

void Application::continueDirtyAction()
{
  const auto action = m_pendingDirtyAction;
  m_pendingDirtyAction = PendingDirtyAction::None;

  if (action == PendingDirtyAction::NewProject)
    newProject();
  else if (action == PendingDirtyAction::OpenProject)
    showProjectLocationDialogForOpen();
  else if (action == PendingDirtyAction::OpenRecentProject) {
    const auto directory = m_pendingProjectDirectory;
    m_pendingProjectDirectory.clear();
    if (!openProject(directory))
      removeRecentProject(directory);
  }
}

std::filesystem::path Application::recentProjectsFile() const
{
  return studioConfigDirectory() / "recent_projects.txt";
}

void Application::loadRecentProjects()
{
  m_recentProjects.clear();

  const auto filename = recentProjectsFile();
  if (!std::filesystem::exists(filename))
    return;

  std::ifstream in(filename);
  if (!in) {
    tsd::core::logWarning(
        "[SciVisStudio] Failed to read recent projects file '%s'",
        filename.string().c_str());
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;

    const auto path = normalizedAbsolutePath(line);
    const auto duplicate = std::any_of(m_recentProjects.begin(),
        m_recentProjects.end(),
        [&](const auto &entry) {
          return pathsReferToSameProject(entry, path);
        });
    if (!duplicate)
      m_recentProjects.push_back(path);

    if (m_recentProjects.size() >= MAX_RECENT_PROJECTS)
      break;
  }
}

void Application::saveRecentProjects() const
{
  const auto filename = recentProjectsFile();
  const auto directory = filename.parent_path();

  try {
    if (!directory.empty())
      std::filesystem::create_directories(directory);
  } catch (const std::exception &e) {
    tsd::core::logWarning(
        "[SciVisStudio] Failed to create recent projects directory '%s': %s",
        directory.string().c_str(),
        e.what());
    return;
  }

  std::ofstream out(filename, std::ios::trunc);
  if (!out) {
    tsd::core::logWarning(
        "[SciVisStudio] Failed to write recent projects file '%s'",
        filename.string().c_str());
    return;
  }

  for (const auto &project : m_recentProjects)
    out << project.string() << '\n';
}

void Application::addRecentProject(const std::filesystem::path &directory)
{
  const auto path = normalizedAbsolutePath(directory);

  m_recentProjects.erase(std::remove_if(m_recentProjects.begin(),
                             m_recentProjects.end(),
                             [&](const auto &entry) {
                               return pathsReferToSameProject(entry, path);
                             }),
      m_recentProjects.end());
  m_recentProjects.insert(m_recentProjects.begin(), path);

  if (m_recentProjects.size() > MAX_RECENT_PROJECTS)
    m_recentProjects.resize(MAX_RECENT_PROJECTS);

  saveRecentProjects();
}

void Application::removeRecentProject(const std::filesystem::path &directory)
{
  const auto oldSize = m_recentProjects.size();
  m_recentProjects.erase(std::remove_if(m_recentProjects.begin(),
                             m_recentProjects.end(),
                             [&](const auto &entry) {
                               return pathsReferToSameProject(entry, directory);
                             }),
      m_recentProjects.end());

  if (m_recentProjects.size() != oldSize)
    saveRecentProjects();
}

void Application::clearRecentProjects()
{
  m_recentProjects.clear();
  saveRecentProjects();
}

void Application::uiRecentProjectsMenu()
{
  std::filesystem::path selectedProject;
  bool clearRequested = false;

  if (ImGui::BeginMenu("Recent")) {
    if (m_recentProjects.empty())
      ImGui::TextDisabled("No recent projects");

    for (const auto &project : m_recentProjects) {
      const auto label = project.string();
      if (ImGui::MenuItem(label.c_str()))
        selectedProject = project;
    }

    ImGui::Separator();
    if (ImGui::MenuItem("Clear Recent"))
      clearRequested = true;
    ImGui::EndMenu();
  }

  if (!selectedProject.empty())
    requestOpenRecentProject(selectedProject);
  else if (clearRequested)
    clearRecentProjects();
}

void Application::showAddStaticDatasetDialog()
{
  m_addStaticDatasetDialog->show();
}

void Application::showAddFileAnimationDatasetDialog()
{
  m_addFileAnimationDatasetDialog->show();
}

void Application::showProjectLocationDialogForOpen()
{
  m_projectLocationDialog->configure(ProjectLocationMode::OpenProject,
      [this](
          const std::filesystem::path &directory) { openProject(directory); });
  m_projectLocationDialog->show();
}

void Application::showProjectLocationDialogForSaveAs()
{
  m_projectLocationDialog->configure(ProjectLocationMode::SaveProjectAs,
      [this](const std::filesystem::path &directory) {
        saveProjectAs(directory);
      });
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

  if (m_viewport) {
    m_viewport->setRenderingEnabled(false);
    m_viewportRenderingDisabledForShotRender = true;
  }

  showTaskModalWithCancel(
      [this](const std::atomic_bool &cancelRequested) {
        RenderShotProgress progress;
        progress.onFrame = [&](int, int) { return !cancelRequested.load(); };
        renderActiveShotToFrames(m_projectContext, &progress);
      },
      "Rendering Active Shot...");

  if (!m_taskModal && m_viewportRenderingDisabledForShotRender) {
    if (m_viewport)
      m_viewport->setRenderingEnabled(true);
    m_viewportRenderingDisabledForShotRender = false;
  }
}

void Application::saveDefaultLayoutFile() const
{
  const std::string layout = ImGui::SaveIniSettingsToMemory();

  std::ofstream out(DEFAULT_LAYOUT_FILE, std::ios::binary | std::ios::trunc);
  if (!out) {
    tsd::core::logError(
        "[SciVisStudio] Failed to open default layout file '%s'",
        DEFAULT_LAYOUT_FILE);
    return;
  }

  out << layout;
  if (layout.empty() || layout.back() != '\n')
    out << '\n';

  if (!out) {
    tsd::core::logError(
        "[SciVisStudio] Failed to write default layout file '%s'",
        DEFAULT_LAYOUT_FILE);
    return;
  }

  tsd::core::logStatus(
      "[SciVisStudio] Saved default layout file '%s'", DEFAULT_LAYOUT_FILE);
}

void Application::uiFrameStart()
{
  const ImGuiIO &io = ImGui::GetIO();
  auto &animMgr = appContext()->tsd.animationMgr;
  animMgr.tick(io.DeltaTime);
  if (auto *shot = project::activeShot(m_projectContext.project()))
    shot->playing = animMgr.isPlaying();

  if (ImGui::BeginMainMenuBar()) {
    uiMainMenuBar();
    ImGui::EndMainMenuBar();
  }

  bool modalActive = false;
  if (m_taskModal && m_taskModal->visible()) {
    m_taskModal->renderUI();
    modalActive = true;
  }

  if (m_viewportRenderingDisabledForShotRender
      && (!m_taskModal || !m_taskModal->visible())) {
    if (m_viewport)
      m_viewport->setRenderingEnabled(true);
    m_viewportRenderingDisabledForShotRender = false;
  }

  if (m_projectLocationDialog && m_projectLocationDialog->visible()) {
    m_projectLocationDialog->renderUI();
    modalActive = true;
  }

  modalActive = renderConfirmationModal(m_confirmationModal) || modalActive;

  if (m_addStaticDatasetDialog && m_addStaticDatasetDialog->visible()) {
    m_addStaticDatasetDialog->renderUI();
    modalActive = true;
  }

  if (m_addFileAnimationDatasetDialog
      && m_addFileAnimationDatasetDialog->visible()) {
    m_addFileAnimationDatasetDialog->renderUI();
    modalActive = true;
  }

  if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space)) {
    if (auto *shot = project::activeShot(m_projectContext.project())) {
      animMgr.togglePlay();
      shot->playing = animMgr.isPlaying();
    }
  }

  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    saveProject();

  if (!modalActive && ImGui::IsKeyChordPressed(ImGuiKey_Escape))
    appContext()->clearSelected();

  if (m_keepBlankProjectCleanAfterViewportSetup
      && (!m_taskModal || !m_taskModal->visible())) {
    m_projectContext.project().markClean();
    m_keepBlankProjectCleanAfterViewportSetup = false;
  }
}

void Application::uiMainMenuBar()
{
  if (ImGui::BeginMenu("Project")) {
    if (ImGui::MenuItem("New"))
      requestDirtyAction(PendingDirtyAction::NewProject);
    if (ImGui::MenuItem("Open ..."))
      requestDirtyAction(PendingDirtyAction::OpenProject);
    ImGui::Separator();
    uiRecentProjectsMenu();
    ImGui::Separator();
    if (ImGui::MenuItem("Save", "Ctrl+S"))
      saveProject();
    if (ImGui::MenuItem("Save As..."))
      showProjectLocationDialogForSaveAs();
    ImGui::Separator();
    if (ImGui::MenuItem("Quit"))
      std::exit(0);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Studio")) {
    if (ImGui::BeginMenu("Add Dataset")) {
      if (ImGui::MenuItem("Static..."))
        showAddStaticDatasetDialog();
      if (ImGui::MenuItem("File Animation..."))
        showAddFileAnimationDatasetDialog();
      ImGui::EndMenu();
    }
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
    if (ImGui::MenuItem("Save Default Layout File")) {
      m_confirmationModal.visible = true;
      m_confirmationModal.title = "Update Default Layout";
      m_confirmationModal.message = "Are you sure?";
      m_confirmationModal.cancelLabel = "No";
      m_confirmationModal.confirmLabel = "Yes";
      m_confirmationModal.minWidth = 700.f;
      m_confirmationModal.onCancel = {};
      m_confirmationModal.onConfirm = [this]() { saveDefaultLayoutFile(); };
    }
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
