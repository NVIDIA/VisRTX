// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/Timer.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// tsd_rendering
#include "tsd/rendering/view/Manipulator.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_font.h"
#include "tsd/ui/imgui/windows/Window.h"
// anari_viewer
#include "anari_viewer/ui_anari.h"
// SDL
#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_video.h>

namespace tsd::ui::imgui {

Application::Application(int argc, const char **argv)
{
  std::vector<std::string> args(argv, argv + argc);
  auto *core = appCore();
  parseCommandLine(args);
  core->parseCommandLine(args);
  if (!core->commandLine.stateFile.empty())
    m_filenameToLoadNextFrame = core->commandLine.stateFile;
}

Application::~Application() = default;

tsd::app::Core *Application::appCore()
{
  return &m_core;
}

UIConfig *Application::uiConfig()
{
  return &m_uiConfig;
}

CommandLineOptions *Application::commandLineOptions()
{
  return &m_commandLine;
}
#ifdef TSD_USE_LUA
ExtensionManager *Application::extensionManager() const
{
  return m_extensionManager.get();
}
#endif

void Application::getFilenameFromDialog(std::string &filenameOut, bool save)
{
  auto fileDialogCb =
      [](void *userdata, const char *const *filelist, int filter) {
        std::string &out = *(std::string *)userdata;
        if (!filelist) {
          tsd::core::logError("SDL DIALOG ERROR: %s\n", SDL_GetError());
          return;
        }

        if (*filelist)
          out = *filelist;
      };

  if (save) {
    SDL_ShowSaveFileDialog(
        fileDialogCb, &filenameOut, this->sdlWindow(), nullptr, 0, nullptr);
  } else {
    SDL_ShowOpenFileDialog(fileDialogCb,
        &filenameOut,
        this->sdlWindow(),
        nullptr,
        0,
        nullptr,
        false);
  }
}

void Application::showImportFileDialog()
{
  m_fileDialog->show();
}

void Application::showExportNanoVDBFileDialog()
{
  m_exportNanoVDBFileDialog->show();
}

void Application::parseCommandLine(std::vector<std::string> &args)
{
  for (int i = 1; i < args.size(); i++) {
    std::string arg = std::move(args[i]); // consume arguments
    if (arg.empty())
      continue;
    if (arg == "--noDefaultLayout")
      m_commandLine.useDefaultLayout = false;
    else if (arg == "--secondaryView" || arg == "-sv")
      m_commandLine.secondaryViewportLibrary = std::move(args[++i]);
    else if (arg == "--noDefaultRenderer")
      m_commandLine.useDefaultRenderer = false;
    else
      args[i] = std::move(arg); // move back unconsumed arguments
  }
}

anari_viewer::WindowArray Application::setupWindows()
{
  anari_viewer::ui::init();

  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  auto *font = io.Fonts->AddFontFromMemoryCompressedTTF(
      tsd_font_compressed_data, tsd_font_compressed_size, 20.f);
  io.Fonts->ConfigData[0].FontDataOwnedByAtlas = false;
  io.FontDefault = font;

  if (commandLineOptions()->useDefaultLayout)
    ImGui::LoadIniSettingsFromMemory(getDefaultLayout());

  m_appSettingsDialog = std::make_unique<AppSettingsDialog>(this);
  m_taskModal = std::make_unique<BlockingTaskModal>(this);
  m_offlineRenderModal = std::make_unique<OfflineRenderModal>(this);
  m_fileDialog = std::make_unique<ImportFileDialog>(this);
  m_exportNanoVDBFileDialog = std::make_unique<ExportNanoVDBFileDialog>(this);

  m_applicationName = SDL_GetWindowTitle(sdlWindow());
  updateWindowTitle();

  m_appSettingsDialog->applySettings();

  SDL_SetRenderVSync(sdlRenderer(), 1);

#ifdef TSD_USE_LUA
  m_extensionManager = std::make_unique<ExtensionManager>();
  m_extensionManager->initialize(appCore());
#endif

  return {};
}

void Application::uiFrameStart()
{
  if (!m_filenameToSaveNextFrame.empty()) {
    saveApplicationState(m_filenameToSaveNextFrame.c_str());
    m_filenameToSaveNextFrame.clear();
  } else if (!m_filenameToLoadNextFrame.empty()) {
    loadStateForNextFrame();
  }

  // Main Menu //

  if (ImGui::BeginMainMenuBar()) {
    uiMainMenuBar();
    ImGui::EndMainMenuBar();
  }

  // Modals //

  bool modalActive = false;
  if (m_appSettingsDialog->visible()) {
    m_appSettingsDialog->renderUI();
    modalActive = true;
  }

  if (m_taskModal->visible()) {
    m_taskModal->renderUI();
    modalActive = true;
  }

  if (m_offlineRenderModal->visible()) {
    m_offlineRenderModal->renderUI();
    modalActive = true;
  }

  if (m_fileDialog->visible()) {
    m_fileDialog->renderUI();
    modalActive = true;
  }

  // Handle app shortcuts //
  if (m_exportNanoVDBFileDialog->visible()) {
    m_exportNanoVDBFileDialog->renderUI();
    modalActive = true;
  }

  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S))
    this->getFilenameFromDialog(m_filenameToSaveNextFrame, true);
  else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Alt | ImGuiKey_S))
    doSave("state.tsd");
  else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    doSave();

  if (!modalActive && ImGui::IsKeyChordPressed(ImGuiKey_Escape))
    m_core.clearSelected();
}

void Application::teardown()
{
  teardownUsdDevice();
  anari_viewer::ui::shutdown();
}

void Application::uiMainMenuBar()
{
  if (ImGui::BeginMenu("File")) {
    if (ImGui::MenuItem("Load"))
      this->getFilenameFromDialog(m_filenameToLoadNextFrame);

    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Load session from a .tsd file");

    ImGui::Separator();

    if (ImGui::MenuItem("Save", "CTRL+S"))
      doSave();

    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Save session to a .tsd file");

    if (ImGui::MenuItem("Save As...", "CTRL+SHIFT+S"))
      this->getFilenameFromDialog(m_filenameToSaveNextFrame, true);

    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Save session to a chosen file name");

    if (ImGui::MenuItem("Quick Save", "CTRL+ALT+S"))
      doSave("state.tsd");

    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Save sesson to 'state.tsd' in the local directory");

    ImGui::Separator();

    if (ImGui::MenuItem("Render Animation Sequence..."))
      m_offlineRenderModal->start();

    ImGui::Separator();

    if (ImGui::MenuItem("Export as USD...")) {
      io::export_SceneToUSD(m_core.tsd.scene,
          "scene.usda",
          m_core.view.pathSettings.framesPerSecond);
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Quit", "CTRL+Q"))
      std::exit(0);

    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Edit")) {
    if (ImGui::MenuItem("Settings"))
      m_appSettingsDialog->show();

    ImGui::Separator();

    if (ImGui::BeginMenu("UI Layout")) {
      if (ImGui::MenuItem("Print"))
        printf("%s\n", ImGui::SaveIniSettingsToMemory());

      ImGui::Separator();

      if (ImGui::MenuItem("Reset"))
        ImGui::LoadIniSettingsFromMemory(getDefaultLayout());

      ImGui::EndMenu();
    }

    ImGui::Separator();

    if (ImGui::BeginMenu("Scene")) {
      if (ImGui::MenuItem("Cleanup Only"))
        m_core.tsd.scene.removeUnusedObjects();

      if (ImGui::MenuItem("Defragment Only"))
        m_core.tsd.scene.defragmentObjectStorage();

      if (ImGui::MenuItem("Cleanup + Defragment")) {
        m_core.tsd.scene.removeUnusedObjects();
        m_core.tsd.scene.defragmentObjectStorage();
      }

      ImGui::EndMenu();
    }

    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Tools")) {
    if (ImGui::BeginMenu("OpenUSD Device")) {
      if (usdDeviceIsSetup()) {
        if (ImGui::MenuItem("Disable"))
          teardownUsdDevice();
      } else {
        if (ImGui::MenuItem("Enable"))
          setupUsdDevice();
      }
      ImGui::Separator();
      ImGui::BeginDisabled(!usdDeviceIsSetup());
      if (ImGui::MenuItem("Sync"))
        syncUsdScene();
      ImGui::EndDisabled();
      ImGui::EndMenu();
    }

    ImGui::Separator();

    if (ImGui::BeginMenu("TSD Device")) {
      if (tsdDeviceIsSetup()) {
        if (ImGui::MenuItem("Disable"))
          teardownTsdDevice();
      } else {
        if (ImGui::MenuItem("Enable"))
          setupTsdDevice();
      }
      ImGui::Separator();
      ImGui::BeginDisabled(!tsdDeviceIsSetup());
      if (ImGui::MenuItem("Sync"))
        syncTsdScene();
      ImGui::EndDisabled();
      ImGui::EndMenu();
    }

    ImGui::EndMenu();
  }

#ifdef TSD_USE_LUA
  renderLuaMenu();
#endif

  if (ImGui::BeginMenu("View")) {
    for (auto &w : m_windows) {
      ImGui::PushID(&w);
      ImGui::Checkbox(w->name(), w->visiblePtr());
      ImGui::PopID();
    }

    ImGui::EndMenu();
  }
}

void Application::doSave(const std::string &name)
{
  if (!name.empty())
    m_filenameToSaveNextFrame = name;
  else if (m_currentSessionFilename.empty())
    this->getFilenameFromDialog(m_filenameToSaveNextFrame, true);
  else
    m_filenameToSaveNextFrame = m_currentSessionFilename;
}

void Application::saveApplicationState(const char *_filename)
{
  std::string f_str = _filename;

  auto doSave = [&, filename = f_str]() {
    tsd::core::logStatus("clearing old settings tree...");

    auto &core = *appCore();
    auto &root = m_settings.root();
    root.reset();

    // Window state
    auto &windows = root["windows"];
    for (auto *w : m_windows)
      w->saveSettings(root["windows"][w->name()]);

    // ImGui window layout
    tsd::core::logStatus("serializing UI state...");
    root["layout"] = ImGui::SaveIniSettingsToMemory();

    // ANARIDeviceManager settings
    core.anari.saveSettings(root["ANARIDeviceManager"]);

    // Offline rendering settings
    auto &offlineSettings = root["offlineRendering"];
    core.offline.saveSettings(offlineSettings);

    // General application settings
    auto &settings = root["settings"];
    settings["logVerbose"] = core.logVerbose();
    settings["logEchoOutput"] = core.logEchoOutput();
    settings["fontScale"] = m_uiConfig.fontScale;
    settings["uiRounding"] = m_uiConfig.rounding;

    // Camera poses
    auto &cameraPoses = root["cameraPoses"];
    for (auto &p : core.view.poses)
      tsd::io::cameraPoseToNode(p, cameraPoses.append());

    // Serialize TSD context
    tsd::core::logStatus("serializing TSD context...");
    root["context"].reset();
    tsd::io::save_Scene(core.tsd.scene, root["context"], false);

    // Save to file
    tsd::core::logStatus("writing state file '%s'...", filename.c_str());
    m_settings.save(filename.c_str());

    // Clear out context tree
    root["context"].reset();

    tsd::core::logStatus("...state saved to '%s'", filename.c_str());

    m_currentSessionFilename = filename;
    updateWindowTitle();
  };

  showTaskModal(doSave, "Please Wait: Saving Session...");
}

void Application::loadApplicationState(const char *filename)
{
  // Load from file
  if (!m_settings.load(filename)) {
    tsd::core::logError("failed to load state from '%s'", filename);
    return;
  }

  auto &core = *appCore();
  auto &root = m_settings.root();

  // Window state
  auto &windows = root["windows"];
  for (auto *w : m_windows)
    w->loadSettings(windows[w->name()]);

  // ImGui window layout
  if (auto *c = root.child("layout"); c != nullptr)
    ImGui::LoadIniSettingsFromMemory(c->getValueAs<std::string>().c_str());

  // ANARIDeviceManager settings
  if (auto *c = root.child("ANARIDeviceManager"); c != nullptr)
    core.anari.loadSettings(*c);

  // Offline rendering settings
  auto &offlineSettings = root["offlineRendering"];
  core.offline.loadSettings(offlineSettings);

  // General application settings
  if (auto *c = root.child("settings"); c != nullptr) {
    auto &settings = *c;

    bool logVerbose = core.logVerbose();
    settings["logVerbose"].getValue(ANARI_BOOL, &logVerbose);
    core.setLogVerbose(logVerbose);
    bool logEchoOutput = core.logEchoOutput();
    settings["logEchoOutput"].getValue(ANARI_BOOL, &logEchoOutput);
    core.setLogEchoOutput(logEchoOutput);

    settings["fontScale"].getValue(ANARI_FLOAT32, &m_uiConfig.fontScale);
    settings["uiRounding"].getValue(ANARI_FLOAT32, &m_uiConfig.rounding);
  }

  core.view.poses.clear();
  if (auto *c = root.child("cameraPoses"); c != nullptr) {
    c->foreach_child([&](auto &p) {
      tsd::rendering::CameraPose pose;
      tsd::io::nodeToCameraPose(p, pose);
      core.view.poses.push_back(std::move(pose));
    });
  }

  // TSD context from app state file, or context-only file
  if (auto *c = root.child("context"); c != nullptr)
    tsd::io::load_Scene(core.tsd.scene, *c);
  else
    tsd::io::load_Scene(core.tsd.scene, root);

  // Clear out context tree
  root["context"].reset();

  m_appSettingsDialog->applySettings();

  tsd::core::logStatus("...loaded state from '%s'", filename);

  m_currentSessionFilename = filename;
  updateWindowTitle();
}

void Application::loadStateForNextFrame()
{
  if (m_filenameToLoadNextFrame.empty())
    return;
  m_core.clearSelected();
  loadApplicationState(m_filenameToLoadNextFrame.c_str());
  m_filenameToLoadNextFrame.clear();
}

void Application::setupUsdDevice()
{
  if (usdDeviceIsSetup())
    return;

  auto d = m_usdDevice.device;

  if (d == nullptr) {
    d = m_core.anari.loadDevice("usd");
    if (!d) {
      tsd::core::logWarning("USD device failed to load");
      return;
    }
    anari::retain(d, d);
    m_usdDevice.device = d;
  }

  m_usdDevice.renderIndex =
      m_core.anari.acquireRenderIndex(m_core.tsd.scene, d);
  m_usdDevice.frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(
      d, m_usdDevice.frame, "world", m_usdDevice.renderIndex->world());
}

bool Application::usdDeviceIsSetup() const
{
  return m_usdDevice.device != nullptr && m_usdDevice.renderIndex != nullptr;
}

void Application::syncUsdScene()
{
  tsd::core::logStatus("synchronizing USD ANARI device scene...");
  if (!usdDeviceIsSetup()) {
    tsd::core::logWarning("USD device not setup -- cannot sync scene");
    return;
  }
  tsd::core::Timer timer;
  timer.start();
  anari::render(m_usdDevice.device, m_usdDevice.frame);
  anari::wait(m_usdDevice.device, m_usdDevice.frame);
  timer.end();
  tsd::core::logStatus("...sync complete (%.2f ms)", timer.milliseconds());
}

void Application::teardownUsdDevice()
{
  if (!usdDeviceIsSetup())
    return;
  tsd::core::logStatus("tearing down USD device...");
  auto d = m_usdDevice.device;
  m_core.anari.releaseRenderIndex(d);
  anari::release(d, m_usdDevice.frame);
  anari::release(d, d);
  m_usdDevice.device = nullptr;
  m_usdDevice.renderIndex = nullptr;
}

void Application::setupTsdDevice()
{
  if (tsdDeviceIsSetup())
    return;

  auto d = m_tsdDevice.device;

  if (d == nullptr) {
    d = m_core.anari.loadDevice("tsd");
    if (!d) {
      tsd::core::logWarning("TSD device failed to load");
      return;
    }
    anari::retain(d, d);
    m_tsdDevice.device = d;
  }

  m_tsdDevice.renderIndex =
      m_core.anari.acquireRenderIndex(m_core.tsd.scene, d);
  m_tsdDevice.frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(
      d, m_tsdDevice.frame, "world", m_tsdDevice.renderIndex->world());

  syncTsdScene();
}

bool Application::tsdDeviceIsSetup() const
{
  return m_tsdDevice.device != nullptr && m_tsdDevice.renderIndex != nullptr;
}

void Application::syncTsdScene()
{
  tsd::core::logStatus("synchronizing TSD ANARI device scene...");
  if (!tsdDeviceIsSetup()) {
    tsd::core::logWarning("TSD device not setup -- cannot sync scene");
    return;
  }
  tsd::core::Timer timer;
  timer.start();
  anari::render(m_tsdDevice.device, m_tsdDevice.frame);
  anari::wait(m_tsdDevice.device, m_tsdDevice.frame);
  timer.end();
  tsd::core::logStatus("...sync complete (%.2f ms)", timer.milliseconds());
}

void Application::teardownTsdDevice()
{
  if (!tsdDeviceIsSetup())
    return;
  tsd::core::logStatus("tearing down TSD device...");
  auto d = m_tsdDevice.device;
  m_core.anari.releaseRenderIndex(d);
  anari::release(d, m_tsdDevice.frame);
  anari::release(d, d);
  m_tsdDevice.device = nullptr;
  m_tsdDevice.renderIndex = nullptr;
}

void Application::setWindowArray(const anari_viewer::WindowArray &wa)
{
  for (auto &w : wa)
    m_windows.push_back((Window *)w.get());
}

void Application::updateWindowTitle()
{
  auto *w = this->sdlWindow();
  if (!w)
    return;

  std::string title = m_applicationName + " | ";

  title += m_currentSessionFilename.empty() ? std::string("{new session}")
                                            : m_currentSessionFilename;

  SDL_SetWindowTitle(w, title.c_str());
}

#ifdef TSD_USE_LUA
void Application::renderLuaMenu()
{
  if (ImGui::BeginMenu("Lua")) {
    const auto &tree = m_extensionManager->getMenuTree();
    renderActionMenu(tree);

    if (!tree.empty())
      ImGui::Separator();

    if (ImGui::MenuItem("Reload Script"))
      m_extensionManager->refresh();

    ImGui::EndMenu();
  }
}

void Application::renderActionMenu(const std::vector<ActionMenuNode> &entries)
{
  for (const auto &entry : entries) {
    if (entry.isSeparator) {
      ImGui::Separator();
    } else if (entry.isFolder) {
      if (ImGui::BeginMenu(entry.name.c_str())) {
        renderActionMenu(entry.children);
        ImGui::EndMenu();
      }
    } else {
      if (ImGui::MenuItem(entry.name.c_str())) {
        showTaskModal(
            [this, actionIndex = entry.actionIndex]() {
              m_extensionManager->executeAction(actionIndex);
            },
            "Executing Action...");
      }

      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", entry.name.c_str());
    }
  }
}
#endif

} // namespace tsd::ui::imgui
