// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TSDApplication.h"
#include "windows/Window.h"
// tsd
#include "tsd_font.h"
#include "tsd_ui.h"
// anari_viewer
#include "anari_viewer/ui_anari.h"
#include "anari_viewer/windows/Window.h"
// SDL
#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_video.h>

namespace tsd_viewer {

TSDApplication::TSDApplication(int argc, const char **argv) : m_core(this)
{
  auto *core = appCore();
  core->parseCommandLine(argc, argv);

  if (core->commandLine.preloadDevices) {
    printf("[TSD] pre-loading all ANARI devices...");
    for (auto l : core->commandLine.libraryList)
      core->loadDevice(l);
    printf("done\n");
  }

  auto &filenames = m_core.commandLine.filenames;
  if (!filenames.empty() && filenames[0].first == ImporterType::NONE) {
    m_filenameToLoadNextFrame = filenames[0].second;
    filenames.clear();
    m_core.commandLine.loadedFromStateFile = true;
  }
}

TSDApplication::~TSDApplication() = default;

AppCore *TSDApplication::appCore()
{
  return &m_core;
}

anari_viewer::WindowArray TSDApplication::setupWindows()
{
  anari_viewer::ui::init();

  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = 1.f;
  io.IniFilename = nullptr;
  auto *font = io.Fonts->AddFontFromMemoryCompressedTTF(
      tsd_font_compressed_data, tsd_font_compressed_size, 20.f);
  io.Fonts->ConfigData[0].FontDataOwnedByAtlas = false;
  io.FontDefault = font;

  if (appCore()->commandLine.useDefaultLayout)
    ImGui::LoadIniSettingsFromMemory(getDefaultLayout());

  m_appSettingsDialog = std::make_unique<AppSettingsDialog>(appCore());
  m_fileDialog = std::make_unique<ImportFileDialog>(appCore());

  m_core.windows.importDialog = m_fileDialog.get();

  m_applicationName = SDL_GetWindowTitle(sdlWindow());
  updateWindowTitle();

  SDL_SetRenderVSync(sdlRenderer(), 1);

  return {};
}

void TSDApplication::uiFrameStart()
{
  if (!m_filenameToSaveNextFrame.empty()) {
    saveApplicationState(m_filenameToSaveNextFrame.c_str());
    m_currentSessionFilename = m_filenameToSaveNextFrame;
    m_filenameToSaveNextFrame.clear();
    updateWindowTitle();
  } else if (!m_filenameToLoadNextFrame.empty()) {
    m_core.clearSelected();
    loadApplicationState(m_filenameToLoadNextFrame.c_str());
    m_currentSessionFilename = m_filenameToLoadNextFrame;
    m_filenameToLoadNextFrame.clear();
    updateWindowTitle();
  }

  // Handle app shortcuts //

  auto doSave = [&]() {
    if (m_currentSessionFilename.empty())
      m_core.getFilenameFromDialog(m_filenameToSaveNextFrame, true);
    else
      m_filenameToSaveNextFrame = m_currentSessionFilename;
  };

  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    doSave();

  // Main Menu //

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Load"))
        m_core.getFilenameFromDialog(m_filenameToLoadNextFrame);

      ImGui::Separator();

      if (ImGui::MenuItem("Save", "CTRL+S"))
        doSave();

      if (ImGui::MenuItem("Save As..."))
        m_core.getFilenameFromDialog(m_filenameToSaveNextFrame, true);

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

      if (ImGui::BeginMenu("Context")) {
        if (ImGui::MenuItem("Cleanup Only"))
          m_core.tsd.ctx.removeUnusedObjects();

        if (ImGui::MenuItem("Defragment Only"))
          m_core.tsd.ctx.defragmentObjectStorage();

        if (ImGui::MenuItem("Cleanup + Defragment")) {
          m_core.tsd.ctx.removeUnusedObjects();
          m_core.tsd.ctx.defragmentObjectStorage();
        }

        ImGui::EndMenu();
      }

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      for (auto &w : m_windows) {
        ImGui::PushID(&w);
        ImGui::Checkbox(w->name(), w->visiblePtr());
        ImGui::PopID();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("USD")) {
      if (usdDeviceSetup()) {
        if (ImGui::MenuItem("Disable"))
          teardownUsdDevice();
        ImGui::Separator();
        if (ImGui::MenuItem("Sync"))
          syncUsdScene();
      } else {
        if (ImGui::MenuItem("Enable"))
          setupUsdDevice();
      }
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();

    // Modals //

    if (m_appSettingsDialog->visible())
      m_appSettingsDialog->renderUI();

    if (m_fileDialog->visible())
      m_fileDialog->renderUI();
  }
}

void TSDApplication::teardown()
{
  teardownUsdDevice();
  anari_viewer::ui::shutdown();
}

void TSDApplication::saveApplicationState(const char *filename)
{
  tsd::logStatus("serializing application state + context...");

  auto &core = *appCore();
  auto &root = m_settings.root();

  // Window state
  auto &windows = root["windows"];
  for (auto *w : m_windows)
    w->saveSettings(root["windows"][w->name()]);

  // ImGui window layout
  root["layout"] = ImGui::SaveIniSettingsToMemory();

  // General application settings
  auto &settings = root["settings"];
  settings["logVerbose"] = core.logging.verbose;
  settings["logEchoOutput"] = core.logging.echoOutput;
  settings["fontScale"] = core.windows.fontScale;

  // Serialize TSD context
  root["context"].reset();
  tsd::save_Context(core.tsd.ctx, root["context"]);

  // Save to file
  m_settings.save(filename);

  // Clear out context tree
  root["context"].reset();

  tsd::logStatus("...state saved to '%s'", filename);
}

void TSDApplication::loadApplicationState(const char *filename)
{
  // Load from file
  m_settings.load(filename);

  auto &core = *appCore();
  auto &root = m_settings.root();

  // Window state
  auto &windows = root["windows"];
  for (auto *w : m_windows)
    w->loadSettings(windows[w->name()]);

  // ImGui window layout
  if (auto *c = root.child("layout"); c != nullptr)
    ImGui::LoadIniSettingsFromMemory(c->getValueAs<std::string>().c_str());

  // General application settings
  if (auto *c = root.child("settings"); c != nullptr) {
    auto &settings = *c;
    settings["logVerbose"].getValue(ANARI_BOOL, &core.logging.verbose);
    settings["logEchoOutput"].getValue(ANARI_BOOL, &core.logging.echoOutput);
    settings["fontScale"].getValue(ANARI_FLOAT32, &core.windows.fontScale);
  }

  // TSD context from app state file, or context-only file
  if (auto *c = root.child("context"); c != nullptr)
    tsd::load_Context(core.tsd.ctx, *c);
  else
    tsd::load_Context(core.tsd.ctx, root);

  // Clear out context tree
  root["context"].reset();

  m_appSettingsDialog->applySettings();

  tsd::logStatus("...loaded state from '%s'", filename);
}

void TSDApplication::setupUsdDevice()
{
  if (usdDeviceSetup())
    return;

  auto d = m_usd.device;

  if (d == nullptr) {
    d = m_core.loadDevice("usd");
    if (!d) {
      tsd::logWarning("USD device failed to load");
      return;
    }
    anari::retain(d, d);
    m_usd.device = d;
  }

  m_usd.renderIndex = m_core.acquireRenderIndex(d);
  m_usd.frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, m_usd.frame, "world", m_usd.renderIndex->world());
}

bool TSDApplication::usdDeviceSetup() const
{
  return m_usd.device != nullptr && m_usd.renderIndex != nullptr;
}

void TSDApplication::syncUsdScene()
{
  if (!usdDeviceSetup()) {
    tsd::logWarning("USD device not setup -- cannot sync scene");
    return;
  }
  anari::render(m_usd.device, m_usd.frame);
  anari::wait(m_usd.device, m_usd.frame);
}

void TSDApplication::teardownUsdDevice()
{
  if (!usdDeviceSetup())
    return;
  auto d = m_usd.device;
  m_core.releaseRenderIndex(d);
  anari::release(d, m_usd.frame);
  anari::release(d, d);
  m_usd.device = nullptr;
  m_usd.renderIndex = nullptr;
}

void TSDApplication::setWindowArray(const anari_viewer::WindowArray &wa)
{
  for (auto &w : wa)
    m_windows.push_back((Window *)w.get());
}

void TSDApplication::updateWindowTitle()
{
  auto *w = m_core.application->sdlWindow();
  if (!w)
    return;

  std::string title = m_applicationName + " | ";

  title += m_currentSessionFilename.empty() ? std::string("{new session}")
                                            : m_currentSessionFilename;

  SDL_SetWindowTitle(w, title.c_str());
}

} // namespace tsd_viewer
