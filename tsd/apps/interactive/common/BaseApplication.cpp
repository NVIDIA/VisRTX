// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BaseApplication.h"
// tsd
#include "tsd_font.h"
#include "tsd_ui.h"
// anari_viewer
#include "anari_viewer/ui_anari.h"

namespace tsd_viewer {

BaseApplication::BaseApplication(int argc, const char **argv)
{
  appCore()->parseCommandLine(argc, argv);
}

BaseApplication::~BaseApplication() = default;

AppCore *BaseApplication::appCore()
{
  return &m_core;
}

anari_viewer::WindowArray BaseApplication::setupWindows()
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

  m_appSettings = std::make_unique<tsd_viewer::AppSettings>();
  m_fileDialog = std::make_unique<tsd_viewer::ImportFileDialog>(appCore());

  m_core.windows.importDialog = m_fileDialog.get();

  return {};
}

void BaseApplication::uiFrameStart()
{
  // Handle app shortcuts //

  ImGuiIO &io = ImGui::GetIO();
  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    saveContext();

  // Main Menu //

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Save", "CTRL+S"))
        saveContext();

      ImGui::Separator();

      if (ImGui::MenuItem("Print UI Layout")) {
        const char *info = ImGui::SaveIniSettingsToMemory();
        printf("%s\n", info);
      }

      ImGui::Separator();

      if (ImGui::MenuItem("Quit", "SHIFT+Q"))
        std::exit(0);

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Edit")) {
      if (ImGui::MenuItem("Settings"))
        m_appSettings->show();
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

    if (m_appSettings->visible())
      m_appSettings->renderUI();

    if (m_fileDialog->visible())
      m_fileDialog->renderUI();
  }
}

void BaseApplication::teardown()
{
  teardownUsdDevice();
  anari_viewer::ui::shutdown();
}

void BaseApplication::saveContext()
{
  tsd::save_Context(appCore()->tsd.ctx, "state.tsd");
  tsd::logStatus("context saved to 'state.tsd'");
}

void BaseApplication::setupUsdDevice()
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

bool BaseApplication::usdDeviceSetup() const
{
  return m_usd.device != nullptr && m_usd.renderIndex != nullptr;
}

void BaseApplication::syncUsdScene()
{
  if (!usdDeviceSetup()) {
    tsd::logWarning("USD device not setup -- cannot sync scene");
    return;
  }
  anari::render(m_usd.device, m_usd.frame);
  anari::wait(m_usd.device, m_usd.frame);
}

void BaseApplication::teardownUsdDevice()
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

void BaseApplication::setWindowArray(const anari_viewer::WindowArray &wa)
{
  for (auto &w : wa)
    m_windows.push_back(w.get());
}

} // namespace tsd_viewer