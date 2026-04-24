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
// std
#include <cstdlib>

namespace tsd::ui::imgui {

Application::Application(int argc, const char **argv)
{
  std::vector<std::string> args(argv, argv + argc);
  auto *ctx = appContext();
  parseCommandLine(args);
  ctx->parseCommandLine(args);
  if (!ctx->commandLine.stateFile.empty())
    m_filenameToLoadNextFrame = ctx->commandLine.stateFile;
}

Application::~Application() = default;

tsd::app::Context *Application::appContext()
{
  return &m_ctx;
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

void Application::saveDefaultApplicationSettings()
{
  saveGlobalApplicationSettings();
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

  auto *window = sdlWindow();
  SDL_MaximizeWindow(window);
  m_uiConfig.fontScale = SDL_GetWindowDisplayScale(window);

  setupImGuiStyle();

  if (commandLineOptions()->useDefaultLayout)
    ImGui::LoadIniSettingsFromMemory(getDefaultLayout());

  m_appSettingsDialog = std::make_unique<AppSettingsDialog>(this);
  m_taskModal = std::make_unique<BlockingTaskModal>(this);
  m_offlineRenderModal = std::make_unique<OfflineRenderModal>(this);
  m_fileDialog = std::make_unique<ImportFileDialog>(this);
  m_exportNanoVDBFileDialog = std::make_unique<ExportNanoVDBFileDialog>(this);
  m_vorticityDialog = std::make_unique<VorticityDialog>(this);
  m_cuttingPlaneDialog = std::make_unique<CuttingPlaneDialog>(this);

  m_applicationName = SDL_GetWindowTitle(sdlWindow());
  updateWindowTitle();

  loadGlobalApplicationSettings();
  m_appSettingsDialog->applySettings();

  SDL_SetRenderVSync(sdlRenderer(), 1);

  m_extensionManager = std::make_unique<ExtensionManager>();
  m_extensionManager->initialize(appContext());

  return {};
}

void Application::uiFrameStart()
{
  const ImGuiIO &io = ImGui::GetIO();

  m_ctx.tsd.animationMgr.tick(ImGui::GetIO().DeltaTime);

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

  if (m_vorticityDialog->visible()) {
    m_vorticityDialog->renderUI();
    modalActive = true;
  }

  if (m_cuttingPlaneDialog->visible()) {
    m_cuttingPlaneDialog->renderUI();
    modalActive = true;
  }

  if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space))
    m_ctx.tsd.animationMgr.togglePlay();

  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S))
    this->getFilenameFromDialog(m_filenameToSaveNextFrame, true);
  else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Alt | ImGuiKey_S))
    doSave("state.tsd");
  else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    doSave();

  if (ImGui::IsKeyPressed(ImGuiKey_F1, false))
    printf("%s\n", ImGui::SaveIniSettingsToMemory());

  if (!modalActive && ImGui::IsKeyChordPressed(ImGuiKey_Escape))
    m_ctx.clearSelected();
}

void Application::teardown()
{
  teardownUsdDevice();
  teardownTsdDevice();
  appContext()->anari.releaseAllDevices();
  anari_viewer::ui::shutdown();
}

void Application::setupImGuiStyle()
{
  ImGuiStyle &style = ImGui::GetStyle();

  style.Alpha = 1.0f;
  style.DisabledAlpha = 0.6f;
  style.WindowPadding = ImVec2(8.0f, 8.0f);
  style.WindowRounding = 4.0f;
  style.WindowBorderSize = 1.0f;
  style.WindowMinSize = ImVec2(32.0f, 32.0f);
  style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
  style.WindowMenuButtonPosition = ImGuiDir_None;
  style.ChildRounding = 4.0f;
  style.ChildBorderSize = 1.0f;
  style.PopupRounding = 4.0f;
  style.PopupBorderSize = 1.0f;
  style.FramePadding = ImVec2(8.0f, 4.0f);
  style.FrameRounding = 4.0f;
  style.FrameBorderSize = 1.0f;
  style.ItemSpacing = ImVec2(8.0f, 4.0f);
  style.ItemInnerSpacing = ImVec2(8.0f, 4.0f);
  style.CellPadding = ImVec2(4.0f, 4.0f);
  style.IndentSpacing = 21.0f;
  style.ColumnsMinSpacing = 6.0f;
  style.ScrollbarSize = 20.0f;
  style.ScrollbarRounding = 4.0f;
  style.GrabMinSize = 10.0f;
  style.GrabRounding = 20.0f;
  style.TabRounding = 4.0f;
  style.TabBorderSize = 1.0f;
  style.TabMinWidthForCloseButton = 0.0f;
  style.ColorButtonPosition = ImGuiDir_Right;
  style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
  style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

  style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  style.Colors[ImGuiCol_TextDisabled] =
      ImVec4(0.49803922f, 0.49803922f, 0.49803922f, 1.0f);
  style.Colors[ImGuiCol_WindowBg] =
      ImVec4(0.11372549f, 0.11372549f, 0.11372549f, 1.0f);
  style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  style.Colors[ImGuiCol_PopupBg] =
      ImVec4(0.078431375f, 0.078431375f, 0.078431375f, 0.94f);
  style.Colors[ImGuiCol_Border] = ImVec4(1.0f, 1.0f, 1.0f, 0.16309011f);
  style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  style.Colors[ImGuiCol_FrameBg] =
      ImVec4(0.08627451f, 0.08627451f, 0.08627451f, 1.0f);
  style.Colors[ImGuiCol_FrameBgHovered] =
      ImVec4(0.15294118f, 0.15294118f, 0.15294118f, 1.0f);
  style.Colors[ImGuiCol_FrameBgActive] =
      ImVec4(0.1882353f, 0.1882353f, 0.1882353f, 1.0f);
  style.Colors[ImGuiCol_TitleBg] =
      ImVec4(0.11372549f, 0.11372549f, 0.11372549f, 1.0f);
  style.Colors[ImGuiCol_TitleBgActive] =
      ImVec4(0.105882354f, 0.105882354f, 0.105882354f, 1.0f);
  style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0f, 0.0f, 0.0f, 0.51f);
  style.Colors[ImGuiCol_MenuBarBg] =
      ImVec4(0.11372549f, 0.11372549f, 0.11372549f, 1.0f);
  style.Colors[ImGuiCol_ScrollbarBg] =
      ImVec4(0.019607844f, 0.019607844f, 0.019607844f, 0.53f);
  style.Colors[ImGuiCol_ScrollbarGrab] =
      ImVec4(0.30980393f, 0.30980393f, 0.30980393f, 1.0f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered] =
      ImVec4(0.40784314f, 0.40784314f, 0.40784314f, 1.0f);
  style.Colors[ImGuiCol_ScrollbarGrabActive] =
      ImVec4(0.50980395f, 0.50980395f, 0.50980395f, 1.0f);
  style.Colors[ImGuiCol_CheckMark] = ImVec4(0.4627451f, 0.7254902f, 0.0f, 1.0f);
  style.Colors[ImGuiCol_SliderGrab] =
      ImVec4(0.8784314f, 0.8784314f, 0.8784314f, 1.0f);
  style.Colors[ImGuiCol_SliderGrabActive] =
      ImVec4(0.8784314f, 0.8784314f, 0.8784314f, 1.0f);
  style.Colors[ImGuiCol_Button] =
      ImVec4(0.14901812f, 0.14901961f, 0.14901817f, 1.0f);
  style.Colors[ImGuiCol_ButtonHovered] =
      ImVec4(0.30386257f, 0.47639483f, 0.0f, 1.0f);
  style.Colors[ImGuiCol_ButtonActive] =
      ImVec4(0.4627451f, 0.7254902f, 0.0f, 1.0f);
  style.Colors[ImGuiCol_Header] =
      ImVec4(0.9764706f, 0.9764706f, 0.9764706f, 0.30980393f);
  style.Colors[ImGuiCol_HeaderHovered] =
      ImVec4(0.9764706f, 0.9764706f, 0.9764706f, 0.49803922f);
  style.Colors[ImGuiCol_HeaderActive] =
      ImVec4(0.9764706f, 0.9764706f, 0.9764706f, 1.0f);
  style.Colors[ImGuiCol_Separator] =
      ImVec4(0.42745098f, 0.42745098f, 0.49803922f, 0.5f);
  style.Colors[ImGuiCol_SeparatorHovered] =
      ImVec4(0.7490196f, 0.7490196f, 0.7490196f, 0.78039217f);
  style.Colors[ImGuiCol_SeparatorActive] =
      ImVec4(0.7490196f, 0.7490196f, 0.7490196f, 1.0f);
  style.Colors[ImGuiCol_ResizeGrip] =
      ImVec4(0.9764706f, 0.9764706f, 0.9764706f, 0.2f);
  style.Colors[ImGuiCol_ResizeGripHovered] =
      ImVec4(0.9372549f, 0.9372549f, 0.9372549f, 0.67058825f);
  style.Colors[ImGuiCol_ResizeGripActive] =
      ImVec4(0.9764706f, 0.9764706f, 0.9764706f, 0.9490196f);
  style.Colors[ImGuiCol_Tab] =
      ImVec4(0.22352941f, 0.22352941f, 0.22352941f, 0.8627451f);
  style.Colors[ImGuiCol_TabHovered] =
      ImVec4(0.32156864f, 0.32156864f, 0.32156864f, 0.8f);
  style.Colors[ImGuiCol_TabActive] =
      ImVec4(0.27450982f, 0.27450982f, 0.27450982f, 1.0f);
  style.Colors[ImGuiCol_TabUnfocused] =
      ImVec4(0.14509805f, 0.14509805f, 0.14509805f, 0.972549f);
  style.Colors[ImGuiCol_TabUnfocusedActive] =
      ImVec4(0.42352942f, 0.42352942f, 0.42352942f, 1.0f);
  style.Colors[ImGuiCol_PlotLines] =
      ImVec4(0.60784316f, 0.60784316f, 0.60784316f, 1.0f);
  style.Colors[ImGuiCol_PlotLinesHovered] =
      ImVec4(1.0f, 0.42745098f, 0.34901962f, 1.0f);
  style.Colors[ImGuiCol_PlotHistogram] =
      ImVec4(0.8980392f, 0.69803923f, 0.0f, 1.0f);
  style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.0f, 0.6f, 0.0f, 1.0f);
  style.Colors[ImGuiCol_TableHeaderBg] =
      ImVec4(0.1882353f, 0.1882353f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_TableBorderStrong] =
      ImVec4(0.30980393f, 0.30980393f, 0.34901962f, 1.0f);
  style.Colors[ImGuiCol_TableBorderLight] =
      ImVec4(0.22745098f, 0.22745098f, 0.24705882f, 1.0f);
  style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.06f);
  style.Colors[ImGuiCol_TextSelectedBg] =
      ImVec4(0.25882354f, 0.5882353f, 0.9764706f, 0.35f);
  style.Colors[ImGuiCol_DragDropTarget] = ImVec4(1.0f, 1.0f, 0.0f, 0.9f);
  style.Colors[ImGuiCol_NavHighlight] =
      ImVec4(0.4627451f, 0.7254902f, 0.0f, 1.0f);
  style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 0.7f);
  style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.8f, 0.8f, 0.8f, 0.2f);
  style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.8f, 0.8f, 0.8f, 0.35f);
}

void Application::uiMainMenuBar()
{
  uiMainMenuBar_File();
  uiMainMenuBar_Edit();
  uiMainMenuBar_Tools();
  uiMainMenuBar_Lua();
  uiMainMenuBar_View();
}

void Application::uiMainMenuBar_File()
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
      io::export_SceneToUSD(m_ctx.tsd.scene,
          "scene.usda",
          m_ctx.view.pathSettings.framesPerSecond);
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Quit", "CTRL+Q"))
      std::exit(0);

    ImGui::EndMenu();
  }
}

void Application::uiMainMenuBar_Edit()
{
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
      if (ImGui::MenuItem("Cleanup Unused Objects"))
        m_ctx.tsd.scene.removeUnusedObjects(false);

      if (ImGui::MenuItem("Cleanup Unused Objects + Renderers"))
        m_ctx.tsd.scene.removeUnusedObjects(true);

      if (ImGui::MenuItem("Defragment Scene Storage"))
        m_ctx.tsd.scene.defragmentObjectStorage();

      if (ImGui::MenuItem("Cleanup Unused Objects + Defragment")) {
        m_ctx.tsd.scene.removeUnusedObjects(false);
        m_ctx.tsd.scene.defragmentObjectStorage();
      }

      if (ImGui::MenuItem("Cleanup Unused + Defragment All")) {
        m_ctx.tsd.scene.removeUnusedObjects(true);
        m_ctx.tsd.scene.defragmentObjectStorage();
      }

      ImGui::EndMenu();
    }

    ImGui::EndMenu();
  }
}

void Application::uiMainMenuBar_Tools()
{
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

    if (ImGui::MenuItem("Flow Analysis"))
      m_vorticityDialog->show();

    if (ImGui::MenuItem("Cutting Plane"))
      m_cuttingPlaneDialog->show();

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
}
void Application::uiMainMenuBar_Lua()
{
#ifdef TSD_USE_LUA
  if (ImGui::BeginMenu("Lua")) {
    const auto &tree = m_extensionManager->getMenuTree();
    uiActionMenu(tree);

    if (!tree.empty())
      ImGui::Separator();

    if (ImGui::MenuItem("Reload Script"))
      m_extensionManager->refresh();

    ImGui::EndMenu();
  }
#endif
}

void Application::uiMainMenuBar_View()
{
  if (ImGui::BeginMenu("View")) {
    for (auto &w : m_windows) {
      ImGui::PushID(&w);
      ImGui::Checkbox(w->name(), w->visiblePtr());
      ImGui::PopID();
    }

    ImGui::EndMenu();
  }
}

void Application::uiActionMenu(const std::vector<ActionMenuNode> &entries)
{
  for (const auto &entry : entries) {
    if (entry.isSeparator) {
      ImGui::Separator();
    } else if (entry.isFolder) {
      if (ImGui::BeginMenu(entry.name.c_str())) {
        uiActionMenu(entry.children);
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

    auto &ctx = *appContext();
    auto &root = m_settings.root();
    root.reset();

    // Window state
    auto &windows = root["windows"];
    for (auto *w : m_windows)
      w->saveSettings(root["windows"][w->name()]);

    // ImGui window layout
    tsd::core::logStatus("serializing UI state...");
    root["layout"] = ImGui::SaveIniSettingsToMemory();

    // Offline rendering settings
    auto &offlineSettings = root["offlineRendering"];
    ctx.offline.saveSettings(offlineSettings);

    // General application settings
    auto &settings = root["settings"];
    settings["logVerbose"] = ctx.logVerbose();
    settings["logEchoOutput"] = ctx.logEchoOutput();
    saveApplicationSettings(root);

    // Camera poses
    auto &cameraPoses = root["cameraPoses"];
    for (auto &p : ctx.view.poses)
      tsd::io::cameraPoseToNode(p, cameraPoses.append());

    // Serialize TSD context
    tsd::core::logStatus("serializing TSD context...");
    root["context"].reset();
    tsd::io::save_Scene(
        ctx.tsd.scene, root["context"], false, &ctx.tsd.animationMgr);

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

  auto &ctx = *appContext();
  auto &root = m_settings.root();

  // TSD context from app state file, or context-only file
  if (auto *c = root.child("context"); c != nullptr)
    tsd::io::load_Scene(ctx.tsd.scene, *c, &ctx.tsd.animationMgr);
  else
    tsd::io::load_Scene(ctx.tsd.scene, root, &ctx.tsd.animationMgr);

  // Clear out context tree
  root["context"].reset();

  // Window state
  auto &windows = root["windows"];
  for (auto *w : m_windows)
    w->loadSettings(windows[w->name()]);

  // ImGui window layout
  if (auto *c = root.child("layout"); c != nullptr)
    ImGui::LoadIniSettingsFromMemory(c->getValueAs<std::string>().c_str());

  // ANARIDeviceManager settings
  loadApplicationSettings(root);

  // Offline rendering settings
  auto &offlineSettings = root["offlineRendering"];
  ctx.offline.loadSettings(offlineSettings);

  // General application settings
  if (auto *c = root.child("settings"); c != nullptr) {
    auto &settings = *c;

    bool logVerbose = ctx.logVerbose();
    settings["logVerbose"].getValue(ANARI_BOOL, &logVerbose);
    ctx.setLogVerbose(logVerbose);
    bool logEchoOutput = ctx.logEchoOutput();
    settings["logEchoOutput"].getValue(ANARI_BOOL, &logEchoOutput);
    ctx.setLogEchoOutput(logEchoOutput);
  }

  ctx.view.poses.clear();
  if (auto *c = root.child("cameraPoses"); c != nullptr) {
    c->foreach_child([&](auto &p) {
      tsd::rendering::CameraPose pose;
      tsd::io::nodeToCameraPose(p, pose);
      ctx.view.poses.push_back(std::move(pose));
    });
  }

  m_appSettingsDialog->applySettings();

  tsd::core::logStatus("...loaded state from '%s'", filename);

  m_currentSessionFilename = filename;
  updateWindowTitle();
}

void Application::saveApplicationSettings(tsd::core::DataNode &root)
{
  auto &ctx = *appContext();

  ctx.anari.saveSettings(root["ANARIDeviceManager"]);

  auto &settings = root["settings"];
  settings["fontScale"] = m_uiConfig.fontScale;
  settings["uiRounding"] = m_uiConfig.rounding;
}

void Application::loadApplicationSettings(tsd::core::DataNode &root)
{
  auto &ctx = *appContext();

  if (auto *c = root.child("ANARIDeviceManager"); c != nullptr)
    ctx.anari.loadSettings(*c);

  if (auto *c = root.child("settings"); c != nullptr) {
    auto &settings = *c;
    settings["fontScale"].getValue(ANARI_FLOAT32, &m_uiConfig.fontScale);
    settings["uiRounding"].getValue(ANARI_FLOAT32, &m_uiConfig.rounding);
  }
}

void Application::saveGlobalApplicationSettings()
{
  const auto filename = globalApplicationSettingsFile();
  const auto directory = filename.parent_path();

  try {
    if (!directory.empty())
      std::filesystem::create_directories(directory);
  } catch (const std::exception &e) {
    tsd::core::logError(
        "[Application] Failed to create config directory '%s': %s",
        directory.string().c_str(),
        e.what());
    return;
  }

  auto &root = m_settings.root();
  root.reset();
  saveApplicationSettings(root);

  if (!m_settings.save(filename.string().c_str())) {
    tsd::core::logError("[Application] Failed to save defaults to '%s'",
        filename.string().c_str());
    return;
  }

  tsd::core::logStatus(
      "...saved application defaults to '%s'", filename.string().c_str());
}

void Application::loadGlobalApplicationSettings()
{
  const auto filename = globalApplicationSettingsFile();

  if (!std::filesystem::exists(filename))
    return;

  auto &root = m_settings.root();
  root.reset();
  if (!m_settings.load(filename.string().c_str())) {
    tsd::core::logWarning("[Application] Failed to load defaults from '%s'",
        filename.string().c_str());
    return;
  }

  loadApplicationSettings(root);
}

std::filesystem::path Application::globalApplicationSettingsFile() const
{
#ifdef _WIN32
  if (const char *appData = std::getenv("APPDATA"); appData != nullptr)
    return std::filesystem::path(appData) / "tsd" / "appSettings.tsd";
#else
  if (const char *home = std::getenv("HOME"); home != nullptr)
    return std::filesystem::path(home) / ".config" / "tsd" / "appSettings.tsd";
#endif

  return std::filesystem::path("appSettings.tsd");
}

void Application::loadStateForNextFrame()
{
  if (m_filenameToLoadNextFrame.empty())
    return;
  m_ctx.clearSelected();
  loadApplicationState(m_filenameToLoadNextFrame.c_str());
  m_filenameToLoadNextFrame.clear();
}

void Application::setupUsdDevice()
{
  if (usdDeviceIsSetup())
    return;

  auto d = m_usdDevice.device;

  if (d == nullptr) {
    d = m_ctx.anari.loadDevice("usd");
    if (!d) {
      tsd::core::logWarning("USD device failed to load");
      return;
    }
    anari::retain(d, d);
    m_usdDevice.device = d;
  }

  m_usdDevice.renderIndex =
      m_ctx.anari.acquireRenderIndex(m_ctx.tsd.scene, "usd", d);
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
  m_ctx.anari.releaseRenderIndex(m_ctx.tsd.scene, d);
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
    d = m_ctx.anari.loadDevice("tsd");
    if (!d) {
      tsd::core::logWarning("TSD device failed to load");
      return;
    }
    anari::retain(d, d);
    m_tsdDevice.device = d;
  }

  m_tsdDevice.renderIndex =
      m_ctx.anari.acquireRenderIndex(m_ctx.tsd.scene, "tsd", d);
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
  m_ctx.anari.releaseRenderIndex(m_ctx.tsd.scene, d);
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

} // namespace tsd::ui::imgui
