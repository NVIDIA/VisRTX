// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ExtensionManager.h"

#include "modals/AppSettingsDialog.h"
#include "modals/BlockingTaskModal.h"
#include "modals/CuttingPlaneDialog.h"
#include "modals/ExportNanoVDBFileDialog.h"
#include "modals/ImportFileDialog.h"
#include "modals/ObjectFileDialog.h"
#include "modals/OfflineRenderModal.h"
#include "modals/VorticityDialog.h"
// tsd_app
#include "tsd/app/Context.h"
// tsd_core
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/TaskQueue.hpp"
// SDL
#include <SDL3/SDL.h>
// std
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tsd::ui::imgui {

struct Window;
using WindowArray = std::vector<std::unique_ptr<Window>>;

struct UIConfig
{
  float fontScale{1.f};
  float rounding{4.f};
};

struct CommandLineOptions
{
  bool useDefaultLayout{true};
  bool useDefaultRenderer{true};
  std::string secondaryViewportLibrary;
};

enum class FileDialogMode
{
  OpenFile,
  SaveFile,
  OpenDirectory
};

class Application
{
 public:
  Application(int argc = 0, const char **argv = nullptr);
  virtual ~Application();

  SDL_Renderer *sdlRenderer();
  SDL_Window *sdlWindow();

  // Start the application run loop
  void run(int width, int height, const char *name);

  tsd::app::Context *appContext();
  UIConfig *uiConfig();
  CommandLineOptions *commandLineOptions();

  void getFilenameFromDialog(
      std::string &filenameOut,
      FileDialogMode mode = FileDialogMode::OpenFile);
  void getFilenameFromDialog(std::string &filenameOut, bool isSaveDialog);
  void getFilenamesFromDialog(std::vector<std::string> &filenamesOut);

  // Enqueue a task to be executed on a background thread
  template <class FUNCTION>
  tsd::core::Future enqueueTask(FUNCTION &&task);

  // Enqueue a task, then show a modal until task is complete
  template <class FUNCTION>
  void showTaskModal(FUNCTION &&f, const char *text = "Please Wait");

  // Enqueue a cancellable task, then show a modal until task is complete
  template <class FUNCTION>
  void showTaskModalWithCancel(FUNCTION &&f, const char *text = "Please Wait");
  void showImportFileDialog();
  void showExportNanoVDBFileDialog();
  void showImportObjectFileDialog(
      TSDObjectFileType fileType, tsd::scene::LayerNodeRef importRoot);
  void showExportObjectFileDialog(TSDObjectFileType fileType,
      anari::DataType objectType,
      size_t objectIndex);
  void showImportLayerSubtreeFileDialog(
      tsd::scene::LayerNodeRef destinationParent);
  void showExportLayerSubtreeFileDialog(tsd::scene::LayerNodeRef sourceRoot);
  void saveDefaultApplicationSettings();

  ExtensionManager *extensionManager() const;

  TSD_NOT_COPYABLE(Application)
  TSD_NOT_MOVEABLE(Application)

 protected:
  void parseCommandLine(std::vector<std::string> &args);
  bool getWindowSize(int &width, int &height) const;
  float getLastFrameLatency() const;

  // Internal API //

  virtual void setupImGuiStyle();

  virtual Uint32 sdlWindowFlags() const;
  virtual WindowArray setupWindows();

  virtual void mainLoopStart();
  virtual void mainLoopEnd();
  virtual void teardown();

  virtual void uiFrameStart();
  virtual void uiRenderStart();
  virtual void uiRenderEnd();
  virtual void uiFrameEnd();

  virtual void uiMainMenuBar();
  void uiMainMenuBar_File();
  void uiMainMenuBar_Edit();
  void uiMainMenuBar_Tools();
  void uiMainMenuBar_Lua();
  void uiMainMenuBar_View();

  void uiActionMenu(const std::vector<ActionMenuNode> &entries);

  void doSave(const std::string &name = "");

  void saveApplicationState(const char *filename = "state.tsd");
  void loadApplicationState(const char *filename = "state.tsd");
  virtual tsd::core::DataTreeMetadata applicationStateMetadata() const;
  virtual bool validateApplicationStateMetadata(
      const tsd::core::DataTreeMetadataReadResult &metadata,
      const tsd::core::DataNode &root,
      const char *filename) const;
  void saveApplicationSettings(tsd::core::DataNode &root);
  void loadApplicationSettings(tsd::core::DataNode &root);
  void saveGlobalApplicationSettings();
  void loadGlobalApplicationSettings();
  std::filesystem::path globalApplicationSettingsFile() const;

  void loadStateForNextFrame();

  void setupUsdDevice();
  bool usdDeviceIsSetup() const;
  void syncUsdScene();
  void teardownUsdDevice();

  void setupTsdDevice();
  bool tsdDeviceIsSetup() const;
  void syncTsdScene();
  void teardownTsdDevice();

  void setWindowArray(const WindowArray &wa);
  virtual const char *getDefaultLayout() const = 0;

  // Data //

  std::vector<Window *> m_windows;
  std::unique_ptr<AppSettingsDialog> m_appSettingsDialog;
  std::unique_ptr<BlockingTaskModal> m_taskModal;
  std::unique_ptr<OfflineRenderModal> m_offlineRenderModal;
  std::unique_ptr<ImportFileDialog> m_fileDialog;
  std::unique_ptr<ExportNanoVDBFileDialog> m_exportNanoVDBFileDialog;
  std::unique_ptr<ObjectFileDialog> m_objectFileDialog;
  std::unique_ptr<VorticityDialog> m_vorticityDialog;
  std::unique_ptr<CuttingPlaneDialog> m_cuttingPlaneDialog;

  tsd::core::DataTree m_settings;

  UIConfig m_uiConfig;
  CommandLineOptions m_commandLine;

 private:
  void mainLoop();
  void updateWindowTitle();

  // Data //

  tsd::app::Context m_ctx;

  struct AppImpl;
  std::unique_ptr<AppImpl> m_impl;

  tsd::core::TaskQueue m_jobs{10};

  std::string m_applicationName = "TSD";

  std::string m_currentSessionFilename;
  std::string m_filenameToSaveNextFrame;
  std::string m_filenameToLoadNextFrame;

  std::unique_ptr<ExtensionManager> m_extensionManager;

  struct UsdDeviceState
  {
    anari::Device device{nullptr};
    anari::Frame frame{nullptr};
    tsd::rendering::RenderIndex *renderIndex{nullptr};
  } m_usdDevice;

  struct TsdDeviceState
  {
    anari::Device device{nullptr};
    anari::Frame frame{nullptr};
    tsd::rendering::RenderIndex *renderIndex{nullptr};
  } m_tsdDevice;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <class FUNCTION>
inline tsd::core::Future Application::enqueueTask(FUNCTION &&task)
{
  return m_jobs.enqueue(std::forward<FUNCTION>(task));
}

template <class F>
inline void Application::showTaskModal(F &&f, const char *text)
{
  auto future = enqueueTask(std::forward<F>(f));

  if (!m_taskModal) {
    tsd::core::logWarning(
        "[Application] No task modal available to show, "
        "executing task without showing modal.");
    future.wait();
  } else {
    m_taskModal->activate(std::move(future), text);
  }
}

template <class F>
inline void Application::showTaskModalWithCancel(F &&f, const char *text)
{
  auto cancelRequested = std::make_shared<std::atomic_bool>(false);
  auto future = enqueueTask(
      [task = std::forward<F>(f), cancelRequested]() mutable {
        task(*cancelRequested);
      });

  if (!m_taskModal) {
    tsd::core::logWarning(
        "[Application] No task modal available to show, "
        "executing task without showing modal.");
    future.wait();
  } else {
    m_taskModal->activate(std::move(future), text, cancelRequested);
  }
}

} // namespace tsd::ui::imgui
