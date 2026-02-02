// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modals/AppSettingsDialog.h"
#include "modals/BlockingTaskModal.h"
#include "modals/ExportNanoVDBFileDialog.h"
#include "modals/ImportFileDialog.h"
#include "modals/OfflineRenderModal.h"
// tsd_app
#include "tsd/app/Core.h"
// anari_viewer
#include <anari_viewer/Application.h>

namespace tsd::ui::imgui {

struct Window;

class Application : public anari_viewer::Application
{
 public:
  Application(int argc = 0, const char **argv = nullptr);
  ~Application() override;

  tsd::app::Core *appCore();

  void getFilenameFromDialog(std::string &filenameOut, bool save = false);

  template <typename T>
  T *findWindowOfType()
  {
    for (auto *w : m_windows) {
      if (auto *typed = dynamic_cast<T *>(w))
        return typed;
    }
    return nullptr;
  }

  // Not movable or copyable //
  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;
  Application(Application &&) = delete;
  Application &operator=(Application &&) = delete;
  /////////////////////////////

 protected:
  // Things from anari_viewer::Application to override //

  virtual anari_viewer::WindowArray setupWindows() override;
  virtual void uiFrameStart() override;
  virtual void teardown() override;

  // Internal API //

  virtual void uiMainMenuBar();

  void doSave(const std::string &name = "");

  void saveApplicationState(const char *filename = "state.tsd");
  void loadApplicationState(const char *filename = "state.tsd");

  void loadStateForNextFrame();

  void setupUsdDevice();
  bool usdDeviceIsSetup() const;
  void syncUsdScene();
  void teardownUsdDevice();

  void setupTsdDevice();
  bool tsdDeviceIsSetup() const;
  void syncTsdScene();
  void teardownTsdDevice();

  void setWindowArray(const anari_viewer::WindowArray &wa);
  virtual const char *getDefaultLayout() const = 0;

  template <class FUNCTION>
  void showTaskModal(FUNCTION &&f, const char *text = "Please Wait");

  // Data //

  std::vector<Window *> m_windows;
  std::unique_ptr<AppSettingsDialog> m_appSettingsDialog;
  std::unique_ptr<BlockingTaskModal> m_taskModal;
  std::unique_ptr<OfflineRenderModal> m_offlineRenderModal;
  std::unique_ptr<ImportFileDialog> m_fileDialog;
  std::unique_ptr<ExportNanoVDBFileDialog> m_exportNanoVDBFileDialog;

  tsd::core::DataTree m_settings;

 private:
  void updateWindowTitle();

  // Data //

  tsd::app::Core m_core;

  std::string m_applicationName = "TSD";

  std::string m_currentSessionFilename;
  std::string m_filenameToSaveNextFrame;
  std::string m_filenameToLoadNextFrame;

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

template <class F>
inline void Application::showTaskModal(F &&f, const char *text)
{
  m_taskModal->activate(std::forward<F>(f), text);
}

} // namespace tsd::ui::imgui
