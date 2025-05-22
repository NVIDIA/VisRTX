// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
#include "Manipulator.h"
#include "modals/AppSettingsDialog.h"
#include "modals/ImportFileDialog.h"
// tsd
#include "tsd/TSD.hpp"

namespace tsd_viewer {

struct Window;

class TSDApplication : public anari_viewer::Application
{
 public:
  TSDApplication(int argc = 0, const char **argv = nullptr);
  ~TSDApplication() override;

  AppCore *appCore();

  // Things from anari_viewer::Application to override //

  virtual anari_viewer::WindowArray setupWindows() override;
  virtual void uiFrameStart() override;
  virtual void teardown() override;

  // Not movable or copyable //

  TSDApplication(const TSDApplication &) = delete;
  TSDApplication &operator=(const TSDApplication &) = delete;
  TSDApplication(TSDApplication &&) = delete;
  TSDApplication &operator=(TSDApplication &&) = delete;

 protected:
  void saveApplicationState(const char *filename = "state.tsd");
  void loadApplicationState(const char *filename = "state.tsd");

  void setupUsdDevice();
  bool usdDeviceSetup() const;
  void syncUsdScene();
  void teardownUsdDevice();

  void setWindowArray(const anari_viewer::WindowArray &wa);
  virtual const char *getDefaultLayout() const = 0;

  // Data //

  manipulators::Orbit m_manipulator;
  std::vector<Window *> m_windows;
  std::unique_ptr<tsd_viewer::AppSettingsDialog> m_appSettingsDialog;
  std::unique_ptr<tsd_viewer::ImportFileDialog> m_fileDialog;

  tsd::serialization::DataTree m_settings;

 private:
  void updateWindowTitle();

  // Data //
  AppCore m_core;

  std::string m_applicationName = "TSD";

  std::string m_currentSessionFilename;
  std::string m_filenameToSaveNextFrame;
  std::string m_filenameToLoadNextFrame;

  struct UsdDeviceState
  {
    anari::Device device{nullptr};
    anari::Frame frame{nullptr};
    tsd::RenderIndex *renderIndex{nullptr};
  } m_usd;
};

} // namespace tsd_viewer
