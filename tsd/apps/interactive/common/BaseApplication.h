// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
#include "Manipulator.h"
#include "modals/AppSettings.h"
#include "modals/ImportFileDialog.h"
// anari_viewer
#include "anari_viewer/Application.h"
// tsd
#include "tsd/TSD.hpp"

namespace tsd_viewer {

class BaseApplication : public anari_viewer::Application
{
 public:
  BaseApplication(int argc = 0, const char **argv = nullptr);
  ~BaseApplication() override;

  AppCore *appCore();

  // Things from anari_viewer::Application to override //

  virtual anari_viewer::WindowArray setupWindows() override;
  virtual void uiFrameStart() override;
  virtual void teardown() override;

 protected:
  void saveContext();

  void setupUsdDevice();
  bool usdDeviceSetup() const;
  void syncUsdScene();
  void teardownUsdDevice();

  void setWindowArray(const anari_viewer::WindowArray &wa);
  virtual const char *getDefaultLayout() const = 0;

  // Data //

  manipulators::Orbit m_manipulator;
  std::vector<anari_viewer::windows::Window *> m_windows;
  std::unique_ptr<tsd_viewer::AppSettings> m_appSettings;
  std::unique_ptr<tsd_viewer::ImportFileDialog> m_fileDialog;

 private:
  AppCore m_core;

  struct UsdDeviceState
  {
    anari::Device device{nullptr};
    anari::Frame frame{nullptr};
    tsd::RenderIndex *renderIndex{nullptr};
  } m_usd;
};

} // namespace tsd_viewer
