// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Context.hpp"
// std
#include <map>
#include <memory>
#include <string>
// anari_viewer
#include "anari_viewer/Application.h"

#include "tsd/app/TaskQueue.h"

namespace tsd::app {

struct BlockingTaskModal;
struct ImportFileDialog;

using CameraPose = tsd::rendering::CameraPose;

enum class ImporterType
{
  ASSIMP = 0,
  ASSIMP_FLAT,
  DLAF,
  E57XYZ,
  NBODY,
  PLY,
  OBJ,
  USD,
  HDRI,
  VOLUME,
  SWC,
  PDB,
  XYZDP,
  HSMESH,
  NEURAL,
  TSD,
  NONE
};

struct CommandLineOptions
{
  bool useDefaultLayout{true};
  bool loadingContext{false};
  bool preloadDevices{false};
  bool loadedFromStateFile{false};
  std::vector<std::pair<ImporterType, std::string>> filenames;
  ImporterType importerType{ImporterType::NONE};
  std::vector<std::string> libraryList;
  std::string secondaryViewportLibrary;
};

struct TSDState
{
  tsd::Context ctx;
  bool sceneLoadComplete{false};
  tsd::Object *selectedObject{nullptr};
  tsd::LayerNodeRef selectedNode;
};

struct ANARIState
{
  anari::Device loadDevice(const std::string &libName);
  const anari::Extensions *loadDeviceExtensions(const std::string &libName);
  tsd::RenderIndex *acquireRenderIndex(anari::Device device);
  void releaseRenderIndex(anari::Device device);
  void releaseAllDevices();

 private:
  struct LiveAnariIndex
  {
    int refCount{0};
    tsd::RenderIndex *idx{nullptr};
  };
  std::map<anari::Device, LiveAnariIndex> rIdxs;
  tsd::MultiUpdateDelegate delegate;
  std::map<std::string, anari::Device> loadedDevices;
  std::map<std::string, anari::Extensions> loadedDeviceExtensions;
};

struct LogState
{
  bool verbose{false};
  bool echoOutput{false};
};

struct CameraState
{
  std::vector<CameraPose> poses;
  tsd::rendering::Manipulator manipulator;
};

struct OfflineRenderSequenceConfig
{
  struct FrameSettings
  {
    uint32_t width{1024};
    uint32_t height{768};
    anari::DataType colorFormat{ANARI_UFIXED8_RGBA_SRGB};
    uint32_t samples{128};
  } frame;

  struct CameraSettings
  {
    float apertureRadius{0.f};
    float focusDistance{1.f};
  } camera;

  struct RenderSettings
  {
    std::vector<tsd::Object> rendererObjects;
    int activeRenderer{-1};
    std::string libraryName;
  } renderer;

  void saveSettings(tsd::core::DataNode &root);
  void loadSettings(tsd::core::DataNode &root);
};

struct Windows
{
  BlockingTaskModal *taskModal{nullptr};
  ImportFileDialog *importDialog{nullptr};
  float fontScale{1.f};
  float uiRounding{9.f};
};

struct Tasking
{
  TaskQueue queue{10};
};

struct AppCore
{
  CommandLineOptions commandLine;
  TSDState tsd;
  ANARIState anari;
  LogState logging;
  CameraState view;
  OfflineRenderSequenceConfig offline;
  Windows windows;
  Tasking jobs;

  anari_viewer::Application *application{nullptr};

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  AppCore(anari_viewer::Application *app);
  ~AppCore();

  void parseCommandLine(int argc, const char **argv);
  void setupSceneFromCommandLine(bool hdriOnly = false);

  void getFilenameFromDialog(std::string &filenameOut, bool save = false);

  // ANARI device management //

  void setOfflineRenderingLibrary(const std::string &libName);

  // Selection //

  void setSelectedObject(tsd::Object *o);
  void setSelectedNode(tsd::LayerNode &n);
  bool objectIsSelected() const;
  void clearSelected();

  // Camera poses //

  void addCurrentViewToCameraPoses(const char *name = "");
  void addTurntableCameraPoses(const tsd::float3 &azimuths, // begin, end, step
      const tsd::float3 &elevations, // begin, end, step
      const tsd::float3 &center,
      float distance,
      const char *name = "");
  void updateExistingCameraPoseFromView(CameraPose &p);
  void setCameraPose(const CameraPose &pose);
  void removeAllPoses();

  // Not copyable or moveable //
  AppCore(const AppCore &) = delete;
  AppCore(AppCore &&) = delete;
  AppCore &operator=(const AppCore &) = delete;
  AppCore &operator=(AppCore &&) = delete;
  //////////////////////////////
};

} // namespace tsd::app
