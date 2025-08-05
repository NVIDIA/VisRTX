// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Context.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/view/Manipulator.hpp"
// std
#include <map>
#include <memory>
#include <string>
#if 0
// anari_viewer
#include "anari_viewer/Application.h"
#endif

#include "tsd/app/TaskQueue.h"

namespace tsd::ui::imgui {
struct BlockingTaskModal;
struct ImportFileDialog;
} // namespace tsd::ui::imgui

namespace tsd::app {

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
  tsd::core::Context ctx;
  bool sceneLoadComplete{false};
  tsd::core::Object *selectedObject{nullptr};
  tsd::core::LayerNodeRef selectedNode;
};

struct ANARIDeviceManager
{
  anari::Device loadDevice(const std::string &libName);
  const anari::Extensions *loadDeviceExtensions(const std::string &libName);
  tsd::rendering::RenderIndex *acquireRenderIndex(
      tsd::core::Context &c, anari::Device device);
  void releaseRenderIndex(anari::Device device);
  void releaseAllDevices();
  tsd::core::MultiUpdateDelegate &getUpdateDelegate();

 private:
  struct LiveAnariIndex
  {
    int refCount{0};
    tsd::rendering::RenderIndex *idx{nullptr};
  };
  std::map<anari::Device, LiveAnariIndex> m_rIdxs;
  tsd::core::MultiUpdateDelegate m_delegate;
  std::map<std::string, anari::Device> m_loadedDevices;
  std::map<std::string, anari::Extensions> m_loadedDeviceExtensions;
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
    std::vector<tsd::core::Object> rendererObjects;
    int activeRenderer{-1};
    std::string libraryName;
  } renderer;

  void saveSettings(tsd::core::DataNode &root);
  void loadSettings(tsd::core::DataNode &root);
};

struct Windows
{
  tsd::ui::imgui::BlockingTaskModal *taskModal{nullptr};
  tsd::ui::imgui::ImportFileDialog *importDialog{nullptr};
  float fontScale{1.f};
  float uiRounding{9.f};
};

struct Tasking
{
  TaskQueue queue{10};
};

struct Core
{
  CommandLineOptions commandLine;
  TSDState tsd;
  ANARIDeviceManager anari;
  LogState logging;
  CameraState view;
  OfflineRenderSequenceConfig offline;
  Windows windows;
  Tasking jobs;

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  Core();
  ~Core();

  void parseCommandLine(int argc, const char **argv);
  void setupSceneFromCommandLine(bool hdriOnly = false);

  // ANARI device management //

  void setOfflineRenderingLibrary(const std::string &libName);

  // Selection //

  void setSelectedObject(tsd::core::Object *o);
  void setSelectedNode(tsd::core::LayerNode &n);
  bool objectIsSelected() const;
  void clearSelected();

  // Camera poses //

  void addCurrentViewToCameraPoses(const char *name = "");
  void addTurntableCameraPoses(
      const tsd::math::float3 &azimuths, // begin, end, step
      const tsd::math::float3 &elevations, // begin, end, step
      const tsd::math::float3 &center,
      float distance,
      const char *name = "");
  void updateExistingCameraPoseFromView(CameraPose &p);
  void setCameraPose(const CameraPose &pose);
  void removeAllPoses();

  // Not copyable or moveable //
  Core(const Core &) = delete;
  Core(Core &&) = delete;
  Core &operator=(const Core &) = delete;
  Core &operator=(Core &&) = delete;
  //////////////////////////////
};

} // namespace tsd::app
