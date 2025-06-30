// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/TSD.hpp"
// std
#include <map>
#include <memory>
#include <string>
// anari_viewer
#include "anari_viewer/Application.h"

#include "TaskQueue.h"

namespace tsd_viewer {

struct BlockingTaskModal;
struct ImportFileDialog;

using CameraPose = tsd::manipulators::CameraPose;

enum class ImporterType
{
  ASSIMP = 0,
  ASSIMP_FLAT,
  DLAF,
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
  TSD,
  TSD_CONDUIT,
  NEURAL,
  NONE
};

struct AppCore
{
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
  } commandLine;

  struct TSDState
  {
    tsd::Context ctx;
    bool sceneLoadComplete{false};
    tsd::Object *selectedObject{nullptr};
    tsd::LayerNodeRef selectedNode;
  } tsd;

  struct ANARIState
  {
    struct LiveAnariIndex
    {
      int refCount{0};
      tsd::RenderIndex *idx{nullptr};
    };
    std::map<anari::Device, LiveAnariIndex> rIdxs;
    tsd::MultiUpdateDelegate delegate;
    std::map<std::string, anari::Device> loadedDevices;
    std::map<std::string, anari::Extensions> loadedDeviceExtensions;
  } anari;

  struct LogState
  {
    bool verbose{false};
    bool echoOutput{false};
  } logging;

  struct CameraState
  {
    std::vector<CameraPose> poses;
    tsd::manipulators::Orbit manipulator;
  } view;

  struct Windows
  {
    BlockingTaskModal *taskModal{nullptr};
    ImportFileDialog *importDialog{nullptr};
    float fontScale{1.f};
  } windows;

  struct Tasking
  {
    tasking::TaskQueue queue{10};
  } jobs;

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

  anari::Device loadDevice(const std::string &libName);
  const anari::Extensions *loadDeviceExtensions(const std::string &libName);
  tsd::RenderIndex *acquireRenderIndex(anari::Device device);
  void releaseRenderIndex(anari::Device device);
  void releaseAllDevices();

  // Selection //

  void setSelectedObject(tsd::Object *o);
  void setSelectedNode(tsd::LayerNode &n);
  bool objectIsSelected() const;
  void clearSelected();

  // Camera poses //

  void addCurrentViewToCameraPoses(const char *name = "");
  void updateExistingCameraPoseFromView(CameraPose &p);
  void setCameraPose(const CameraPose &pose);

  // Not copyable or moveable //
  AppCore(const AppCore &) = delete;
  AppCore(AppCore &&) = delete;
  AppCore &operator=(const AppCore &) = delete;
  AppCore &operator=(AppCore &&) = delete;
  //////////////////////////////
};

} // namespace tsd_viewer
