// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/TSD.hpp"
// std
#include <map>
#include <memory>
#include <string>

namespace tsd_viewer {

struct ImportFileDialog;

enum class ImporterType
{
  ASSIMP = 0,
  ASSIMP_FLAT,
  DLAF,
  NBODY,
  PLY,
  OBJ,
  HDRI,
  VOLUME,
  TSD,
  NONE
};

struct AppCore
{
  struct CommandLineOptions
  {
    bool useDefaultLayout{true};
    bool enableDebug{false};
    bool loadingContext{false};
    anari::Library debug{nullptr};
    std::string traceDir;
    std::vector<std::pair<ImporterType, std::string>> filenames;
    ImporterType importerType{ImporterType::NONE};
    std::vector<std::string> libraryList;
  } commandLine;

  struct TSDState
  {
    tsd::Context ctx;
    bool sceneLoadComplete{false};
    tsd::Object *selectedObject{nullptr};
    tsd::InstanceNode::Ref selectedNode;
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
  } anari;

  struct LogState
  {
    bool verbose{false};
    bool echoOutput{false};
  } logging;

  struct Windows
  {
    ImportFileDialog *importDialog{nullptr};
  } windows;

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  AppCore();
  ~AppCore();

  void parseCommandLine(int argc, const char **argv);
  void setupSceneFromCommandLine(bool hdriOnly = false);

  anari::Device loadDevice(const std::string &libName);
  tsd::RenderIndex *acquireRenderIndex(anari::Device device);
  void releaseRenderIndex(anari::Device device);
  void releaseAllDevices();

  void setSelectedObject(tsd::Object *o);
  void setSelectedNode(tsd::InstanceNode &n);
  void clearSelected();

  // Not copyable or moveable //
  AppCore(const AppCore &) = delete;
  AppCore(AppCore &&) = delete;
  AppCore &operator=(const AppCore &) = delete;
  AppCore &operator=(AppCore &&) = delete;
  //////////////////////////////
};

} // namespace tsd_viewer
