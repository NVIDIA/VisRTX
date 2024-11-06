// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/TSD.hpp"
// std
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace tsd_viewer {

enum class ImporterType
{
  TSD,
  ASSIMP,
  ASSIMP_FLAT,
  DLAF,
  NBODY,
  PLY,
  OBJ,
  HDRI,
  NONE
};

struct AppContext
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
  } tsd;

  struct DeviceState
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

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  AppContext();
  ~AppContext();

  void parseCommandLine(int argc, char *argv[]);
  void setupSceneFromCommandLine(bool hdriOnly = false);

  anari::Device loadDevice(const std::string &libName);
  tsd::RenderIndex *acquireRenderIndex(anari::Device device);
  void releaseRenderIndex(anari::Device device);
  void releaseAllDevices();

  void setSelectedObject(tsd::Object *o);

  // Not copyable or moveable //
  AppContext(const AppContext &) = delete;
  AppContext(AppContext &&) = delete;
  AppContext &operator=(const AppContext &) = delete;
  AppContext &operator=(AppContext &&) = delete;
  //////////////////////////////
};

} // namespace tsd_viewer
