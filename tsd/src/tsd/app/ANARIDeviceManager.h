// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
// std
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tsd::app {

using DeviceInitParam = std::pair<std::string, tsd::core::Any>;

struct ANARIDeviceManager
{
  ANARIDeviceManager(const bool *verboseFlag = nullptr);

  const std::vector<std::string> &libraryList() const;
  void setLibraryList(const std::vector<std::string> &libs);

  anari::Device loadDevice(const std::string &libName,
      const std::vector<DeviceInitParam> &initialDeviceParams = {});

  const anari::Extensions *loadDeviceExtensions(const std::string &libName);
  tsd::rendering::RenderIndex *acquireRenderIndex(
      tsd::core::Scene &c, tsd::core::Token deviceName, anari::Device device);
  void releaseRenderIndex(anari::Device device);
  void releaseAllDevices();

  tsd::core::MultiUpdateDelegate &getUpdateDelegate();

  void setUseFlatRenderIndex(
      bool f); // next acquireRenderIndex(...) will use flat render index
  bool useFlatRenderIndex() const;

  void saveSettings(tsd::core::DataNode &root);
  void loadSettings(tsd::core::DataNode &root);

 private:
  const bool *m_verboseFlag{nullptr};
  struct LiveAnariIndex
  {
    int refCount{0};
    tsd::rendering::RenderIndex *idx{nullptr};
  };
  std::map<anari::Device, LiveAnariIndex> m_rIdxs;
  tsd::core::MultiUpdateDelegate m_delegate;
  std::map<std::string, anari::Device> m_loadedDevices;
  std::map<std::string, anari::Extensions> m_loadedDeviceExtensions;
  std::vector<std::string> m_libraryList;

  // Settings //

  struct Settings
  {
    // Use flat render index by default, unless set otherwise
    // This is to avoid issues with instancing in the scene graph
    // and to allow for faster rendering in some cases.
    bool forceFlat{false};
  } m_settings;
};

void anariStatusFunc(const void *_core,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message);

} // namespace tsd::app
