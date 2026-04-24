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

enum class RenderIndexKind : int
{
  ALL_LAYERS = 0,
  FLAT
};

using DeviceInitParam = std::pair<std::string, tsd::core::Any>;

/*
 * Manages the lifecycle of ANARI devices and their associated RenderIndex
 * instances; loads libraries on demand and reference-counts one scene-owned
 * RenderIndex per ANARI device.
 *
 * Example:
 *   ANARIDeviceManager mgr;
 *   auto device = mgr.loadDevice("visrtx");
 *   auto *idx = mgr.acquireRenderIndex(scene, deviceToken, device);
 *   mgr.releaseRenderIndex(scene, device);
 */
struct ANARIDeviceManager
{
  ANARIDeviceManager(const bool *verboseFlag = nullptr);

  const std::vector<std::string> &libraryList() const;
  void setLibraryList(const std::vector<std::string> &libs);

  anari::Device loadDevice(const std::string &libName,
      const std::vector<DeviceInitParam> &initialDeviceParams = {});

 const anari::Extensions *loadDeviceExtensions(const std::string &libName);
  tsd::rendering::RenderIndex *acquireRenderIndex(
      tsd::scene::Scene &c, tsd::core::Token deviceName, anari::Device device);
  void releaseRenderIndex(tsd::scene::Scene &c, anari::Device device);
  void releaseAllDevices();

  void setRenderIndexKind(RenderIndexKind k);
  RenderIndexKind renderIndexKind() const;

  void saveSettings(tsd::core::DataNode &root);
  void loadSettings(tsd::core::DataNode &root);

 private:
  const bool *m_verboseFlag{nullptr};
  struct LiveAnariIndex
  {
    tsd::scene::Scene *scene{nullptr};
    int refCount{0};
    tsd::rendering::RenderIndex *idx{nullptr};
  };
  std::map<anari::Device, LiveAnariIndex> m_rIdxs;
  std::map<std::string, anari::Device> m_loadedDevices;
  std::map<std::string, anari::Extensions> m_loadedDeviceExtensions;
  std::vector<std::string> m_libraryList;

  // Settings //

  struct Settings
  {
    RenderIndexKind renderIndexKind{RenderIndexKind::ALL_LAYERS};
  } m_settings;
};

void anariStatusFunc(const void *_core,
    ANARIDevice device,
    ANARIObject source,
    anari::DataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message);

} // namespace tsd::app
