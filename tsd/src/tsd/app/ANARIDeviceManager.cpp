// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ANARIDeviceManager.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/rendering/index/RenderIndexFlatRegistry.hpp"

namespace tsd::app {

void anariStatusFunc(const void *_verboseFlag,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  const char *typeStr = anari::toString(sourceType);
  const auto *verboseFlag = (const bool *)_verboseFlag;
  const bool verbose = verboseFlag ? *verboseFlag : false;

  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[ANARI][FATAL][%s][%p] %s", typeStr, source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    tsd::core::logError("[ANARI][ERROR][%s][%p] %s", typeStr, source, message);
  else if (severity == ANARI_SEVERITY_WARNING)
    tsd::core::logWarning(
        "[ANARI][WARN ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    tsd::core::logPerfWarning(
        "[ANARI][PERF ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_INFO)
    tsd::core::logInfo("[ANARI][INFO ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_DEBUG)
    tsd::core::logDebug("[ANARI][DEBUG][%s][%p] %s", typeStr, source, message);
}

static std::vector<std::string> parseLibraryList()
{
  const char *libsFromEnv = getenv("TSD_ANARI_LIBRARIES");

  auto splitString = [](const std::string &input,
                         const std::string &delim) -> std::vector<std::string> {
    std::vector<std::string> tokens;
    size_t pos = 0;
    while (true) {
      size_t begin = input.find_first_not_of(delim, pos);
      if (begin == input.npos)
        return tokens;
      size_t end = input.find_first_of(delim, begin);
      tokens.push_back(input.substr(
          begin, (end == input.npos) ? input.npos : (end - begin)));
      pos = end;
    }
  };

  auto libList = splitString(libsFromEnv ? libsFromEnv : "", ",");
  if (libList.empty()) {
    libList.push_back("helide");
    libList.push_back("visrtx");
    libList.push_back("visgl");
    if (getenv("ANARI_LIBRARY"))
      libList.push_back("environment");
  }

  libList.push_back("{none}");

  return libList;
}

// ANARIDeviceManager definitions /////////////////////////////////////////////

ANARIDeviceManager::ANARIDeviceManager(const bool *verboseFlag)
    : m_verboseFlag(verboseFlag)
{
  m_libraryList = parseLibraryList();
}

const std::vector<std::string> &ANARIDeviceManager::libraryList() const
{
  return m_libraryList;
}

void ANARIDeviceManager::setLibraryList(const std::vector<std::string> &libs)
{
  m_libraryList = libs;
}

anari::Device ANARIDeviceManager::loadDevice(const std::string &libraryName,
    const std::vector<DeviceInitParam> &initialDeviceParams)
{
  if (libraryName.empty() || libraryName == "{none}")
    return nullptr;

  anari::Device dev = m_loadedDevices[libraryName];
  if (dev) {
    anari::retain(dev, dev);
    return dev;
  }

  auto library =
      anari::loadLibrary(libraryName.c_str(), anariStatusFunc, m_verboseFlag);
  if (!library)
    return nullptr;

  dev = anari::newDevice(library, "default");

  m_loadedDeviceExtensions[libraryName] =
      anari::extension::getDeviceExtensionStruct(library, "default");

  anari::unloadLibrary(library);

  anari::setParameter(dev, dev, "glAPI", "OpenGL");

  for (const auto &param : initialDeviceParams) {
    anari::setParameter(dev,
        dev,
        param.first.c_str(),
        param.second.type(),
        param.second.data());
  }

  anari::commitParameters(dev, dev);

  m_loadedDevices[libraryName] = dev;
  anari::retain(dev, dev);

  return dev;
}

const anari::Extensions *ANARIDeviceManager::loadDeviceExtensions(
    const std::string &libName)
{
  auto d = loadDevice(libName);
  if (!d)
    return nullptr;
  anari::release(d, d);
  return &m_loadedDeviceExtensions[libName];
}

tsd::rendering::RenderIndex *ANARIDeviceManager::acquireRenderIndex(
    tsd::core::Scene &c, tsd::core::Token n, anari::Device d)
{
  auto &liveIdx = m_rIdxs[d];
  if (liveIdx.refCount == 0) {
    if (useFlatRenderIndex()) {
      liveIdx.idx =
          m_delegate.emplace<tsd::rendering::RenderIndexFlatRegistry>(c, n, d);
    } else {
      liveIdx.idx =
          m_delegate.emplace<tsd::rendering::RenderIndexAllLayers>(c, n, d);
    }
    liveIdx.idx->populate(false);
  }
  liveIdx.refCount++;
  return liveIdx.idx;
}

void ANARIDeviceManager::releaseRenderIndex(anari::Device d)
{
  auto &liveIdx = m_rIdxs[d];
  if (liveIdx.refCount == 0)
    return;
  else if (liveIdx.refCount == 1)
    m_delegate.erase(liveIdx.idx);
  liveIdx.refCount--;
}

void ANARIDeviceManager::releaseAllDevices()
{
  for (auto &d : m_loadedDevices) {
    if (d.second)
      anari::release(d.second, d.second);
  }
  m_loadedDevices.clear();
}

tsd::core::MultiUpdateDelegate &ANARIDeviceManager::getUpdateDelegate()
{
  return m_delegate;
}

void ANARIDeviceManager::setUseFlatRenderIndex(bool f)
{
  m_settings.forceFlat = f;
}

bool ANARIDeviceManager::useFlatRenderIndex() const
{
  return m_settings.forceFlat;
}

void ANARIDeviceManager::saveSettings(tsd::core::DataNode &root)
{
  root.reset(); // clear all previous values, if they exist
  root["useFlatRenderIndex"] = m_settings.forceFlat;
}

void ANARIDeviceManager::loadSettings(tsd::core::DataNode &root)
{
  root["useFlatRenderIndex"].getValue(ANARI_BOOL, &m_settings.forceFlat);
}

} // namespace tsd::app
