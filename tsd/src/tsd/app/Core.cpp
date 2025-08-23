// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#define ANARI_EXTENSION_UTILITY_IMPL

#include "Core.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/importers.hpp"
#include "tsd/io/procedural.hpp"
#include "tsd/io/serialization.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/rendering/index/RenderIndexFlatRegistry.hpp"

namespace tsd::app {

void anariStatusFunc(const void *_core,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  const char *typeStr = anari::toString(sourceType);
  const auto *core = (const Core *)_core;
  const bool verbose = core->logging.verbose;

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

// Core definitions ////////////////////////////////////////////////////////

Core::Core() : anari(this)
{
  tsd.ctx.setUpdateDelegate(&anari.getUpdateDelegate());
}

Core::~Core()
{
  anari.releaseAllDevices();
}

void Core::parseCommandLine(int argc, const char **argv)
{
  this->commandLine.libraryList = parseLibraryList();

  if (argc < 2 || argv == nullptr)
    return;

  auto importerType = ImporterType::NONE;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose")
      this->logging.verbose = true;
    else if (arg == "-e" || arg == "--echoOutput")
      this->logging.echoOutput = true;
    else if (arg == "--noDefaultLayout")
      this->commandLine.useDefaultLayout = false;
    else if (arg == "-pd" || arg == "--preloadDevices")
      this->commandLine.preloadDevices = true;
    else if (arg == "--secondaryView" || arg == "-sv")
      this->commandLine.secondaryViewportLibrary = argv[++i];
    else if (arg == "-tsd") {
      importerType = ImporterType::TSD;
      this->commandLine.loadingContext = true;
    } else if (arg == "-hdri")
      importerType = ImporterType::HDRI;
    else if (arg == "-dlaf")
      importerType = ImporterType::DLAF;
    else if (arg == "-e57xyz")
      importerType = ImporterType::E57XYZ;
    else if (arg == "-nbody")
      importerType = ImporterType::NBODY;
    else if (arg == "-obj")
      importerType = ImporterType::OBJ;
    else if (arg == "-usd")
      importerType = ImporterType::USD;
    else if (arg == "-assimp")
      importerType = ImporterType::ASSIMP;
    else if (arg == "-assimp_flat")
      importerType = ImporterType::ASSIMP_FLAT;
    else if (arg == "-ply")
      importerType = ImporterType::PLY;
    else if (arg == "-volume")
      importerType = ImporterType::VOLUME;
    else if (arg == "-swc")
      importerType = ImporterType::SWC;
    else if (arg == "-pdb")
      importerType = ImporterType::PDB;
    else if (arg == "-xyzdp")
      importerType = ImporterType::XYZDP;
    else if (arg == "-hsmesh")
      importerType = ImporterType::HSMESH;
    else if (arg == "-pt")
      importerType = ImporterType::NEURAL;
    else if (arg == "-gltf")
      importerType = ImporterType::GLTF;
    else
      this->commandLine.filenames.push_back({importerType, arg});
  }
}

void Core::setupSceneFromCommandLine(bool hdriOnly)
{
  if (hdriOnly) {
    for (const auto &f : commandLine.filenames) {
      tsd::core::logStatus("...loading file '%s'", f.second.c_str());
      if (f.first == ImporterType::HDRI)
        tsd::io::import_HDRI(tsd.ctx, f.second.c_str());
    }
    return;
  }

  if (!commandLine.loadedFromStateFile && commandLine.filenames.empty()) {
    tsd::core::logStatus("...generating material_orb from embedded data");
    tsd::io::generate_material_orb(tsd.ctx);
  } else {
    for (const auto &f : commandLine.filenames) {
      tsd::core::logStatus("...loading file '%s'", f.second.c_str());
      auto root = tsd.ctx.defaultLayer()->root();
      if (f.first == ImporterType::TSD)
        tsd::io::load_Context(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::PLY)
        tsd::io::import_PLY(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::OBJ)
        tsd::io::import_OBJ(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::USD)
        tsd::io::import_USD(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::ASSIMP)
        tsd::io::import_ASSIMP(tsd.ctx, f.second.c_str(), root, false);
      else if (f.first == ImporterType::ASSIMP_FLAT)
        tsd::io::import_ASSIMP(tsd.ctx, f.second.c_str(), root, true);
      else if (f.first == ImporterType::DLAF)
        tsd::io::import_DLAF(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::E57XYZ)
        tsd::io::import_E57XYZ(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::NBODY)
        tsd::io::import_NBODY(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::HDRI)
        tsd::io::import_HDRI(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::SWC)
        tsd::io::import_SWC(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::PDB)
        tsd::io::import_PDB(tsd.ctx, f.second.c_str(), root);
      else if (f.first == ImporterType::XYZDP)
        tsd::io::import_XYZDP(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::HSMESH)
        tsd::io::import_HSMESH(tsd.ctx, f.second.c_str(), root);
      else if (f.first == ImporterType::VOLUME)
        tsd::io::import_volume(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::NEURAL)
        tsd::io::import_PT(tsd.ctx, f.second.c_str(), root);
      else if (f.first == ImporterType::GLTF)
        tsd::io::import_GLTF(tsd.ctx, f.second.c_str(), root);
      else
        tsd::core::logWarning("...skipping unknown file type for '%s'",
            f.second.c_str());
    }
  }
}

ANARIDeviceManager::ANARIDeviceManager(Core *core) : m_core(core) {}

anari::Device ANARIDeviceManager::loadDevice(const std::string &libraryName)
{
  if (libraryName.empty() || libraryName == "{none}")
    return nullptr;

  anari::Device dev = m_loadedDevices[libraryName];
  if (dev) {
    anari::retain(dev, dev);
    return dev;
  }

  auto library =
      anari::loadLibrary(libraryName.c_str(), anariStatusFunc, m_core);
  if (!library)
    return nullptr;

  dev = anari::newDevice(library, "default");

  m_loadedDeviceExtensions[libraryName] =
      anari::extension::getDeviceExtensionStruct(library, "default");

  anari::unloadLibrary(library);

  anari::setParameter(dev, dev, "glAPI", "OpenGL");
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
    tsd::core::Context &c, anari::Device d)
{
  auto &liveIdx = m_rIdxs[d];
  if (liveIdx.refCount == 0) {
    if (useFlatRenderIndex()) {
      liveIdx.idx =
          m_delegate.emplace<tsd::rendering::RenderIndexFlatRegistry>(c, d);
    } else {
      liveIdx.idx =
          m_delegate.emplace<tsd::rendering::RenderIndexAllLayers>(c, d);
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

void Core::setOfflineRenderingLibrary(const std::string &libName)
{
  auto &dm = this->anari;
  auto d = dm.loadDevice(libName);
  if (!d) {
    tsd::core::logError(
        "[Core] Failed to load ANARI device for offline rendering: %s",
        libName.c_str());
    return;
  }

  this->offline.renderer.rendererObjects.clear();
  this->offline.renderer.activeRenderer = 0;
  this->offline.renderer.libraryName = libName;

  for (auto &name : tsd::core::getANARIObjectSubtypes(d, ANARI_RENDERER)) {
    auto o = tsd::core::parseANARIObjectInfo(d, ANARI_RENDERER, name.c_str());
    o.setName(name.c_str());
    this->offline.renderer.rendererObjects.push_back(std::move(o));
  }

  anari::release(d, d);
}

void Core::setSelectedObject(tsd::core::Object *o)
{
  tsd.selectedNode = {};
  tsd.selectedObject = o;
  anari.getUpdateDelegate().signalObjectFilteringChanged();
}

void Core::setSelectedNode(tsd::core::LayerNode &n)
{
  setSelectedObject(tsd.ctx.getObject(n->value));
  auto *layer = n.container();
  tsd.selectedNode = layer->at(n.index());
}

bool Core::objectIsSelected() const
{
  return tsd.selectedObject != nullptr;
}

void Core::clearSelected()
{
  if (tsd.selectedObject != nullptr || tsd.selectedNode) {
    tsd.selectedObject = nullptr;
    tsd.selectedNode = {};
    anari.getUpdateDelegate().signalObjectFilteringChanged();
  }
}

void Core::addCurrentViewToCameraPoses(const char *_name)
{
  auto azel = view.manipulator.azel();
  auto dist = view.manipulator.distance();
  tsd::math::float3 azeldist(azel.x, azel.y, dist);

  std::string name = _name;
  if (name.empty())
    name = "user_view" + std::to_string(view.poses.size());

  CameraPose pose;
  pose.name = name;
  pose.lookat = view.manipulator.at();
  pose.azeldist = azeldist;
  pose.upAxis = static_cast<int>(view.manipulator.axis());

  view.poses.push_back(std::move(pose));
}

void Core::addTurntableCameraPoses(const tsd::math::float3 &azs,
    const tsd::math::float3 &els,
    const tsd::math::float3 &center,
    float dist,
    const char *_name)
{
  if (azs.z <= 0.f || els.z <= 0.f) {
    tsd::core::logError("invalid turntable azimuth/elevation step size");
    return;
  }

  std::string baseName = _name;
  if (baseName.empty())
    baseName = "turntable_view";

  int j = 0;
  for (float el = els.x; el <= els.y; el += els.z, j++) {
    int i = 0;
    for (float az = azs.x; az <= azs.y; az += azs.z, i++) {
      CameraPose pose;
      pose.name = baseName + "_" + std::to_string(i) + "_" + std::to_string(j);
      pose.lookat = center;
      pose.azeldist = {az, el, dist};
      pose.upAxis = static_cast<int>(view.manipulator.axis());
      view.poses.push_back(std::move(pose));
#if 0
      printf("added turntable pose '%s' at azimuth %.2f, elevation %.2f\n",
          view.poses.back().name.c_str(),
          az,
          el);
#endif
    }
  }
}

void Core::updateExistingCameraPoseFromView(CameraPose &p)
{
  auto azel = view.manipulator.azel();
  auto dist = view.manipulator.distance();
  tsd::math::float3 azeldist(azel.x, azel.y, dist);

  p.lookat = view.manipulator.at();
  p.azeldist = azeldist;
  p.upAxis = static_cast<int>(view.manipulator.axis());
}

void Core::setCameraPose(const CameraPose &pose)
{
  view.manipulator.setConfig(
      pose.lookat, pose.azeldist.z, {pose.azeldist.x, pose.azeldist.y});
  view.manipulator.setAxis(static_cast<tsd::rendering::UpAxis>(pose.upAxis));
}

void Core::removeAllPoses()
{
  view.poses.clear();
}

void OfflineRenderSequenceConfig::saveSettings(tsd::core::DataNode &root)
{
  root.reset(); // clear all previous values, if they exist

  auto &frameRoot = root["frame"];
  frameRoot["width"] = frame.width;
  frameRoot["height"] = frame.height;
  frameRoot["colorFormat"] = frame.colorFormat;
  frameRoot["samples"] = frame.samples;

  auto &cameraRoot = root["camera"];
  cameraRoot["apertureRadius"] = camera.apertureRadius;
  cameraRoot["focusDistance"] = camera.focusDistance;

  auto &rendererRoot = root["renderer"];
  rendererRoot["activeRenderer"] = renderer.activeRenderer;
  rendererRoot["libraryName"] = renderer.libraryName;

  auto &rendererObjectsRoot = rendererRoot["rendererObjects"];
  for (auto &ro : renderer.rendererObjects)
    tsd::io::objectToNode(ro, rendererObjectsRoot[ro.name()]);
}

void OfflineRenderSequenceConfig::loadSettings(tsd::core::DataNode &root)
{
  auto &frameRoot = root["frame"];
  frameRoot["width"].getValue(ANARI_UINT32, &frame.width);
  frameRoot["height"].getValue(ANARI_UINT32, &frame.height);
  frameRoot["colorFormat"].getValue(ANARI_DATA_TYPE, &frame.colorFormat);
  frameRoot["samples"].getValue(ANARI_UINT32, &frame.samples);

  auto &cameraRoot = root["camera"];
  cameraRoot["apertureRadius"].getValue(ANARI_FLOAT32, &camera.apertureRadius);
  cameraRoot["focusDistance"].getValue(ANARI_FLOAT32, &camera.focusDistance);

  auto &rendererRoot = root["renderer"];
  rendererRoot["activeRenderer"].getValue(
      ANARI_INT32, &renderer.activeRenderer);
  rendererRoot["libraryName"].getValue(ANARI_STRING, &renderer.libraryName);

  auto &rendererObjectsRoot = rendererRoot["rendererObjects"];
  renderer.rendererObjects.clear();
  rendererObjectsRoot.foreach_child([&](auto &node) {
    tsd::core::Object ro(ANARI_RENDERER, node.name().c_str());
    tsd::io::nodeToObject(node, ro);
    renderer.rendererObjects.push_back(std::move(ro));
  });
}

} // namespace tsd::app
