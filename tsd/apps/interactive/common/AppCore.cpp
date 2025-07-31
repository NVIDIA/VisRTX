// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#define ANARI_EXTENSION_UTILITY_IMPL

#include "AppCore.h"
#include "windows/Log.h"
// SDL
#include <SDL3/SDL_dialog.h>

namespace tsd_viewer {

static void statusFunc(const void *_core,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  const char *typeStr = anari::toString(sourceType);
  const auto *core = (const AppCore *)_core;
  const bool verbose = core->logging.verbose;

  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[ANARI][FATAL][%s][%p] %s", typeStr, source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    tsd::logError("[ANARI][ERROR][%s][%p] %s", typeStr, source, message);
  else if (severity == ANARI_SEVERITY_WARNING)
    tsd::logWarning("[ANARI][WARN ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    tsd::logPerfWarning("[ANARI][PERF ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_INFO)
    tsd::logInfo("[ANARI][INFO ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_DEBUG)
    tsd::logDebug("[ANARI][DEBUG][%s][%p] %s", typeStr, source, message);
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

// AppCore definitions ////////////////////////////////////////////////////////

AppCore::AppCore(anari_viewer::Application *app) : application(app)
{
  tsd.ctx.setUpdateDelegate(&anari.delegate);
}

AppCore::~AppCore()
{
  releaseAllDevices();
}

void AppCore::parseCommandLine(int argc, const char **argv)
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
    else
      this->commandLine.filenames.push_back({importerType, arg});
  }
}

void AppCore::setupSceneFromCommandLine(bool hdriOnly)
{
  if (hdriOnly) {
    for (const auto &f : commandLine.filenames) {
      tsd::logStatus("...loading file '%s'", f.second.c_str());
      if (f.first == ImporterType::HDRI)
        tsd::import_HDRI(tsd.ctx, f.second.c_str());
    }
    return;
  }

  if (commandLine.filenames.empty()) {
    tsd::logStatus("...generating material_orb from embedded data");
    tsd::generate_material_orb(tsd.ctx);
  } else {
    for (const auto &f : commandLine.filenames) {
      tsd::logStatus("...loading file '%s'", f.second.c_str());
      auto root = tsd.ctx.defaultLayer()->root();
      if (f.first == ImporterType::TSD)
        tsd::load_Context(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::PLY)
        tsd::import_PLY(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::OBJ)
        tsd::import_OBJ(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::USD)
        tsd::import_USD(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::ASSIMP)
        tsd::import_ASSIMP(tsd.ctx, f.second.c_str(), root, false);
      else if (f.first == ImporterType::ASSIMP_FLAT)
        tsd::import_ASSIMP(tsd.ctx, f.second.c_str(), root, true);
      else if (f.first == ImporterType::DLAF)
        tsd::import_DLAF(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::E57XYZ)
        tsd::import_E57XYZ(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::NBODY)
        tsd::import_NBODY(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::HDRI)
        tsd::import_HDRI(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::SWC)
        tsd::import_SWC(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::PDB)
        tsd::import_PDB(tsd.ctx, f.second.c_str(), root);
      else if (f.first == ImporterType::XYZDP)
        tsd::import_XYZDP(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::HSMESH)
        tsd::import_HSMESH(tsd.ctx, f.second.c_str(), root);
      else if (f.first == ImporterType::VOLUME)
        tsd::import_volume(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::NEURAL)
        tsd::import_PT(tsd.ctx, f.second.c_str(), root);
    }
  }
}

void AppCore::getFilenameFromDialog(std::string &filenameOut, bool save)
{
  auto fileDialogCb =
      [](void *userdata, const char *const *filelist, int filter) {
        std::string &out = *(std::string *)userdata;
        if (!filelist) {
          tsd::logError("SDL DIALOG ERROR: %s\n", SDL_GetError());
          return;
        }

        if (*filelist)
          out = *filelist;
      };

  if (save) {
    SDL_ShowSaveFileDialog(fileDialogCb,
        &filenameOut,
        application->sdlWindow(),
        nullptr,
        0,
        nullptr);
  } else {
    SDL_ShowOpenFileDialog(fileDialogCb,
        &filenameOut,
        application->sdlWindow(),
        nullptr,
        0,
        nullptr,
        false);
  }
}

anari::Device AppCore::loadDevice(const std::string &libraryName)
{
  if (libraryName.empty() || libraryName == "{none}")
    return nullptr;

  anari::Device dev = this->anari.loadedDevices[libraryName];
  if (dev) {
    anari::retain(dev, dev);
    return dev;
  }

  auto library = anari::loadLibrary(libraryName.c_str(), statusFunc, this);
  if (!library)
    return nullptr;

  dev = anari::newDevice(library, "default");

  this->anari.loadedDeviceExtensions[libraryName] =
      anari::extension::getDeviceExtensionStruct(library, "default");

  anari::unloadLibrary(library);

  anari::setParameter(dev, dev, "glAPI", "OpenGL");
  anari::commitParameters(dev, dev);

  this->anari.loadedDevices[libraryName] = dev;
  anari::retain(dev, dev);

  return dev;
}

const anari::Extensions *AppCore::loadDeviceExtensions(
    const std::string &libName)
{
  auto d = loadDevice(libName);
  if (!d)
    return nullptr;
  anari::release(d, d);
  return &this->anari.loadedDeviceExtensions[libName];
}

tsd::RenderIndex *AppCore::acquireRenderIndex(anari::Device d)
{
  auto &liveIdx = this->anari.rIdxs[d];
  if (liveIdx.refCount == 0) {
#if 1
    liveIdx.idx =
        anari.delegate.emplace<tsd::RenderIndexAllLayers>(this->tsd.ctx, d);
#else
    liveIdx.idx =
        anari.delegate.emplace<tsd::RenderIndexFlatRegistry>(this->tsd.ctx, d);
#endif
    liveIdx.idx->populate(false);
  }
  liveIdx.refCount++;
  return liveIdx.idx;
}

void AppCore::releaseRenderIndex(anari::Device d)
{
  auto &liveIdx = this->anari.rIdxs[d];
  if (liveIdx.refCount == 0)
    return;
  else if (liveIdx.refCount == 1)
    anari.delegate.erase(liveIdx.idx);
  liveIdx.refCount--;
}

void AppCore::releaseAllDevices()
{
  for (auto &d : anari.loadedDevices) {
    if (d.second)
      anari::release(d.second, d.second);
  }
  anari.loadedDevices.clear();
}

void AppCore::setOfflineRenderingLibrary(const std::string &libName)
{
  auto d = this->loadDevice(libName);
  if (!d) {
    tsd::logError(
        "[AppCore] Failed to load ANARI device for offline rendering: %s",
        libName.c_str());
    return;
  }

  this->offline.renderer.rendererObjects.clear();
  this->offline.renderer.activeRenderer = 0;
  this->offline.renderer.libraryName = libName;

  for (auto &name : tsd::getANARIObjectSubtypes(d, ANARI_RENDERER)) {
    auto o = tsd::parseANARIObjectInfo(d, ANARI_RENDERER, name.c_str());
    o.setName(name.c_str());
    this->offline.renderer.rendererObjects.push_back(std::move(o));
  }

  anari::release(d, d);
}

void AppCore::setSelectedObject(tsd::Object *o)
{
  tsd.selectedNode = {};
  tsd.selectedObject = o;
  anari.delegate.signalObjectFilteringChanged();
}

void AppCore::setSelectedNode(tsd::LayerNode &n)
{
  setSelectedObject(tsd.ctx.getObject(n->value));
  auto *layer = n.container();
  tsd.selectedNode = layer->at(n.index());
}

bool AppCore::objectIsSelected() const
{
  return tsd.selectedObject != nullptr;
}

void AppCore::clearSelected()
{
  if (tsd.selectedObject != nullptr || tsd.selectedNode) {
    tsd.selectedObject = nullptr;
    tsd.selectedNode = {};
    anari.delegate.signalObjectFilteringChanged();
  }
}

void AppCore::addCurrentViewToCameraPoses(const char *_name)
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

void AppCore::addTurntableCameraPoses(const tsd::float3 &azs,
    const tsd::float3 &els,
    const tsd::float3 &center,
    float dist,
    const char *_name)
{
  if (azs.z <= 0.f || els.z <= 0.f) {
    tsd::logError("invalid turntable azimuth/elevation step size");
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

void AppCore::updateExistingCameraPoseFromView(CameraPose &p)
{
  auto azel = view.manipulator.azel();
  auto dist = view.manipulator.distance();
  tsd::math::float3 azeldist(azel.x, azel.y, dist);

  p.lookat = view.manipulator.at();
  p.azeldist = azeldist;
  p.upAxis = static_cast<int>(view.manipulator.axis());
}

void AppCore::setCameraPose(const CameraPose &pose)
{
  view.manipulator.setConfig(
      pose.lookat, pose.azeldist.z, {pose.azeldist.x, pose.azeldist.y});
  view.manipulator.setAxis(static_cast<tsd::manipulators::UpAxis>(pose.upAxis));
}

void AppCore::removeAllPoses()
{
  view.poses.clear();
}

void AppCore::OfflineRenderSequenceConfig::saveSettings(
    tsd::serialization::DataNode &root)
{
  root.reset(); // clear all previous values, if they exist

  auto &frameRoot = root["frame"];
  frameRoot["width"] = frame.width;
  frameRoot["height"] = frame.height;
  frameRoot["colorFormat"] = frame.colorFormat;
  frameRoot["samples"] = frame.samples;

  auto &rendererRoot = root["renderer"];
  rendererRoot["activeRenderer"] = renderer.activeRenderer;
  rendererRoot["libraryName"] = renderer.libraryName;

  auto &rendererObjectsRoot = rendererRoot["rendererObjects"];
  for (auto &ro : renderer.rendererObjects)
    objectToNode(ro, rendererObjectsRoot[ro.name()]);
}

void AppCore::OfflineRenderSequenceConfig::loadSettings(
    tsd::serialization::DataNode &root)
{
  auto &frameRoot = root["frame"];
  frameRoot["width"].getValue(ANARI_UINT32, &frame.width);
  frameRoot["height"].getValue(ANARI_UINT32, &frame.height);
  frameRoot["colorFormat"].getValue(ANARI_DATA_TYPE, &frame.colorFormat);
  frameRoot["samples"].getValue(ANARI_UINT32, &frame.samples);

  auto &rendererRoot = root["renderer"];
  rendererRoot["activeRenderer"].getValue(
      ANARI_INT32, &renderer.activeRenderer);
  rendererRoot["libraryName"].getValue(ANARI_STRING, &renderer.libraryName);

  auto &rendererObjectsRoot = rendererRoot["rendererObjects"];
  renderer.rendererObjects.clear();
  rendererObjectsRoot.foreach_child([&](auto &node) {
    tsd::Object ro(ANARI_RENDERER, node.name().c_str());
    nodeToObject(node, ro);
    renderer.rendererObjects.push_back(std::move(ro));
  });
}

} // namespace tsd_viewer
