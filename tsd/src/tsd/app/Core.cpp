// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#define ANARI_EXTENSION_UTILITY_IMPL

#include "Core.h"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
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

static std::vector<std::string> parseLibraryList(bool defaultNone)
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

  if (defaultNone)
    libList.insert(libList.begin(), "{none}");
  else
    libList.push_back("{none}");

  return libList;
}

// Core definitions ////////////////////////////////////////////////////////

Core::Core() : anari(this)
{
  tsd.scene.setUpdateDelegate(&anari.getUpdateDelegate());

  // Initialize default transfer function
  for (const auto& c : core::colormap::viridis) {
    importer.transferFunction.colorPoints.push_back({
      float(importer.transferFunction.colorPoints.size()) / float(core::colormap::viridis.size() - 1), 
      c.x, c.y, c.z
    });
  }
  
  importer.transferFunction.opacityPoints = {
    {0.0f, 0.0f},
    {1.0f, 1.0f}
  };
  importer.transferFunction.range = {};
}

Core::~Core()
{
  anari.releaseAllDevices();
}

void Core::parseCommandLine(int argc, const char **argv)
{
  if (argc < 2 || argv == nullptr) {
    this->commandLine.libraryList =
        parseLibraryList(!this->commandLine.useDefaultRenderer);
    return;
  }

  auto &importerType = this->commandLine.importerType;

  for (int i = 1; i < argc; i++) {
    if (!argv[i])
      continue;
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
    else if (arg == "--noDefaultRenderer")
      this->commandLine.useDefaultRenderer = false;
    else if (arg == "-l" || arg == "--layer")
      this->commandLine.currentLayerName = argv[++i];
    else if (arg == "-tsd") {
      importerType = ImporterType::TSD;
      this->commandLine.loadingScene = true;
    } else if (arg == "-agx")
      importerType = ImporterType::AGX;
    else if (arg == "-assimp")
      importerType = ImporterType::ASSIMP;
    else if (arg == "-assimp_flat")
      importerType = ImporterType::ASSIMP_FLAT;
    else if (arg == "-axyz")
      importerType = ImporterType::AXYZ;
    else if (arg == "-dlaf")
      importerType = ImporterType::DLAF;
    else if (arg == "-e57xyz")
      importerType = ImporterType::E57XYZ;
    else if (arg == "-gltf")
      importerType = ImporterType::GLTF;
    else if (arg == "-hdri")
      importerType = ImporterType::HDRI;
    else if (arg == "-hsmesh")
      importerType = ImporterType::HSMESH;
    else if (arg == "-nbody")
      importerType = ImporterType::NBODY;
    else if (arg == "-obj")
      importerType = ImporterType::OBJ;
    else if (arg == "-pdb")
      importerType = ImporterType::PDB;
    else if (arg == "-ply")
      importerType = ImporterType::PLY;
    else if (arg == "-pointsbin") {
      this->commandLine.currentAnimationSequence = nullptr; // reset to new seq
      importerType = ImporterType::POINTSBIN_MULTIFILE;
    } else if (arg == "-pt")
      importerType = ImporterType::PT;
    else if (arg == "-silo")
      importerType = ImporterType::SILO;
    else if (arg == "-smesh")
      importerType = ImporterType::SMESH;
    else if (arg == "-smesh_animation")
      importerType = ImporterType::SMESH_ANIMATION;
    else if (arg == "-swc")
      importerType = ImporterType::SWC;
    else if (arg == "-trk")
      importerType = ImporterType::TRK;
    else if (arg == "-usd")
      importerType = ImporterType::USD;
    else if (arg == "-usd2")
      importerType = ImporterType::USD2;
    else if (arg == "-xyzdp")
      importerType = ImporterType::XYZDP;
    else if (arg == "-volume")
      importerType = ImporterType::VOLUME;
    else if (arg == "-blank")
      importerType = ImporterType::BLANK;
    else if (arg == "-xf" || arg == "--transferFunction")
      importerType = ImporterType::XF;
    else if (arg == "-camera" || arg == "--camera")
      this->commandLine.cameraFile = argv[++i];
    else {
      if (importerType != ImporterType::NONE) {
        if (importerType == ImporterType::POINTSBIN_MULTIFILE) {
          if (!this->commandLine.currentAnimationSequence) {
            this->commandLine.animationFilenames.push_back(
                {ImporterType::POINTSBIN_MULTIFILE, {}});
            this->commandLine.currentAnimationSequence =
                &this->commandLine.animationFilenames.back();
            this->commandLine.animationLayerNames.push_back(
                this->commandLine.currentLayerName);
          }
          this->commandLine.currentAnimationSequence->second.push_back(arg);
        } else {
          this->commandLine.filenames.push_back(
              {importerType, arg + ';' + this->commandLine.currentLayerName});
          this->commandLine.currentAnimationSequence = nullptr;
        }
      } else {
        this->commandLine.stateFile = arg;
        this->commandLine.loadedFromStateFile = true;
      }
    }
  }

  this->commandLine.currentAnimationSequence = nullptr;

  this->commandLine.libraryList =
      parseLibraryList(!this->commandLine.useDefaultRenderer);
}

void Core::setupSceneFromCommandLine(bool hdriOnly)
{
  if (hdriOnly) {
    for (const auto &f : commandLine.filenames) {
      tsd::core::logStatus("...loading file '%s'", f.second.c_str());
      if (f.first == ImporterType::HDRI)
        tsd::io::import_HDRI(tsd.scene, f.second.c_str());
    }
    return;
  }

  const bool haveFiles = commandLine.filenames.size() > 0
      || commandLine.animationFilenames.size() > 0;
  const bool blankImport =
      !haveFiles && commandLine.importerType == ImporterType::BLANK;
  const bool loadFromState = commandLine.loadedFromStateFile;

  const bool generateOrb = !(blankImport || haveFiles || loadFromState);

  if (generateOrb) {
    tsd::core::logStatus("...generating material_orb from embedded data");
    tsd::io::generate_material_orb(tsd.scene);
  } else if (!loadFromState) {
    importFiles(commandLine.filenames);
    importAnimations(commandLine.animationFilenames);
  }
}

void Core::importFile(const ImportFile &f, tsd::core::LayerNodeRef root)
{
  const bool customLocation = root;

  auto files = tsd::io::splitString(f.second, ';');
  std::string file = files[0];
  std::string layerName = files.size() > 1 ? files[1] : "";
  if (layerName.empty())
    layerName = "default";

  if (!customLocation) {
    tsd::core::logStatus(
        "...loading file '%s' in layer '%s'", file.c_str(), layerName.c_str());
    root = tsd.scene.addLayer(layerName)->root();
  } else {
    tsd::core::logStatus("...loading file '%s'", file.c_str());
  }

  if (f.first == ImporterType::TSD)
    tsd::io::load_Scene(tsd.scene, file.c_str());
  else if (f.first == ImporterType::AGX)
    tsd::io::import_AGX(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::ASSIMP)
    tsd::io::import_ASSIMP(tsd.scene, file.c_str(), root, false);
  else if (f.first == ImporterType::ASSIMP_FLAT)
    tsd::io::import_ASSIMP(tsd.scene, file.c_str(), root, true);
  else if (f.first == ImporterType::AXYZ)
    tsd::io::import_AXYZ(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::DLAF)
    tsd::io::import_DLAF(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::E57XYZ)
    tsd::io::import_E57XYZ(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::GLTF)
    tsd::io::import_GLTF(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::HDRI)
    tsd::io::import_HDRI(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::HSMESH)
    tsd::io::import_HSMESH(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::NBODY)
    tsd::io::import_NBODY(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::OBJ)
    tsd::io::import_OBJ(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::PDB)
    tsd::io::import_PDB(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::PLY)
    tsd::io::import_PLY(tsd.scene, file.c_str());
  else if (f.first == ImporterType::POINTSBIN_MULTIFILE)
    tsd::io::import_POINTSBIN(tsd.scene, {file.c_str()}, root);
  else if (f.first == ImporterType::PT)
    tsd::io::import_PT(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::SILO)
    tsd::io::import_SILO(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::SMESH)
    tsd::io::import_SMESH(tsd.scene, file.c_str(), root, false);
  else if (f.first == ImporterType::SMESH_ANIMATION)
    tsd::io::import_SMESH(tsd.scene, file.c_str(), root, true);
  else if (f.first == ImporterType::SWC)
    tsd::io::import_SWC(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::TRK)
    tsd::io::import_TRK(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::USD)
    tsd::io::import_USD(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::USD2) {
    tsd::io::import_USD(tsd.scene, file.c_str(), root);
    tsd::io::import_USD2(tsd.scene, file.c_str(), root);
  } else if (f.first == ImporterType::XYZDP)
    tsd::io::import_XYZDP(tsd.scene, file.c_str(), root);
  else if (f.first == ImporterType::VOLUME)
    tsd::io::import_volume(tsd.scene, file.c_str(), importer.transferFunction, root);
  else if (f.first == ImporterType::XF) {
    importer.transferFunction = tsd::io::importTransferFunction(file);
  }
  else if (f.first == ImporterType::BLANK) {
    // no-op
  } else {
    tsd::core::logWarning(
        "...skipping unknown file type for '%s'", file.c_str());
  }
}

void Core::importFiles(
    const std::vector<ImportFile> &files, tsd::core::LayerNodeRef root)
{
  for (const auto &f : files)
    importFile(f, root);
}

void Core::importAnimations(const std::vector<ImportAnimationFiles> &files,
    tsd::core::LayerNodeRef root)
{
  const bool customLocation = root;

  for (size_t i = 0; i < commandLine.animationFilenames.size(); ++i) {
    auto &f = commandLine.animationFilenames[i];
    auto &l = commandLine.animationLayerNames[i];
    if (!customLocation)
      root = tsd.scene.addLayer(l)->root();

    if (f.first == ImporterType::POINTSBIN_MULTIFILE)
      tsd::io::import_POINTSBIN(tsd.scene, f.second, root);
    else
      tsd::core::logWarning("...skipping unknown animation file importer type");
  }
}

ANARIDeviceManager::ANARIDeviceManager(Core *core) : m_core(core) {}

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
      anari::loadLibrary(libraryName.c_str(), anariStatusFunc, m_core);
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
    tsd::core::Scene &c, anari::Device d)
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
  setSelectedObject(n->getObject());
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
  frameRoot["numFrames"] = frame.numFrames;
  frameRoot["renderSubset"] = frame.renderSubset;
  frameRoot["startFrame"] = frame.startFrame;
  frameRoot["endFrame"] = frame.endFrame;
  frameRoot["frameIncrement"] = frame.frameIncrement;

  auto &cameraRoot = root["camera"];
  cameraRoot["apertureRadius"] = camera.apertureRadius;
  cameraRoot["focusDistance"] = camera.focusDistance;
  cameraRoot["cameraIndex"] = camera.cameraIndex;

  auto &rendererRoot = root["renderer"];
  rendererRoot["activeRenderer"] = renderer.activeRenderer;
  rendererRoot["libraryName"] = renderer.libraryName;

  auto &rendererObjectsRoot = rendererRoot["rendererObjects"];
  for (auto &ro : renderer.rendererObjects)
    tsd::io::objectToNode(ro, rendererObjectsRoot[ro.name()]);

  auto &outputRoot = root["output"];
  outputRoot["outputDirectory"] = output.outputDirectory;
  outputRoot["filePrefix"] = output.filePrefix;
}

void OfflineRenderSequenceConfig::loadSettings(tsd::core::DataNode &root)
{
  auto &frameRoot = root["frame"];
  frameRoot["width"].getValue(ANARI_UINT32, &frame.width);
  frameRoot["height"].getValue(ANARI_UINT32, &frame.height);
  frameRoot["colorFormat"].getValue(ANARI_DATA_TYPE, &frame.colorFormat);
  frameRoot["samples"].getValue(ANARI_UINT32, &frame.samples);
  frameRoot["numFrames"].getValue(ANARI_INT32, &frame.numFrames);
  frameRoot["renderSubset"].getValue(ANARI_BOOL, &frame.renderSubset);
  frameRoot["startFrame"].getValue(ANARI_INT32, &frame.startFrame);
  frameRoot["endFrame"].getValue(ANARI_INT32, &frame.endFrame);
  frameRoot["frameIncrement"].getValue(ANARI_INT32, &frame.frameIncrement);

  auto &cameraRoot = root["camera"];
  cameraRoot["apertureRadius"].getValue(ANARI_FLOAT32, &camera.apertureRadius);
  cameraRoot["focusDistance"].getValue(ANARI_FLOAT32, &camera.focusDistance);
  cameraRoot["cameraIndex"].getValue(ANARI_UINT64, &camera.cameraIndex);

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

  auto &outputRoot = root["output"];
  outputRoot["outputDirectory"].getValue(ANARI_STRING, &output.outputDirectory);
  outputRoot["filePrefix"].getValue(ANARI_STRING, &output.filePrefix);
}

} // namespace tsd::app
