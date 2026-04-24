// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#define ANARI_EXTENSION_UTILITY_IMPL

#include "Context.h"
// tsd_animation
#include "tsd/animation/Animation.hpp"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/procedural.hpp"
#include "tsd/io/serialization.hpp"

namespace tsd::app {

TSDState::TSDState() : animationMgr(&scene) {}

Context::Context() : anari(&m_logging.verbose) {}

Context::~Context()
{
  anari.releaseAllDevices();
}

void Context::parseCommandLine(int argc, const char **argv)
{
  std::vector<std::string> args(argv, argv + argc);
  parseCommandLine(args);
}

void Context::parseCommandLine(std::vector<std::string> &args)
{
  auto &importerType = this->commandLine.importerType;

  for (int i = 1; i < args.size(); i++) {
    std::string &arg = args[i];
    if (arg.empty())
      continue;

    if (arg == "-v" || arg == "--verbose")
      setLogVerbose(true);
    else if (arg == "-e" || arg == "--echoOutput")
      setLogEchoOutput(true);
    else if (arg == "-l" || arg == "--layer")
      this->commandLine.currentLayerName = args[++i];
    else if (arg == "-tsd")
      importerType = tsd::io::ImporterType::TSD;
    else if (arg == "-agx")
      importerType = tsd::io::ImporterType::AGX;
    else if (arg == "-assimp")
      importerType = tsd::io::ImporterType::ASSIMP;
    else if (arg == "-assimp_flat")
      importerType = tsd::io::ImporterType::ASSIMP_FLAT;
    else if (arg == "-axyz")
      importerType = tsd::io::ImporterType::AXYZ;
    else if (arg == "-dlaf")
      importerType = tsd::io::ImporterType::DLAF;
    else if (arg == "-e57xyz")
      importerType = tsd::io::ImporterType::E57XYZ;
    else if (arg == "-ensight")
      importerType = tsd::io::ImporterType::ENSIGHT;
    else if (arg == "-ensight_fields") {
      this->commandLine.ensightFields.clear();
      auto tokens = tsd::io::splitString(args[++i], ',');
      for (auto &t : tokens)
        if (!t.empty())
          this->commandLine.ensightFields.push_back(t);
    } else if (arg == "-gltf")
      importerType = tsd::io::ImporterType::GLTF;
    else if (arg == "-hdri")
      importerType = tsd::io::ImporterType::HDRI;
    else if (arg == "-hsmesh")
      importerType = tsd::io::ImporterType::HSMESH;
    else if (arg == "-nbody")
      importerType = tsd::io::ImporterType::NBODY;
    else if (arg == "-obj")
      importerType = tsd::io::ImporterType::OBJ;
    else if (arg == "-pdb")
      importerType = tsd::io::ImporterType::PDB;
    else if (arg == "-ply")
      importerType = tsd::io::ImporterType::PLY;
    else if (arg == "-pointsbin") {
      this->commandLine.currentAnimationSequence = nullptr; // reset to new seq
      importerType = tsd::io::ImporterType::POINTSBIN_MULTIFILE;
    } else if (arg == "-pt")
      importerType = tsd::io::ImporterType::PT;
    else if (arg == "-silo")
      importerType = tsd::io::ImporterType::SILO;
    else if (arg == "-smesh")
      importerType = tsd::io::ImporterType::SMESH;
    else if (arg == "-smesh_animation")
      importerType = tsd::io::ImporterType::SMESH_ANIMATION;
    else if (arg == "-swc")
      importerType = tsd::io::ImporterType::SWC;
    else if (arg == "-trk")
      importerType = tsd::io::ImporterType::TRK;
    else if (arg == "-usd")
      importerType = tsd::io::ImporterType::USD;
    else if (arg == "-vtp")
      importerType = tsd::io::ImporterType::VTP;
    else if (arg == "-vtu")
      importerType = tsd::io::ImporterType::VTU;
    else if (arg == "-vtu_property")
      this->commandLine.vtuProperty = args[++i];
    else if (arg == "-xyzdp")
      importerType = tsd::io::ImporterType::XYZDP;
    else if (arg == "-volume")
      importerType = tsd::io::ImporterType::VOLUME;
    else if (arg == "-volume_animation") {
      this->commandLine.currentAnimationSequence = nullptr; // reset to new seq
      importerType = tsd::io::ImporterType::VOLUME_ANIMATION;
    } else if (arg == "-blank")
      importerType = tsd::io::ImporterType::BLANK;
    else if (arg == "-xf" || arg == "--transferFunction")
      importerType = tsd::io::ImporterType::XF;
    else if (arg == "-camera" || arg == "--camera")
      this->commandLine.cameraFile = args[++i];
    else {
      if (importerType != tsd::io::ImporterType::NONE) {
        if (importerType == tsd::io::ImporterType::POINTSBIN_MULTIFILE
            || importerType == tsd::io::ImporterType::VOLUME_ANIMATION) {
          if (!this->commandLine.currentAnimationSequence) {
            this->commandLine.animationFilenames.push_back({importerType, {}});
            this->commandLine.currentAnimationSequence =
                &this->commandLine.animationFilenames.back();
            this->commandLine.animationLayerNames.push_back(
                this->commandLine.currentLayerName);
          }
          auto file = arg;
          if (importerType == tsd::io::ImporterType::VOLUME_ANIMATION
              && !this->commandLine.vtuProperty.empty())
            file += ';' + this->commandLine.vtuProperty;
          this->commandLine.currentAnimationSequence->second.push_back(file);
        } else {
          auto file = arg + ';' + this->commandLine.currentLayerName;
          if (importerType == tsd::io::ImporterType::VTU
              && !this->commandLine.vtuProperty.empty())
            file += ';' + this->commandLine.vtuProperty;
          this->commandLine.filenames.push_back({importerType, file});
          this->commandLine.currentAnimationSequence = nullptr;
        }
      } else {
        this->commandLine.stateFile = arg;
        this->commandLine.loadedFromStateFile = true;
      }
    }
  }

  this->commandLine.currentAnimationSequence = nullptr;
}

void Context::setupSceneFromCommandLine(bool hdriOnly)
{
  if (hdriOnly) {
    for (const auto &f : commandLine.filenames) {
      tsd::core::logStatus("...loading file '%s'", f.second.c_str());
      if (f.first == tsd::io::ImporterType::HDRI)
        tsd::io::import_HDRI(tsd.scene, tsd.animationMgr, f.second.c_str());
    }
    return;
  }

  const bool haveFiles = commandLine.filenames.size() > 0
      || commandLine.animationFilenames.size() > 0;
  const bool blankImport =
      !haveFiles && commandLine.importerType == tsd::io::ImporterType::BLANK;
  const bool loadFromState = commandLine.loadedFromStateFile;

  const bool generateOrb = !(blankImport || haveFiles || loadFromState);

  if (generateOrb) {
    tsd::core::logStatus("...generating material_orb from embedded data");
    tsd::io::generate_material_orb(tsd.scene);
  } else if (!loadFromState) {
    tsd::io::import_files(tsd.scene, tsd.animationMgr, commandLine.filenames);
    tsd::io::import_animations(
        tsd.scene, tsd.animationMgr, commandLine.animationFilenames);
  }
}

bool Context::logVerbose() const
{
  return m_logging.verbose;
}

void Context::setLogVerbose(bool v)
{
  m_logging.verbose = v;
}

bool Context::logEchoOutput() const
{
  return m_logging.echoOutput;
}

void Context::setLogEchoOutput(bool v)
{
  m_logging.echoOutput = v;
}

void Context::setOfflineRenderingLibrary(const std::string &libName)
{
  auto &dm = this->anari;
  auto d = dm.loadDevice(libName);
  if (!d) {
    tsd::core::logError(
        "[Context] Failed to load ANARI device for offline rendering: %s",
        libName.c_str());
    return;
  }

  this->offline.renderer.rendererObjects.clear();
  this->offline.renderer.activeRenderer = 0;
  this->offline.renderer.libraryName = libName;

  for (auto &name : tsd::scene::getANARIObjectSubtypes(d, ANARI_RENDERER)) {
    auto o = tsd::scene::parseANARIObjectInfo(d, ANARI_RENDERER, name.c_str());
    o.setName(name.c_str());
    this->offline.renderer.rendererObjects.push_back(std::move(o));
  }

  anari::release(d, d);
}

tsd::scene::LayerNodeRef Context::getFirstSelected() const
{
  return tsd.selectedNodes.empty() ? tsd::scene::LayerNodeRef{}
                                   : tsd.selectedNodes[0];
}

const std::vector<tsd::scene::LayerNodeRef> &Context::getSelectedNodes() const
{
  return tsd.selectedNodes;
}

void Context::setSelected(tsd::scene::LayerNodeRef node)
{
  setSelected(std::vector<tsd::scene::LayerNodeRef>{
      node.valid() ? node : tsd::scene::LayerNodeRef{}});
}

void Context::setSelected(const std::vector<tsd::scene::LayerNodeRef> &nodes)
{
  tsd.selectedNodes = nodes;
  tsd.scene.updateDelegate().signalObjectFilteringChanged();
}

void Context::setSelected(const tsd::scene::Object *obj)
{
  if (!obj) {
    clearSelected();
    return;
  }

  // Search all layers for first node referencing this object
  const auto &layers = tsd.scene.layers();
  for (auto &&[layerTk, state] : layers) {
    auto layer = state.ptr.get();
    tsd::scene::LayerNodeRef foundNode;
    layer->traverse_const(layer->root(), [&](const auto &node, int level) {
      if (foundNode.valid())
        return false;
      if (level > 0) {
        auto *nodeObj = node->getObject();
        if (nodeObj == obj) {
          foundNode = layer->at(node.index());
          return false;
        }
      }
      return true;
    });

    if (foundNode.valid()) {
      tsd::core::logStatus(
          "[selection] Selected object %s[%zu] as node %zu on layer %s",
          obj->name().c_str(),
          obj->index(),
          foundNode.index(),
          layerTk);
      setSelected(foundNode);
      return;
    }
  }

  tsd::core::logStatus(
      "[selection] Object not found in any layer, clearing selection");
  clearSelected();
}

void Context::addToSelection(tsd::scene::LayerNodeRef node)
{
  if (!node.valid())
    return;

  for (const auto &selected : tsd.selectedNodes) {
    if (selected == node)
      return;
  }

  tsd.selectedNodes.push_back(node);
  tsd.scene.updateDelegate().signalObjectFilteringChanged();
}

void Context::removeFromSelection(tsd::scene::LayerNodeRef node)
{
  auto it = std::find(tsd.selectedNodes.begin(), tsd.selectedNodes.end(), node);
  if (it != tsd.selectedNodes.end()) {
    tsd.selectedNodes.erase(it);
    tsd.scene.updateDelegate().signalObjectFilteringChanged();
  }
}

bool Context::isSelected(tsd::scene::LayerNodeRef node) const
{
  return std::find(tsd.selectedNodes.begin(), tsd.selectedNodes.end(), node)
      != tsd.selectedNodes.end();
}

void Context::clearSelected()
{
  if (!tsd.selectedNodes.empty()) {
    tsd.selectedNodes.clear();
    tsd.scene.updateDelegate().signalObjectFilteringChanged();
  }
}
std::vector<tsd::scene::LayerNodeRef> Context::getParentOnlySelectedNodes()
    const
{
  std::vector<tsd::scene::LayerNodeRef> parentOnly;

  for (const auto &node : tsd.selectedNodes) {
    if (!node.valid())
      continue;

    bool isChildOfSelected = false;

    // Check if any other selected node is an ancestor of this node
    for (const auto &potentialParent : tsd.selectedNodes) {
      if (!potentialParent.valid() || potentialParent == node)
        continue;

      auto current = node;
      while (current.valid()) {
        auto parentRef = current->parent();
        if (!parentRef.valid())
          break;

        if (parentRef == potentialParent) {
          isChildOfSelected = true;
          break;
        }

        current = parentRef;
      }

      if (isChildOfSelected)
        break;
    }

    if (!isChildOfSelected)
      parentOnly.push_back(node);
  }

  return parentOnly;
}
void Context::addCurrentViewToCameraPoses(const char *_name)
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

void Context::addTurntableCameraPoses(const tsd::math::float3 &azs,
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
    }
  }
}

void Context::updateExistingCameraPoseFromView(CameraPose &p)
{
  auto azel = view.manipulator.azel();
  auto dist = view.manipulator.distance();
  tsd::math::float3 azeldist(azel.x, azel.y, dist);

  p.lookat = view.manipulator.at();
  p.azeldist = azeldist;
  p.upAxis = static_cast<int>(view.manipulator.axis());
}

bool Context::updateCameraPathAnimation()
{
  auto &scene = tsd.scene;

  if (view.poses.size() < 2) {
    tsd::core::logWarning(
        "[camera path] Need at least 2 poses to build animation");
    return false;
  }

  size_t cameraIndex = view.cameraPathCameraIndex;
  if (cameraIndex == TSD_INVALID_INDEX)
    cameraIndex = offline.camera.cameraIndex;

  auto camera = scene.getObject<tsd::scene::Camera>(cameraIndex);
  if (!camera) {
    tsd::core::logWarning("[camera path] No camera selected for animation");
    return false;
  }

  std::vector<tsd::rendering::CameraPose> samples;
  tsd::rendering::buildCameraPathSamples(
      view.poses, view.pathSettings, samples);

  if (samples.empty()) {
    tsd::core::logWarning("[camera path] No samples generated");
    return false;
  }

  offline.frame.numFrames = static_cast<int>(samples.size());
  if (offline.frame.renderSubset) {
    offline.frame.startFrame =
        std::clamp(offline.frame.startFrame, 0, offline.frame.numFrames - 1);
    offline.frame.endFrame =
        std::clamp(offline.frame.endFrame, 0, offline.frame.numFrames - 1);
  }

  // Remove existing camera path animation by name
  auto &anims = tsd.animationMgr.animations();
  for (size_t i = 0; i < anims.size(); i++) {
    if (anims[i].name() == view.cameraPathAnimationName) {
      tsd.animationMgr.removeAnimation(i);
      break;
    }
  }

  view.cameraPathAnimationName = "camera_path";

  // Build time base: linear 0..1
  std::vector<float> timeBase(samples.size());
  for (size_t i = 0; i < samples.size(); i++)
    timeBase[i] = static_cast<float>(i) / (samples.size() - 1);

  std::vector<tsd::math::float3> positions(samples.size());
  std::vector<tsd::math::float3> directions(samples.size());
  std::vector<tsd::math::float3> ups(samples.size());

  tsd::rendering::Manipulator tempManipulator;
  for (size_t i = 0; i < samples.size(); ++i) {
    tempManipulator.setConfig(samples[i]);
    positions[i] = tempManipulator.eye();
    directions[i] = tempManipulator.dir();
    ups[i] = tempManipulator.up();
  }

  const auto firstPosition = positions[0];
  const auto firstDirection = directions[0];
  const auto firstUp = ups[0];

  auto &anim = tsd.animationMgr.addAnimation(view.cameraPathAnimationName);
  anim.addObjectParameterBinding(camera.data(),
      "position",
      ANARI_FLOAT32_VEC3,
      positions.data(),
      timeBase.data(),
      samples.size());
  anim.addObjectParameterBinding(camera.data(),
      "direction",
      ANARI_FLOAT32_VEC3,
      directions.data(),
      timeBase.data(),
      samples.size());
  anim.addObjectParameterBinding(camera.data(),
      "up",
      ANARI_FLOAT32_VEC3,
      ups.data(),
      timeBase.data(),
      samples.size());

  // Seed camera parameters with the first sample for immediate feedback
  camera->setParameter("position", firstPosition);
  camera->setParameter("direction", firstDirection);
  camera->setParameter("up", firstUp);

  tsd::core::logStatus(
      "[camera path] Built animation with %zu samples for camera '%s'",
      samples.size(),
      camera->name().c_str());
  return true;
}

void Context::setCameraPose(const CameraPose &pose)
{
  view.manipulator.setConfig(
      pose.lookat, pose.azeldist.z, {pose.azeldist.x, pose.azeldist.y});
  view.manipulator.setAxis(static_cast<tsd::rendering::UpAxis>(pose.upAxis));
}

void Context::removeAllPoses()
{
  view.poses.clear();
  if (!view.cameraPathAnimationName.empty()) {
    tsd::core::logStatus("[camera path] Clearing camera path animation");
    auto &anims = tsd.animationMgr.animations();
    for (size_t i = 0; i < anims.size(); i++) {
      if (anims[i].name() == view.cameraPathAnimationName) {
        tsd.animationMgr.removeAnimation(i);
        break;
      }
    }
    view.cameraPathAnimationName.clear();
  }
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

  auto &aovRoot = root["aov"];
  aovRoot["aovType"] = static_cast<int>(aov.aovType);
  aovRoot["depthMin"] = aov.depthMin;
  aovRoot["depthMax"] = aov.depthMax;
  aovRoot["edgeInvert"] = aov.edgeInvert;
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
    tsd::scene::Object ro(ANARI_RENDERER, node.name().c_str());
    tsd::io::nodeToObject(node, ro);
    renderer.rendererObjects.push_back(std::move(ro));
  });

  auto &outputRoot = root["output"];
  outputRoot["outputDirectory"].getValue(ANARI_STRING, &output.outputDirectory);
  outputRoot["filePrefix"].getValue(ANARI_STRING, &output.filePrefix);

  auto &aovRoot = root["aov"];
  int aovTypeInt = static_cast<int>(aov.aovType);
  aovRoot["aovType"].getValue(ANARI_INT32, &aovTypeInt);
  aov.aovType = static_cast<tsd::rendering::AOVType>(aovTypeInt);
  aovRoot["depthMin"].getValue(ANARI_FLOAT32, &aov.depthMin);
  aovRoot["depthMax"].getValue(ANARI_FLOAT32, &aov.depthMax);
  aovRoot["edgeInvert"].getValue(ANARI_BOOL, &aov.edgeInvert);
}

} // namespace tsd::app
