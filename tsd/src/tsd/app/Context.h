// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_animation
#include "tsd/animation/AnimationManager.hpp"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/scene/Scene.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/pipeline/passes/VisualizeAOVPass.h"
#include "tsd/rendering/view/CameraPath.h"
#include "tsd/rendering/view/Manipulator.hpp"
// tsd_io
#include "tsd/io/importers.hpp"

#include "tsd/app/ANARIDeviceManager.h"
#include "tsd/app/renderAnimationSequence.h"

namespace tsd::app {

using CameraPose = tsd::rendering::CameraPose;
using DeviceInitParam = std::pair<std::string, tsd::core::Any>;

struct CommandLineOptions
{
  bool loadedFromStateFile{false};
  std::string stateFile;
  std::string currentLayerName{"default"};
  std::vector<tsd::io::ImportFile> filenames;
  std::vector<tsd::io::ImportAnimationFiles> animationFilenames;
  std::vector<tsd::core::Token> animationLayerNames;
  tsd::io::ImportAnimationFiles *currentAnimationSequence{nullptr};
  tsd::io::ImporterType importerType{tsd::io::ImporterType::NONE};
  std::string cameraFile;
  std::vector<std::string> ensightFields;
};

struct TSDState
{
  struct StashedSelection
  {
    std::vector<tsd::scene::LayerNodeRef> nodes;
    bool shouldDeleteAfterPaste{false};
  };

  TSDState();

  // NOTE(jda) - FIX: scene must be declared before animation manager since the
  // manager needs a pointer to it, and animation manager must be destroyed
  // first...
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animationMgr;
  bool sceneLoadComplete{false};
  std::vector<tsd::scene::LayerNodeRef> selectedNodes;
  StashedSelection stashedSelection;
};

struct CameraState
{
  std::vector<CameraPose> poses;
  tsd::rendering::Manipulator manipulator;
  tsd::rendering::CameraPathSettings pathSettings;
  size_t cameraPathCameraIndex{TSD_INVALID_INDEX};
  std::string cameraPathAnimationName;
};

struct OfflineRenderSequenceConfig
{
  struct FrameSettings
  {
    uint32_t width{1024};
    uint32_t height{768};
    anari::DataType colorFormat{ANARI_UFIXED8_RGBA_SRGB};
    uint32_t samples{128};
    int numFrames{1};
    bool renderSubset{false}; // use start/end
    int startFrame{0};
    int endFrame{0};
    int frameIncrement{1};
  } frame;

  struct CameraSettings
  {
    float apertureRadius{0.f};
    float focusDistance{1.f};
    size_t cameraIndex{TSD_INVALID_INDEX};
  } camera;

  struct RenderSettings
  {
    std::vector<tsd::scene::Object> rendererObjects;
    int activeRenderer{-1};
    std::string libraryName;
  } renderer;

  struct OutputSettings
  {
    std::string outputDirectory{"./"};
    std::string filePrefix{"frame_"};
  } output;

  struct AOVSettings
  {
    tsd::rendering::AOVType aovType{tsd::rendering::AOVType::NONE};
    float depthMin{0.f};
    float depthMax{1.f};
    float edgeThreshold{0.5f};
    bool edgeInvert{false};
  } aov;

  void saveSettings(tsd::core::DataNode &root);
  void loadSettings(tsd::core::DataNode &root);
};

struct Context
{
  CommandLineOptions commandLine;
  TSDState tsd;
  ANARIDeviceManager anari;
  CameraState view;
  OfflineRenderSequenceConfig offline;

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  Context();
  ~Context();

  // Command line parsing //

  void parseCommandLine(int argc, const char **argv); // raw main() arguments
  void parseCommandLine(std::vector<std::string> &args); // removes used args
  void setupSceneFromCommandLine(bool hdriOnly = false);

  // Logging //

  bool logVerbose() const;
  void setLogVerbose(bool v);
  bool logEchoOutput() const;
  void setLogEchoOutput(bool v);

  // Offline rendering //

  void setOfflineRenderingLibrary(const std::string &libName);

  // Selection //

  tsd::scene::LayerNodeRef getFirstSelected() const;
  const std::vector<tsd::scene::LayerNodeRef> &getSelectedNodes() const;
  void setSelected(tsd::scene::LayerNodeRef node);
  void setSelected(const std::vector<tsd::scene::LayerNodeRef> &nodes);
  void setSelected(const tsd::scene::Object *obj);
  void addToSelection(tsd::scene::LayerNodeRef node);
  void removeFromSelection(tsd::scene::LayerNodeRef node);
  bool isSelected(tsd::scene::LayerNodeRef node) const;
  void clearSelected();

  // Returns only parent nodes from selection (filters out children of selected
  // nodes)
  std::vector<tsd::scene::LayerNodeRef> getParentOnlySelectedNodes() const;

  // Camera poses //

  void addCurrentViewToCameraPoses(const char *name = "");
  void addTurntableCameraPoses(
      const tsd::math::float3 &azimuths, // begin, end, step
      const tsd::math::float3 &elevations, // begin, end, step
      const tsd::math::float3 &center,
      float distance,
      const char *name = "");
  void updateExistingCameraPoseFromView(CameraPose &p);
  void setCameraPose(const CameraPose &pose);
  void removeAllPoses();
  bool updateCameraPathAnimation();

  TSD_NOT_COPYABLE(Context)
  TSD_NOT_MOVEABLE(Context)

 private:
  struct LogState
  {
    bool verbose{false};
    bool echoOutput{false};
  } m_logging;
};

} // namespace tsd::app
