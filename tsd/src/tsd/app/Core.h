// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/scene/Scene.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/pipeline/passes/VisualizeAOVPass.h"
#include "tsd/rendering/view/Manipulator.hpp"
// std
#include <map>
#include <string>
#include <utility>

#include "tsd/app/TaskQueue.h"
#include "tsd/app/renderAnimationSequence.h"

namespace tsd::ui::imgui {
struct BlockingTaskModal;
struct ImportFileDialog;
struct ExportNanoVDBFileDialog;
} // namespace tsd::ui::imgui

namespace tsd::app {

struct Core;

using CameraPose = tsd::rendering::CameraPose;
using DeviceInitParam = std::pair<std::string, tsd::core::Any>;

enum class ImporterType
{
  AGX,
  ASSIMP,
  ASSIMP_FLAT,
  AXYZ,
  DLAF,
  E57XYZ,
  GLTF,
  HDRI,
  HSMESH,
  NBODY,
  OBJ,
  PDB,
  PLY,
  POINTSBIN_MULTIFILE,
  PT,
  SILO,
  SMESH,
  SMESH_ANIMATION, // time series version
  SWC,
  TRK,
  USD,
  USD2,
  XYZDP,
  VOLUME,
  TSD,
  XF, // Special case for transfer function files
      // Not an actual scene importer, but used to set transfer function from
      // CLI
  BLANK, // Must be last import type before 'NONE'
  NONE
};

using ImportFile = std::pair<ImporterType, std::string>;
using ImportAnimationFiles = std::pair<ImporterType, std::vector<std::string>>;

struct CommandLineOptions
{
  bool useDefaultLayout{true};
  bool useDefaultRenderer{true};
  bool loadingScene{false};
  bool preloadDevices{false};
  bool loadedFromStateFile{false};
  std::string stateFile;
  std::string currentLayerName{"default"};
  std::vector<ImportFile> filenames;
  std::vector<ImportAnimationFiles> animationFilenames;
  std::vector<tsd::core::Token> animationLayerNames;
  ImportAnimationFiles *currentAnimationSequence{nullptr};
  ImporterType importerType{ImporterType::NONE};
  std::vector<std::string> libraryList;
  std::string secondaryViewportLibrary;
  std::string cameraFile;
};

struct TSDState
{
  struct StashedSelection
  {
    std::vector<tsd::core::LayerNodeRef> nodes;
    bool shouldDeleteAfterPaste{false};
  };

  tsd::core::Scene scene;
  bool sceneLoadComplete{false};
  std::vector<tsd::core::LayerNodeRef> selectedNodes;
  StashedSelection stashedSelection;
};

struct ANARIDeviceManager
{
  ANARIDeviceManager(const bool *verboseFlag = nullptr);

  anari::Device loadDevice(const std::string &libName,
      const std::vector<DeviceInitParam> &initialDeviceParams = {});

  const anari::Extensions *loadDeviceExtensions(const std::string &libName);
  tsd::rendering::RenderIndex *acquireRenderIndex(
      tsd::core::Scene &c, anari::Device device);
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

  // Settings //

  struct Settings
  {
    // Use flat render index by default, unless set otherwise
    // This is to avoid issues with instancing in the scene graph
    // and to allow for faster rendering in some cases.
    bool forceFlat{false};
  } m_settings;
};

struct LogState
{
  bool verbose{false};
  bool echoOutput{false};
};

struct CameraState
{
  std::vector<CameraPose> poses;
  tsd::rendering::Manipulator manipulator;
};

struct ImporterState
{
  core::TransferFunction transferFunction;
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
    std::vector<tsd::core::Object> rendererObjects;
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

struct Windows
{
  tsd::ui::imgui::BlockingTaskModal *taskModal{nullptr};
  tsd::ui::imgui::ImportFileDialog *importDialog{nullptr};
  tsd::ui::imgui::ExportNanoVDBFileDialog *exportNanoVDBDialog{nullptr};
  float fontScale{1.f};
  float uiRounding{9.f};
};

struct Tasking
{
  TaskQueue queue{10};
};

struct Core
{
  CommandLineOptions commandLine;
  TSDState tsd;
  ANARIDeviceManager anari;
  LogState logging;
  CameraState view;
  ImporterState importer;
  OfflineRenderSequenceConfig offline;
  Windows windows;
  Tasking jobs;

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  Core();
  ~Core();

  void parseCommandLine(int argc, const char **argv);
  void setupSceneFromCommandLine(bool hdriOnly = false);
  void importFile(const ImportFile &file, tsd::core::LayerNodeRef root = {});
  void importFiles(
      const std::vector<ImportFile> &files, tsd::core::LayerNodeRef root = {});
  void importAnimations(const std::vector<ImportAnimationFiles> &files,
      tsd::core::LayerNodeRef root = {});

  // Offline rendering //

  void setOfflineRenderingLibrary(const std::string &libName);
  void renderOfflineAnimationSequence(RenderSequenceCallback cb = {});

  // Selection //

  tsd::core::LayerNodeRef getFirstSelected() const;
  const std::vector<tsd::core::LayerNodeRef> &getSelectedNodes() const;
  void setSelected(tsd::core::LayerNodeRef node);
  void setSelected(const std::vector<tsd::core::LayerNodeRef> &nodes);
  void setSelected(const tsd::core::Object *obj);
  void addToSelection(tsd::core::LayerNodeRef node);
  void removeFromSelection(tsd::core::LayerNodeRef node);
  bool isSelected(tsd::core::LayerNodeRef node) const;
  void clearSelected();

  // Returns only parent nodes from selection (filters out children of selected
  // nodes)
  std::vector<tsd::core::LayerNodeRef> getParentOnlySelectedNodes() const;

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

  // Not copyable or moveable //
  Core(const Core &) = delete;
  Core(Core &&) = delete;
  Core &operator=(const Core &) = delete;
  Core &operator=(Core &&) = delete;
  //////////////////////////////
};

void anariStatusFunc(const void *_core,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message);

} // namespace tsd::app
