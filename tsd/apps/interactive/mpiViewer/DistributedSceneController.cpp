// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DistributedSceneController.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// std
#include <random>

namespace tsd::mpi_viewer {

DistributedSceneController::DistributedSceneController() = default;

DistributedSceneController::~DistributedSceneController()
{
  shutdown();
}

void DistributedSceneController::initialize(int argc, const char **argv)
{
  {
    int rank = 0, numRanks = 1;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    m_core = std::make_unique<tsd::app::Core>();
    m_core->tsd.scene.setMpiRankInfo(rank, numRanks);
    m_distributedState = std::make_unique<DistributedState>();
    m_mpiInitialized = true;
  }

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-ptc") {
      m_anari.libraryName = "ptc";
      argv[i] = nullptr; // remove from args passed to tsd::app::Core
    } else if (arg == "-gpn" || arg == "--gpusPerNode") {
      m_anari.gpusPerNode = std::stoi(argv[++i]);
      argv[i] = nullptr; // remove from args passed to tsd::app::Core
      argv[i - 1] = nullptr;
    }
  }

  m_core->parseCommandLine(argc, argv);

  // Load scene files assigned to this rank //

  auto &filenames = m_core->commandLine.filenames;
  for (size_t i = 0; i < filenames.size(); i++) {
    if (numRanks() > 1 && (i % numRanks() != rank()))
      continue;
    m_core->importFile(filenames[i]);
    tsd::core::logStatus(
        "[DistributedSceneController] "
        "rank '%i' loaded file '%s'",
        rank(),
        filenames[i].second.c_str());
  }

  // Setup ANARI state //

  std::vector<tsd::app::DeviceInitParam> deviceParameters;
  if (m_anari.gpusPerNode > 1) {
    auto cudaDeviceId = int(rank() % m_anari.gpusPerNode);
    deviceParameters.push_back({"cudaDevice", tsd::core::Any(cudaDeviceId)});
    deviceParameters.push_back({"dataGroupID", tsd::core::Any(-1)});
  }

  auto d = m_anari.device =
      m_core->anari.loadDevice(m_anari.libraryName, deviceParameters);

  auto f = m_anari.frame = anari::newObject<anari::Frame>(d);
  m_anari.camera = anari::newObject<anari::Camera>(d, "perspective");
  m_anari.renderer = anari::newObject<anari::Renderer>(d, "default");
  m_anari.hdriLight = anari::newObject<anari::Light>(d, "hdri");

  auto &scene = m_core->tsd.scene;
  auto *ri = m_anari.renderIndex = m_core->anari.acquireRenderIndex(scene, d);

  anari::setParameter(d, f, "camera", m_anari.camera);
  anari::setParameter(d, f, "renderer", m_anari.renderer);
  anari::setParameter(d, f, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, f, "world", ri->world());
  anari::commitParameters(d, f);

  // Randomize default material color //

  std::mt19937 rng;
  rng.seed(rank());
  std::normal_distribution<float> dist(0.2f, 0.8f);

  auto mat = m_core->tsd.scene.defaultMaterial();
  mat->setParameter(
      "color", tsd::math::float3(dist(rng), dist(rng), dist(rng)));

  // Invalidate distributed state to force initial sync //

  m_distributedState->lights.write();
  m_distributedState->camera.write();
  m_distributedState->renderer.write();
  m_distributedState->frame.write();
}

void DistributedSceneController::executeFrame()
{
  if (!m_mpiInitialized) {
    tsd::core::logError("DistributedSceneController not initialized!\n");
    return;
  }
  executeFrame_syncFrame();
  if (!isRunning())
    return;
  executeFrame_syncLights();
  executeFrame_syncCamera();
  executeFrame_syncRenderer();
  executeFrame_syncAnimation();
  executeFrame_render();
}

void DistributedSceneController::signalStop()
{
  if (!isMain()) {
    tsd::core::logError(
        "[DistributedSceneController] "
        "signalStop() can only be called from main rank!");
    return;
  }
  m_distributedState->frame.write()->running = false;
  executeFrame(); // render extra frame to stay in step with workers
}

void DistributedSceneController::shutdown()
{
  if (!m_mpiInitialized)
    return;

  tsd::core::logDebug(
      "[DistributedSceneController] shutting down rank '%i'...\n", rank());

  m_mpiInitialized = false;

  MPI_Barrier(MPI_COMM_WORLD);

  auto d = m_anari.device;
  m_core->anari.releaseRenderIndex(d);
  anari::release(d, m_anari.hdriLight);
  anari::release(d, m_anari.renderer);
  anari::release(d, m_anari.camera);
  anari::release(d, m_anari.frame);
  m_core->anari.releaseAllDevices();

  MPI_Finalize();
}

bool DistributedSceneController::isRunning() const
{
  return m_distributedState->frame.read()->running;
}

tsd::app::Core *DistributedSceneController::appCore()
{
  return m_core.get();
}

DistributedState *DistributedSceneController::distributedState()
{
  return m_distributedState.get();
}

int DistributedSceneController::rank() const
{
  return m_core->tsd.scene.mpiRank();
}

int DistributedSceneController::numRanks() const
{
  return m_core->tsd.scene.mpiNumRanks();
}

bool DistributedSceneController::isMain() const
{
  return rank() == 0;
}

const char *DistributedSceneController::anariLibraryName() const
{
  return m_anari.libraryName.c_str();
}

anari::Device DistributedSceneController::anariDevice() const
{
  return m_anari.device;
}

anari::Camera DistributedSceneController::anariCamera() const
{
  return m_anari.camera;
}

anari::Renderer DistributedSceneController::anariRenderer() const
{
  return m_anari.renderer;
}

anari::Frame DistributedSceneController::anariFrame() const
{
  return m_anari.frame;
}

anari::Light DistributedSceneController::anariHDRILight() const
{
  return m_anari.hdriLight;
}

void DistributedSceneController::executeFrame_syncLights()
{
  rank_printf(rank(), "synchronizing lights...\n");
  if (auto d = anariDevice(); d && m_distributedState->lights.sync()) {
    auto *state = m_distributedState->lights.read();
#if 0
    auto l = anariHDRILight();
    anari::setParameter(d, l, "up", state->hdriUp);
    anari::setParameter(d, l, "direction", state->hdriDirection);
    anari::setParameter(d, l, "scale", state->hdriScale);
    anari::commitParameters(d, l);
#endif
    rank_printf(rank(), "    -> synchronized lights\n");
  }
}

void DistributedSceneController::executeFrame_syncCamera()
{
  rank_printf(rank(), "synchronizing camera...\n");
  if (auto d = anariDevice(); d && m_distributedState->camera.sync()) {
    auto *state = m_distributedState->camera.read();
    auto c = anariCamera();
    anari::setParameter(d, c, "position", state->position);
    anari::setParameter(d, c, "direction", state->direction);
    anari::setParameter(d, c, "up", state->up);
    anari::setParameter(d, c, "aspect", state->aspect);
    anari::setParameter(d, c, "fovy", state->fovy);
    anari::setParameter(d, c, "apertureRadius", state->apertureRadius);
    anari::setParameter(d, c, "focusDistance", state->focusDistance);
    anari::commitParameters(d, c);
    rank_printf(rank(), "    -> synchronized camera\n");
  }
}

void DistributedSceneController::executeFrame_syncRenderer()
{
  rank_printf(rank(), "synchronizing renderer...\n");
  if (auto d = anariDevice(); d && m_distributedState->renderer.sync()) {
    auto *state = m_distributedState->renderer.read();
    auto r = anariRenderer();
    anari::setParameter(d, r, "background", state->background);
    anari::setParameter(d, r, "ambientColor", state->ambientColor);
    anari::setParameter(d, r, "ambientRadiance", state->ambientRadiance);
    anari::setParameter(d, r, "denoise", state->denoise);
    anari::commitParameters(d, r);
    rank_printf(rank(), "    -> synchronized renderer\n");
  }
}

void DistributedSceneController::executeFrame_syncFrame()
{
  rank_printf(rank(), "synchronizing frame...\n");
  if (auto d = anariDevice(); d && m_distributedState->frame.sync()) {
    auto *state = m_distributedState->frame.read();
    auto f = anariFrame();
    anari::setParameter(d, f, "size", tsd::math::uint2(state->size));
    anari::commitParameters(d, f);
    rank_printf(rank(), "    -> synchronized frame\n");
  }
}

void DistributedSceneController::executeFrame_syncAnimation()
{
  rank_printf(rank(), "synchronizing animation...\n");
  if (auto d = anariDevice(); d && m_distributedState->animation.sync()) {
    auto *state = m_distributedState->animation.read();
    auto *core = appCore();
    core->tsd.scene.setAnimationTime(state->time);
    rank_printf(rank(), "    -> synchronized animation\n");
  }
}

void DistributedSceneController::executeFrame_render()
{
  MPI_Barrier(MPI_COMM_WORLD);
  rank_printf(rank(), "**** rendering frame ****\n");
  if (auto d = anariDevice(); d) {
    auto f = anariFrame();
    anari::render(d, f);
    anari::wait(d, f);
    rank_printf(rank(), "    -> rendered frame\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace tsd::mpi_viewer