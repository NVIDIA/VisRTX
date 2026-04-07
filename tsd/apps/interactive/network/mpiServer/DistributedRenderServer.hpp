// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
// tsd_network
#include "tsd/network/NetworkChannel.hpp"
// tsd_app
#include "tsd/app/Context.h"
// tsd_rendering
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/pipeline/passes/AnariSceneRenderPass.h"

#include "../RenderSession.hpp"

namespace tsd::network {

/*
 * MPI-distributed headless render server that exposes the same TCP network
 * interface as tsdServer (port 12345).  Rank 0 is the network-facing host;
 * all other ranks participate in distributed rendering.
 *
 * - Camera / renderer / frame / animation state: replicated to every rank
 *   each frame via ReplicatedObject<ControlState> + MPI_Bcast.
 * - Scene object mutations (geometry, material, etc.): queued by rank 0 and
 *   broadcast each frame; only ops addressed to a rank are applied there.
 * - Startup scene files: loaded round-robin across ranks (like mpiViewer).
 * - Client-created objects (SERVER_ADD_OBJECT): always assigned to rank 0.
 *
 * Build requirements: TSD_USE_MPI=ON and TSD_USE_NETWORKING=ON.
 */
struct DistributedRenderServer
{
  DistributedRenderServer(int argc, const char **argv);
  ~DistributedRenderServer();

  void run(short port = 12345);

 private:
  enum class ServerMode : int
  {
    DISCONNECTED = 0,
    PAUSED,
    RENDERING,
    SEND_SCENE,
    SHUTDOWN
  };

  // MPI helpers
  int rank() const;
  int numRanks() const;
  bool isMain() const;

  // Setup (all ranks unless noted)
  void setup_Scene();
  void setup_ANARIDevice();
  void setup_Camera();
  void setup_ImagePipeline();
  void setup_Messaging(); // rank 0 only

  // Per-frame sync (all ranks)
  void syncControlState();
  void syncSceneOps();
  void renderFrame();
  void send_FrameBuffer(); // rank 0 only

  // Rank-0 helpers
  void signalServerMode(ServerMode mode);
  int determineTargetRank(const Message &msg);
  void enqueueSceneOp(const Message &msg, int targetRank);

  ///////////////////////////////////////////////////////////////////////////
  // Data
  ///////////////////////////////////////////////////////////////////////////

  short m_port{12345};
  bool m_mpiInitialized{false};
  std::string m_libName;

  tsd::app::Context m_ctx;
  anari::Device m_device{nullptr};
  tsd::rendering::RenderIndex *m_renderIndex{nullptr};
  tsd::rendering::ImagePipeline m_renderPipeline;
  tsd::rendering::AnariSceneRenderPass *m_sceneImagePass{nullptr};
  tsd::scene::CameraAppRef m_camera;
  std::vector<tsd::scene::RendererAppRef> m_renderers;

  // Network (rank 0 only)
  RenderSession m_session;
  std::shared_ptr<NetworkServer> m_server;
  MessageFuture m_lastSentFrame;
  std::mutex m_frameSendMutex;
  std::atomic<int> m_pendingServerMode{int(ServerMode::DISCONNECTED)};
  ServerMode m_previousMode{ServerMode::DISCONNECTED};

  // Distributed control state (POD, replicated via MPI_Bcast each frame)
  struct ControlState
  {
    tsd::math::int2 frameSize{800, 600};
    float animationTime{0.f};
    int serverMode{int(ServerMode::DISCONNECTED)};
    uint64_t cameraIndex{0};
    uint64_t rendererIndex{0};
  };
  static_assert(std::is_trivially_copyable_v<ControlState>,
      "ControlState must be trivially copyable for MPI broadcast");
  static_assert(std::is_standard_layout_v<ControlState>,
      "ControlState must be standard layout for MPI broadcast");

  // DistState is constructed after MPI_Init (ReplicatedObject reads rank in
  // ctor)
  struct DistState;
  std::unique_ptr<DistState> m_distState;
  std::mutex m_controlMutex; // guards ControlState writes from network thread

  // Scene op queue: rank 0 fills from network thread; all ranks drain each
  // frame
  struct PendingSceneOp
  {
    uint8_t messageType;
    int32_t targetRank; // -1 = all ranks, N = specific rank
    std::vector<std::byte> payload;
  };
  std::mutex m_opsMutex;
  std::vector<PendingSceneOp> m_pendingOps;
};

} // namespace tsd::network
