// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include "tsd/ui/imgui/tsd_ui_imgui.h"
#include "tsd/ui/imgui/windows/BaseViewport.h"
// tsd_rendering
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/view/Manipulator.hpp"
// tsd_network
#include "tsd/network/NetworkChannel.hpp"

#include "../RenderSession.hpp"
// std
#include <mutex>

using tsd::network::MessageType;

namespace tsd::ui::imgui {

struct RemoteViewport : public BaseViewport
{
  RemoteViewport(Application *app,
      tsd::rendering::Manipulator *m,
      tsd::network::NetworkChannel *c,
      const char *name = "Remote Viewport");
  ~RemoteViewport();

  void buildUI() override;
  void setManipulator(tsd::rendering::Manipulator *m);
  void setNetworkChannel(tsd::network::NetworkChannel *c);
  void disconnect();

 private:
  void imagePipeline_populate(tsd::rendering::ImagePipeline &p) override;

  void camera_resetView(bool resetAzEl = true) override;
  void camera_centerView() override;

  void renderer_resetParameterDefaults() override;

  void viewport_reshape(tsd::math::int2 newWindowSize) override;

  void updateRenderer();
  void updateCamera();
  void applyIncomingFrame();

  void ui_menubar();
  void ui_overlay();

  // Data /////////////////////////////////////////////////////////////////////

  bool m_wasConnected{false};
  bool m_showOverlay{true};

  size_t m_receivedRendererIdx{TSD_INVALID_INDEX};
  size_t m_receivedCameraIdx{TSD_INVALID_INDEX};
  tsd::scene::RendererAppRef m_prevRenderer;
  tsd::scene::CameraAppRef m_prevCamera;

  // Camera manipulator //

  bool m_manipulatorSynchronized{false};

  // Networking //

  tsd::network::NetworkChannel *m_channel{nullptr};

  // Display //

  std::mutex m_incomingFrameMutex;
  std::vector<uint8_t> m_incomingColorBuffer;
  std::vector<uint8_t> m_pendingColorBuffer;
  bool m_hasPendingFrame{false};
  tsd::network::RenderSession::Frame::Config m_frameConfig;

  tsd::rendering::ClearBuffersPass *m_clearPass{nullptr};
  tsd::rendering::CopyToColorBufferPass *m_incomingFramePass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};
};

} // namespace tsd::ui::imgui
