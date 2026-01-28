// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include "tsd/ui/imgui/tsd_ui_imgui.h"
#include "tsd/ui/imgui/windows/Window.h"
// tsd_rendering
#include "tsd/rendering/pipeline/RenderPipeline.h"
#include "tsd/rendering/view/Manipulator.hpp"
// tsd_network
#include "tsd/network/NetworkChannel.hpp"

#include "CopyToColorBufferPass.hpp"

#include "../RenderSession.hpp"

using tsd::network::MessageType;

namespace tsd::ui::imgui {

struct RemoteViewport : public Window
{
  RemoteViewport(Application *app,
      tsd::rendering::Manipulator *m,
      tsd::network::NetworkChannel *c,
      const char *name = "Remote Viewport");
  ~RemoteViewport();

  void buildUI() override;
  void setManipulator(tsd::rendering::Manipulator *m);
  void setNetworkChannel(tsd::network::NetworkChannel *c);

 private:
  void saveSettings(tsd::core::DataNode &thisWindowRoot) override;
  void loadSettings(tsd::core::DataNode &thisWindowRoot) override;

  void setupRenderPipeline();
  void reshape(tsd::math::int2 newWindowSize);

  void updateCamera();

  void ui_menubar();
  void ui_handleInput();
  void ui_overlay();

  int windowFlags() const override; // anari_viewer::Window

  // Data /////////////////////////////////////////////////////////////////////

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_wasConnected{false};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_showOverlay{true};
  bool m_showCameraInfo{false};

  float m_fov{40.f};

  // Camera manipulator //

  int m_arcballUp{1};
  tsd::rendering::Manipulator m_localArcball;
  tsd::rendering::Manipulator *m_arcball{nullptr};
  tsd::rendering::UpdateToken m_cameraToken{0};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // Networking //

  tsd::network::NetworkChannel *m_channel{nullptr};

  // Display //

  std::vector<uint8_t> m_incomingColorBuffer;

  tsd::rendering::RenderPipeline m_pipeline;
  tsd::rendering::ClearBuffersPass *m_clearPass{nullptr};
  tsd::rendering::CopyToColorBufferPass *m_incomingFramePass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};

  tsd::math::int2 m_viewportSize{0, 0};
};

} // namespace tsd::ui::imgui
