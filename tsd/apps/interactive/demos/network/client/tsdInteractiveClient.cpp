// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/DatabaseEditor.h>
#include <tsd/ui/imgui/windows/Log.h>
#include <tsd/ui/imgui/windows/ObjectEditor.h>
#include <tsd/ui/imgui/windows/Viewport.h>
// tsd_io
#include "tsd/io/procedural.hpp"

#include "RemoteViewport.h"

namespace tsd::demo {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

struct Application : public TSDApplication
{
  Application()
  {
    m_client = std::make_shared<tsd::network::NetworkClient>();

    m_client->registerHandler(
        MessageType::ERROR, [](const tsd::network::Message &msg) {
          tsd::core::logError("[Client] Received error from server: '%s'",
              tsd::network::payloadAs<char>(msg));
          std::exit(1);
        });

    m_client->registerHandler(
        MessageType::CLIENT_PING, [](const tsd::network::Message &msg) {
          tsd::core::logStatus("[Client] Received PING from server");
        });

    m_client->registerHandler(MessageType::CLIENT_RECEIVE_VIEW,
        [this](const tsd::network::Message &msg) {
          tsd::core::logStatus("[Client] Received view from server");
          const auto *viewMsg =
              tsd::network::payloadAs<tsd::network::RenderSession::View>(msg);
          auto *core = appCore();
          auto *manipulator = &core->view.manipulator;
          manipulator->setConfig(
              tsd::math::float3(
                  viewMsg->lookat.x, viewMsg->lookat.y, viewMsg->lookat.z),
              viewMsg->azeldist.z,
              tsd::math::float2(viewMsg->azeldist.x, viewMsg->azeldist.y));
        });

    m_client->start();
  }

  ~Application() override
  {
    m_client->stop();
  }

  anari_viewer::WindowArray setupWindows() override
  {
    auto windows = TSDApplication::setupWindows();

    auto *core = appCore();
    auto *manipulator = &core->view.manipulator;
    core->tsd.sceneLoadComplete = true;

    auto *log = new tsd_ui::Log(this);
    m_viewport = new tsd_ui::RemoteViewport(
        this, manipulator, m_client.get(), "Viewport");

    windows.emplace_back(m_viewport);
    windows.emplace_back(log);

    setWindowArray(windows);

    return windows;
  }

  void uiMainMenuBar() override
  {
    // Menu //

    if (ImGui::BeginMenu("Client")) {
      ImGui::BeginDisabled(m_client->isConnected());
      if (ImGui::BeginMenu("Connect")) {
        ImGui::InputText("Host", &m_host);
        int port = m_port;
        if (ImGui::InputInt("Port", &port))
          m_port = static_cast<short>(port);
        if (ImGui::Button("Connect")) {
          m_client->connect(m_host, m_port);
        }
        ImGui::EndMenu();
      } // Connect
      ImGui::EndDisabled();

      ImGui::Separator();

      ImGui::BeginDisabled(!m_client->isConnected());
      if (ImGui::MenuItem("Disconnect", "", false, true)) {
        tsd::core::logStatus("[Client] Disconnecting from server...");
        tsd::core::logWarning("[Client] Disconnect not yet implemented");
      }
      ImGui::EndDisabled();

      ImGui::Separator();

      if (ImGui::MenuItem("Quit", "Esc", false, true)) {
        std::exit(0);
      }
      ImGui::EndMenu();
    }

    ImGui::BeginDisabled(!m_client->isConnected());

    if (ImGui::BeginMenu("Commands")) {
      if (ImGui::MenuItem("Start Rendering")) {
        tsd::core::logStatus("[Client] Sending START_RENDERING command");
        m_client->send(
            tsd::network::make_message(MessageType::SERVER_START_RENDERING));
      }

      if (ImGui::MenuItem("Stop Rendering")) {
        tsd::core::logStatus("[Client] Sending STOP_RENDERING command");
        m_client->send(
            tsd::network::make_message(MessageType::SERVER_STOP_RENDERING));
      }

      if (ImGui::MenuItem("Shutdown")) {
        tsd::core::logStatus("[Client] Sending SHUTDOWN command");
        m_client->send(
            tsd::network::make_message(MessageType::SERVER_SHUTDOWN));
      }

      ImGui::EndMenu();
    }

    ImGui::EndDisabled();

    // Keyboard shortcuts //

    if (ImGui::IsKeyPressed(ImGuiKey_P, false)) {
      tsd::core::logStatus("[Client] Sending PING");
      m_client->send(tsd::network::make_message(MessageType::SERVER_PING));
    }
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,26
Size=1920,1105
Collapsed=0

[Window][Viewport]
Pos=549,26
Size=1371,857
Collapsed=0
DockId=0x00000003,0

[Window][Secondary View]
Pos=1237,26
Size=683,857
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=549,885
Size=1371,246
Collapsed=0
DockId=0x0000000A,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Layers]
Pos=0,25
Size=548,347
Collapsed=0
DockId=0x00000005,0

[Window][Object Editor]
Pos=0,609
Size=547,522
Collapsed=0
DockId=0x00000008,0

[Window][Scene Controls]
Pos=0,26
Size=547,581
Collapsed=0
DockId=0x00000007,0

[Window][Database Editor]
Pos=0,609
Size=547,522
Collapsed=0
DockId=0x00000008,1

[Table][0x39E9F5ED,1]
Column 0  Weight=1.0000

[Table][0x418F6C9E,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xE57DC2D0,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x65B57849,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xE53C80DF,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x7FC3FA09,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xA96A74B3,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xC00D0D97,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x782A6D6B Pos=0,25 Size=1920,1054 Split=X Selected=0x13926F0B
  DockNode      ID=0x00000005 Parent=0x782A6D6B SizeRef=548,626 Selected=0x1FD98235
  DockNode      ID=0x00000006 Parent=0x782A6D6B SizeRef=1370,626 CentralNode=1 Selected=0x13926F0B
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1105 Split=X
  DockNode      ID=0x00000001 Parent=0x80F5B4C5 SizeRef=547,1054 Split=Y Selected=0x6426B955
    DockNode    ID=0x00000007 Parent=0x00000001 SizeRef=547,581 Selected=0x6426B955
    DockNode    ID=0x00000008 Parent=0x00000001 SizeRef=547,522 Selected=0x82B4C496
  DockNode      ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1371,1054 Split=Y Selected=0xC450F867
    DockNode    ID=0x00000009 Parent=0x00000002 SizeRef=1371,806 Split=X Selected=0xC450F867
      DockNode  ID=0x00000003 Parent=0x00000009 SizeRef=686,857 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x00000004 Parent=0x00000009 SizeRef=683,857 Selected=0xA3219422
    DockNode    ID=0x0000000A Parent=0x00000002 SizeRef=1371,246 Selected=0x139FDA3F
)layout";
  }

 private:
  tsd::ui::imgui::RemoteViewport *m_viewport{nullptr};
  std::shared_ptr<tsd::network::NetworkClient> m_client;
  std::string m_host{"127.0.0.1"};
  short m_port{12345};
};

} // namespace tsd::demo

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  {
    tsd::demo::Application app;
    app.run(1920, 1080, "TSD Interactive Client");
  }

  return 0;
}
