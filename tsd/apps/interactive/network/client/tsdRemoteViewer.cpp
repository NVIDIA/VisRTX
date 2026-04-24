// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/DatabaseEditor.h>
#include <tsd/ui/imgui/windows/LayerTree.h>
#include <tsd/ui/imgui/windows/Log.h>
#include <tsd/ui/imgui/windows/ObjectEditor.h>
#include <tsd/ui/imgui/windows/Viewport.h>
// tsd_io
#include <tsd/io/serialization.hpp>
// tsd_network
#include <tsd/network/messages/TransferScene.hpp>

#include "NetworkUpdateDelegate.hpp"
#include "RemoteViewport.h"

namespace tsd::demo {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

struct Application : public TSDApplication
{
  Application();
  ~Application() override;

  anari_viewer::WindowArray setupWindows() override;
  void uiMainMenuBar() override;
  void teardown() override;
  const char *getDefaultLayout() const override;

 private:
  void connect();
  void disconnect();

  tsd::network::NetworkUpdateDelegate *m_updateDelegate{nullptr};
  tsd::ui::imgui::RemoteViewport *m_viewport{nullptr};
  std::shared_ptr<tsd::network::NetworkClient> m_client;
  std::string m_host{"127.0.0.1"};
  short m_port{12345};
  std::string m_stateFileName{"state.tsd"};
  bool m_timeUpdatesEnabled{true};
};

// Application definitions ////////////////////////////////////////////////////

Application::Application()
{
  auto *ctx = appContext();

  m_client = std::make_shared<tsd::network::NetworkClient>();

  m_updateDelegate =
      ctx->tsd.scene.updateDelegate().emplace<tsd::network::NetworkUpdateDelegate>(
          &ctx->tsd.scene, m_client.get());

  ctx->tsd.animationMgr.setTimeChangedCallback([this](float time) {
    if (m_timeUpdatesEnabled)
      m_client->send(MessageType::SERVER_UPDATE_TIME, &time);
  });

  m_client->registerHandler(
      MessageType::ERROR, [](const tsd::network::Message &msg) {
        tsd::core::logError("[Client] Received error from server: '%s'",
            tsd::network::payloadAs<char>(msg));
        std::exit(1);
      });

  m_client->registerHandler(
      MessageType::PING, [](const tsd::network::Message &msg) {
        tsd::core::logStatus("[Client] Received PING from server");
      });

  m_client->registerHandler(MessageType::CLIENT_SCENE_TRANSFER_BEGIN,
      [this](const tsd::network::Message &msg) {
        tsd::core::logStatus("[Client] Server has initiated scene transfer...");
        m_updateDelegate->setEnabled(false);
      });

  m_client->registerHandler(MessageType::CLIENT_RECEIVE_SCENE,
      [this](const tsd::network::Message &msg) {
        auto &scene = appContext()->tsd.scene;
        tsd::network::messages::TransferScene sceneMsg(msg, &scene);
        sceneMsg.execute();
        m_updateDelegate->setEnabled(true);
        tsd::core::logStatus("[Client] Scene contents:");
        tsd::core::logStatus(
            "\n%s", tsd::scene::objectDBInfo(scene.objectDB()).c_str());
        tsd::core::logStatus("[Client] Requesting start of rendering...");
        m_client->send(MessageType::SERVER_START_RENDERING);
        appContext()->tsd.sceneLoadComplete = true;
      });

  m_client->registerHandler(MessageType::CLIENT_RECEIVE_TIME,
      [this](const tsd::network::Message &msg) {
        m_timeUpdatesEnabled = false;
        float time = *tsd::network::payloadAs<float>(msg);
        appContext()->tsd.animationMgr.setAnimationTime(time);
        m_timeUpdatesEnabled = true;
      });

  ctx->tsd.sceneLoadComplete = false;
}

Application::~Application()
{
  if (m_updateDelegate)
    appContext()->tsd.scene.updateDelegate().erase(m_updateDelegate);
}

anari_viewer::WindowArray Application::setupWindows()
{
  auto windows = TSDApplication::setupWindows();

  auto *ctx = appContext();
  auto *manipulator = &ctx->view.manipulator;

  auto *log = new tsd_ui::Log(this);
  m_viewport =
      new tsd_ui::RemoteViewport(this, manipulator, m_client.get(), "Viewport");
  auto *ltree = new tsd_ui::LayerTree(this);
  auto *oeditor = new tsd_ui::ObjectEditor(this);
  auto *dbeditor = new tsd_ui::DatabaseEditor(this);

  windows.emplace_back(m_viewport);
  windows.emplace_back(log);
  windows.emplace_back(ltree);
  windows.emplace_back(dbeditor);
  windows.emplace_back(oeditor);

  setWindowArray(windows);

  return windows;
}

void Application::uiMainMenuBar()
{
  // Menu //

  if (ImGui::BeginMenu("Client")) {
    ImGui::BeginDisabled(m_client->isConnected());
    if (ImGui::BeginMenu("Connect")) {
      ImGui::InputText("Host", &m_host);

      int port = m_port;
      if (ImGui::InputInt("Port", &port))
        m_port = static_cast<short>(port);

      if (ImGui::Button("Connect"))
        connect();

      ImGui::EndMenu();
    } // Connect
    ImGui::EndDisabled();

    ImGui::Separator();

    ImGui::BeginDisabled(!m_client->isConnected());
    if (ImGui::MenuItem("Disconnect", "", false, true))
      disconnect();
    ImGui::EndDisabled();

    ImGui::Separator();

    if (ImGui::MenuItem("Quit", "Esc", false, true)) {
      teardown();
      std::exit(0);
    }

    ImGui::EndMenu(); // "Client"
  }

  ImGui::BeginDisabled(!m_client->isConnected());

  if (ImGui::BeginMenu("Server")) {
    if (ImGui::MenuItem("Start Rendering")) {
      tsd::core::logStatus("[Client] Sending START_RENDERING command");
      m_client->send(MessageType::SERVER_START_RENDERING);
    }

    if (ImGui::MenuItem("Pause Rendering")) {
      tsd::core::logStatus("[Client] Sending STOP_RENDERING command");
      m_client->send(MessageType::SERVER_STOP_RENDERING);
    }

    ImGui::Separator();

    if (ImGui::BeginMenu("Save State File")) {
      ImGui::InputText("Filename", &m_stateFileName);
      ImGui::Separator();
      if (ImGui::Button("Save")) {
        if (!m_stateFileName.empty()) {
          tsd::core::logStatus(
              "[Client] Sending command to save state file '%s'",
              m_stateFileName.c_str());
          auto msg =
              tsd::network::makeMessage(MessageType::SERVER_SAVE_STATE_FILE);
          tsd::network::payloadWrite(msg, m_stateFileName);
          m_client->send(std::move(msg));
        }
      }
      ImGui::EndMenu();
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Shutdown")) {
      tsd::core::logStatus("[Client] Sending SHUTDOWN command");
      m_client->send(MessageType::SERVER_SHUTDOWN).get();
      disconnect();
    }

    ImGui::EndMenu(); // "Server"
  }

  ImGui::EndDisabled();

  // Keyboard shortcuts //

  if (ImGui::IsKeyPressed(ImGuiKey_P, false)) {
    tsd::core::logStatus("[Client] Sending PING");
    m_client->send(MessageType::PING);
  }
}

void Application::teardown()
{
  disconnect();
  m_client->removeAllHandlers();
  m_viewport->setNetworkChannel(nullptr);
  TSDApplication::teardown();
}

const char *Application::getDefaultLayout() const
{
  return R"layout(
[Window][MainDockSpace]
Pos=0,56
Size=3840,2206
Collapsed=0

[Window][Viewport]
Pos=957,56
Size=2883,1683
Collapsed=0
DockId=0x00000003,0

[Window][Secondary View]
Pos=1237,26
Size=683,857
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=957,1741
Size=2883,521
Collapsed=0
DockId=0x0000000A,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Layers]
Pos=0,56
Size=955,1341
Collapsed=0
DockId=0x00000005,0

[Window][Object Editor]
Pos=0,1399
Size=955,863
Collapsed=0
DockId=0x00000006,0

[Window][Scene Controls]
Pos=0,26
Size=547,581
Collapsed=0
DockId=0x00000007,0

[Window][Database Editor]
Pos=0,1399
Size=955,863
Collapsed=0
DockId=0x00000006,1

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

[Table][0x413D162D,1]
Column 0  Weight=1.0000

[Docking][Data]
DockSpace       ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,56 Size=3840,2206 Split=X
  DockNode      ID=0x00000001 Parent=0x80F5B4C5 SizeRef=955,1054 Split=Y Selected=0x6426B955
    DockNode    ID=0x00000007 Parent=0x00000001 SizeRef=547,581 Selected=0x6426B955
    DockNode    ID=0x00000008 Parent=0x00000001 SizeRef=547,522 Split=Y Selected=0x8B73155F
      DockNode  ID=0x00000005 Parent=0x00000008 SizeRef=547,640 Selected=0xCD8384B1
      DockNode  ID=0x00000006 Parent=0x00000008 SizeRef=547,412 Selected=0x82B4C496
  DockNode      ID=0x00000002 Parent=0x80F5B4C5 SizeRef=2883,1054 Split=Y Selected=0xC450F867
    DockNode    ID=0x00000009 Parent=0x00000002 SizeRef=1371,1683 Split=X Selected=0xC450F867
      DockNode  ID=0x00000003 Parent=0x00000009 SizeRef=686,857 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x00000004 Parent=0x00000009 SizeRef=683,857 Selected=0xA3219422
    DockNode    ID=0x0000000A Parent=0x00000002 SizeRef=1371,521 Selected=0x139FDA3F
)layout";
}

void Application::connect()
{
  m_client->connect(m_host, m_port);
  if (m_client->isConnected()) {
    tsd::core::logStatus(
        "[Client] Connected to server at %s:%d", m_host.c_str(), m_port);
    tsd::core::logStatus("[Client] Requesting scene from server");
    m_client->send(MessageType::SERVER_REQUEST_SCENE).get();
  } else {
    tsd::core::logError("[Client] Failed to connect to server at %s:%d",
        m_host.c_str(),
        m_port);
  }
}

void Application::disconnect()
{
  tsd::core::logStatus("[Client] Disconnecting from server...");
  m_updateDelegate->setEnabled(false);
  m_viewport->disconnect();
  m_client->send(MessageType::DISCONNECT).get();
  m_client->disconnect();

  auto *ctx = appContext();
  ctx->tsd.sceneLoadComplete = false;
  ctx->clearSelected();
  auto &scene = ctx->tsd.scene;
  scene.removeAllObjects();
}

} // namespace tsd::demo

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *argv[])
{
  {
    tsd::core::setLogToStdout();
    tsd::demo::Application app;
    app.run(1920, 1080, "TSD Remote Viewer");
  }

  return 0;
}
