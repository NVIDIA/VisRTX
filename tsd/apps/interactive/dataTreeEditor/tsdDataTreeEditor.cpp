// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/Log.h>
// tsd_core
#include <tsd/core/DataTree.hpp>
// imgui
#include <imgui.h>

#include "DataTreeWindow.h"

namespace tsd::datatree_editor {

using TSDApplication = tsd::ui::imgui::Application;
namespace tsd_ui = tsd::ui::imgui;

class Application : public TSDApplication
{
 public:
  Application(int argc, const char **argv) : TSDApplication(argc, argv)
  {
    // Consume a positional argument as the file to open at startup
    for (int i = 1; i < argc; ++i) {
      std::string_view arg(argv[i]);
      if (!arg.empty() && arg[0] != '-') {
        m_fileToOpenNextFrame = std::string(arg);
        break;
      }
    }
  }

  ~Application() override = default;

  tsd::ui::imgui::WindowArray setupWindows() override
  {
    auto windows = TSDApplication::setupWindows();

    // Mark scene as complete so any base-class UI doesn't block
    appContext()->tsd.sceneLoadComplete = true;

    auto *treeWin = new DataTreeWindow(this, &m_tree, &m_dirty, &m_currentFile);
    auto *log = new tsd_ui::Log(this);

    windows.emplace_back(treeWin);
    windows.emplace_back(log);
    setWindowArray(windows);

    return windows;
  }

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,28
Size=1920,1052
Collapsed=0

[Window][DataTree Editor]
Pos=0,28
Size=1920,795
Collapsed=0
DockId=0x00000006,0

[Window][Log]
Pos=0,825
Size=1920,255
Collapsed=0
DockId=0x00000005,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Viewport]
Pos=549,28
Size=1371,795
Collapsed=0
DockId=0x00000006,0

[Window][Database Editor]
Pos=0,577
Size=547,503
Collapsed=0
DockId=0x00000009,1

[Window][Layers]
Pos=0,28
Size=547,547
Collapsed=0
DockId=0x00000008,0

[Window][Object Editor]
Pos=0,577
Size=547,503
Collapsed=0
DockId=0x00000009,0

[Window][Terminal]
Pos=549,825
Size=1371,255
Collapsed=0
DockId=0x00000002,1

[Window][Secondary View]
Pos=1237,26
Size=683,848
Collapsed=0
DockId=0x00000007,0

[Window][Isosurface Editor]
Pos=1370,26
Size=550,1054
Collapsed=0
DockId=0x0000000C,0

[Window][TF Editor]
Pos=2483,26
Size=550,1769
Collapsed=0
DockId=0x0000000B,0

[Window][Camera Poses]
Pos=0,28
Size=547,547
Collapsed=0
DockId=0x00000008,2

[Window][Animations]
Pos=0,28
Size=547,547
Collapsed=0
DockId=0x00000008,1

[Window][##]
Pos=792,507
Size=336,116
Collapsed=0

[Window][Timeline]
Pos=549,825
Size=1371,255
Collapsed=0
DockId=0x00000002,2

[Window][##blocking_task_modal]
Pos=792,480
Size=16,44
Collapsed=0

[Window][Cutting Plane]
Pos=1293,781
Size=446,232
Collapsed=0

[Table][0x44C159D3,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x9E1800B1,1]
Column 0  Weight=1.0000

[Table][0xFAE9835A,1]
Column 0  Weight=1.0000

[Table][0x413D162D,1]
Column 0  Weight=1.0000

[Table][0x34853C34,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xEEE697AB,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x50507568,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xF4075185,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xF945472E,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xC7D50986,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x7AB8FCE0,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0xC6287F21,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x3DC35996,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x7C7C00B3,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x7B5A3115,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x2FB69A7F,2]
Column 0  Weight=0.4828
Column 1  Weight=0.5172

[Docking][Data]
DockSpace         ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,28 Size=1920,1052 Split=Y
  DockNode        ID=0x00000004 Parent=0x80F5B4C5 SizeRef=1920,795 Split=X
    DockNode      ID=0x00000003 Parent=0x00000004 SizeRef=1368,1054 Split=X
      DockNode    ID=0x00000001 Parent=0x00000003 SizeRef=547,1105 Split=Y Selected=0xCD8384B1
        DockNode  ID=0x00000008 Parent=0x00000001 SizeRef=547,575 Selected=0xCD8384B1
        DockNode  ID=0x00000009 Parent=0x00000001 SizeRef=547,528 Selected=0x82B4C496
      DockNode    ID=0x00000002 Parent=0x00000003 SizeRef=1371,1105 Split=X
        DockNode  ID=0x00000006 Parent=0x00000002 SizeRef=685,848 CentralNode=1 Selected=0x7DE8CD70
        DockNode  ID=0x00000007 Parent=0x00000002 SizeRef=683,848 Selected=0xA3219422
    DockNode      ID=0x0000000A Parent=0x00000004 SizeRef=550,1054 Split=Y Selected=0x3429FA32
      DockNode    ID=0x0000000B Parent=0x0000000A SizeRef=550,590 Selected=0x3429FA32
      DockNode    ID=0x0000000C Parent=0x0000000A SizeRef=550,462 Selected=0xBCE6538B
  DockNode        ID=0x00000005 Parent=0x80F5B4C5 SizeRef=1920,255 Selected=0x139FDA3F
)layout";
  }

 protected:
  void uiFrameStart() override
  {
    // Process pending file operations before base-class UI
    if (!m_fileToOpenNextFrame.empty()) {
      m_tree.load(m_fileToOpenNextFrame.c_str());
      m_currentFile = m_fileToOpenNextFrame;
      m_dirty = false;
      m_fileToOpenNextFrame.clear();
    }
    if (!m_fileToSaveNextFrame.empty()) {
      m_tree.save(m_fileToSaveNextFrame.c_str());
      m_currentFile = m_fileToSaveNextFrame;
      m_dirty = false;
      m_fileToSaveNextFrame.clear();
    }

    TSDApplication::uiFrameStart();
  }

  void uiMainMenuBar() override
  {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("New")) {
        m_tree.root().reset();
        m_currentFile.clear();
        m_dirty = false;
      }

      if (ImGui::MenuItem("Open..."))
        getFilenameFromDialog(m_fileToOpenNextFrame, false);

      ImGui::Separator();

      const bool hasSavePath = !m_currentFile.empty();
      if (ImGui::MenuItem("Save", "Ctrl+S", false, hasSavePath))
        m_fileToSaveNextFrame = m_currentFile;

      if (ImGui::MenuItem("Save As..."))
        getFilenameFromDialog(m_fileToSaveNextFrame, true);

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Edit")) {
      if (ImGui::MenuItem("Settings"))
        m_appSettingsDialog->show();

      ImGui::Separator();

      if (ImGui::BeginMenu("UI Layout")) {
        if (ImGui::MenuItem("Print"))
          printf("%s\n", ImGui::SaveIniSettingsToMemory());

        ImGui::Separator();

        if (ImGui::MenuItem("Reset"))
          ImGui::LoadIniSettingsFromMemory(getDefaultLayout());

        ImGui::EndMenu();
      }

      ImGui::EndMenu();
    }
  }

 private:
  tsd::core::DataTree m_tree;
  bool m_dirty{false};
  std::string m_currentFile;

  // Filenames filled by the async SDL file dialog; processed next frame
  std::string m_fileToOpenNextFrame;
  std::string m_fileToSaveNextFrame;
};

} // namespace tsd::datatree_editor

///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv)
{
  {
    tsd::datatree_editor::Application app(argc, argv);
    app.run(1920, 1080, "TSD DataTree Editor");
  }
  return 0;
}
