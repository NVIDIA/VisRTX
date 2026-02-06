// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatabaseEditor.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// tsd_app
#include "tsd/app/Core.h"

namespace tsd::ui::imgui {

DatabaseEditor::DatabaseEditor(Application *app, const char *name)
    : Window(app, name)
{}

void DatabaseEditor::buildUI()
{
  ImGui::BeginDisabled(!appCore()->tsd.sceneLoadComplete);

  auto buildUI_objectSection = [&](const auto &ctxList,
                                   const char *headerText) {
    if (ctxList.empty())
      return;
    ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
    if (ImGui::CollapsingHeader(headerText, ImGuiTreeNodeFlags_None)) {
      tsd::core::foreach_item_const(ctxList, [&](auto *o) {
        if (!o)
          return;

        ImGui::Separator();

        ImGui::PushID(o);
        ImGui::BeginDisabled(o->totalUseCount() > 0);
        const bool doDelete = ImGui::Button("delete");
        ImGui::EndDisabled();
        if (doDelete)
          appCore()->tsd.scene.removeObject(o);
        else
          tsd::ui::buildUI_object(*o, appCore()->tsd.scene, true);
        ImGui::PopID();
      });
    }
  };

  const auto &db = appCore()->tsd.scene.objectDB();

  buildUI_objectSection(db.camera, "Cameras");
  buildUI_objectSection(db.light, "Lights");
  buildUI_objectSection(db.sampler, "Samplers");
  buildUI_objectSection(db.material, "Materials");
  buildUI_objectSection(db.geometry, "Geometries");
  buildUI_objectSection(db.surface, "Surfaces");
  buildUI_objectSection(db.field, "Spatial Fields");
  buildUI_objectSection(db.volume, "Volumes");
  buildUI_objectSection(db.array, "Arrays");

  ImGui::EndDisabled();
}

} // namespace tsd::ui::imgui