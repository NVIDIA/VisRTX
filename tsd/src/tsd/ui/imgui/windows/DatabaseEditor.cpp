// Copyright 2024-2025 NVIDIA Corporation
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
        if (ImGui::Button("delete"))
          appCore()->tsd.ctx.removeObject(*o);
        else
          tsd::ui::buildUI_object(*o, appCore()->tsd.ctx, true);
        ImGui::PopID();
      });
    }
  };

  const auto &db = appCore()->tsd.ctx.objectDB();

  buildUI_objectSection(db.light, "Lights");
  buildUI_objectSection(db.sampler, "Samplers");
  buildUI_objectSection(db.material, "Materials");
  buildUI_objectSection(db.geometry, "Geometries");
  buildUI_objectSection(db.surface, "Surfaces");
  buildUI_objectSection(db.field, "SpatialFields");
  buildUI_objectSection(db.volume, "Volumes");

  ImGui::EndDisabled();
}

} // namespace tsd::ui::imgui