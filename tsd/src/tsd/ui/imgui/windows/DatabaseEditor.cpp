// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatabaseEditor.h"
#include "tsd_ui.h"

#include "../AppCore.h"

namespace tsd_viewer {

DatabaseEditor::DatabaseEditor(AppCore *core, const char *name)
    : Window(core, name)
{}

void DatabaseEditor::buildUI()
{
  ImGui::BeginDisabled(!m_core->tsd.sceneLoadComplete);

  auto buildUI_objectSection = [&](const auto &ctxList,
                                   const char *headerText) {
    if (ctxList.empty())
      return;
    ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
    if (ImGui::CollapsingHeader(headerText, ImGuiTreeNodeFlags_None)) {
      tsd::foreach_item_const(ctxList, [&](auto *o) {
        if (!o)
          return;

        ImGui::Separator();

        ImGui::PushID(o);
        if (ImGui::Button("delete"))
          m_core->tsd.ctx.removeObject(*o);
        else
          tsd::ui::buildUI_object(*o, m_core->tsd.ctx, true);
        ImGui::PopID();
      });
    }
  };

  const auto &db = m_core->tsd.ctx.objectDB();

  buildUI_objectSection(db.light, "Lights");
  buildUI_objectSection(db.sampler, "Samplers");
  buildUI_objectSection(db.material, "Materials");
  buildUI_objectSection(db.geometry, "Geometries");
  buildUI_objectSection(db.surface, "Surfaces");
  buildUI_objectSection(db.field, "SpatialFields");
  buildUI_objectSection(db.volume, "Volumes");

  ImGui::EndDisabled();
}

} // namespace tsd_viewer