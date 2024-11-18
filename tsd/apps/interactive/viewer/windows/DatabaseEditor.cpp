// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatabaseEditor.h"
#include "tsd_ui.h"

#include "AppContext.h"

namespace tsd_viewer {

DatabaseEditor::DatabaseEditor(AppContext *ctx, const char *name)
    : anari_viewer::windows::Window(name, true), m_context(ctx)
{}

void DatabaseEditor::buildUI()
{
  ImGui::BeginDisabled(!m_context->tsd.sceneLoadComplete);

  auto buildUI_objectSection = [&](const auto &ctxList,
                                   const char *headerText) {
    if (ctxList.empty())
      return;
    ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
    if (ImGui::CollapsingHeader(headerText, ImGuiTreeNodeFlags_None)) {
      tsd::foreach_item_const(ctxList, [&](auto *o) {
        tsd::ui::buildUI_object(*o, m_context->tsd.ctx, true);
      });
    }
  };

  const auto &db = m_context->tsd.ctx.objectDB();

  buildUI_objectSection(db.light, "Lights");
  buildUI_objectSection(db.sampler, "Samplers");
  buildUI_objectSection(db.material, "Materials");
  buildUI_objectSection(db.geometry, "Geometries");
  buildUI_objectSection(db.field, "SpatialFields");
  buildUI_objectSection(db.volume, "Volumes");

  ImGui::EndDisabled();
}

} // namespace tsd_viewer