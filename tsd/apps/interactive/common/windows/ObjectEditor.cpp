// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ObjectEditor.h"
#include "tsd_ui.h"

namespace tsd_viewer {

ObjectEditor::ObjectEditor(AppCore *state, const char *name)
    : anari_viewer::windows::Window(name, true), m_core(state)
{}

void ObjectEditor::buildUI()
{
  if (m_core->tsd.selectedObject == nullptr) {
    ImGui::Text("{no object selected}");
    return;
  }

  ImGui::BeginDisabled(!m_core->tsd.sceneLoadComplete);
  tsd::ui::buildUI_object(
      *m_core->tsd.selectedObject, m_core->tsd.ctx, true);
  ImGui::EndDisabled();
}

} // namespace tsd_viewer