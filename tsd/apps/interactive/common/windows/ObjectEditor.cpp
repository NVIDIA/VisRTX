// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ObjectEditor.h"
#include "tsd_ui.h"

namespace tsd_viewer {

ObjectEditor::ObjectEditor(AppContext *state, const char *name)
    : anari_viewer::windows::Window(name, true), m_context(state)
{}

void ObjectEditor::buildUI()
{
  if (m_context->tsd.selectedObject == nullptr)
    return;

  ImGui::BeginDisabled(!m_context->tsd.sceneLoadComplete);
  tsd::ui::buildUI_object(
      *m_context->tsd.selectedObject, m_context->tsd.ctx, true);
  ImGui::EndDisabled();
}

} // namespace tsd_viewer