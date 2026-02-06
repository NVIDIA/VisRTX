// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ObjectEditor.h"
#include "tsd/app/Core.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"

namespace math = tsd::math;

namespace tsd::ui::imgui {

ObjectEditor::ObjectEditor(Application *app, const char *name)
    : Window(app, name)
{}

void ObjectEditor::buildUI()
{
  auto selectedNode = appCore()->getFirstSelected();
  if (!selectedNode.valid()) {
    ImGui::Text("{no object selected}");
    return;
  }

  ImGui::BeginDisabled(!appCore()->tsd.sceneLoadComplete);

  auto *scene = &appCore()->tsd.scene;

  auto &node = *selectedNode;

  if (auto *selectedObject = node->getObject(); selectedObject) {
    tsd::ui::buildUI_object(*selectedObject, appCore()->tsd.scene, true);
  } else if (node->isTransform()) {
    // Setup transform values //

    auto srt = node->getTransformSRT();
    auto &sc = srt[0];
    auto &azelrot = srt[1];
    auto &tl = srt[2];

    // UI widgets //

    bool doUpdate = false;

    ImGui::BeginDisabled(node->isDefaultValue());
    if (ImGui::Button("reset")) {
      node->setToDefaultValue();
      doUpdate = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("set default"))
      node->setCurrentValueAsDefault();
    ImGui::EndDisabled();

    doUpdate |= ImGui::DragFloat3("scale", &sc.x);
    doUpdate |= ImGui::SliderFloat3("rotation", &azelrot.x, 0.f, 360.f);
    doUpdate |= ImGui::DragFloat3("translation", &tl.x);

    // Handle transform update //

    if (doUpdate) {
      node->setAsTransform(srt);
      auto *layer = selectedNode->container();
      scene->signalLayerChange(layer);
    }
  } else if (!node->isEmpty()) {
    ImGui::Text("{unhandled '%s' node}", anari::toString(node->type()));
  } else {
    ImGui::Text("TODO: empty node");
  }

  ImGui::EndDisabled();
}

} // namespace tsd::ui::imgui