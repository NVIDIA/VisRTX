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

  if (node->isTransform()) {
    auto *transformObject = node->getTransformObject();
    if (transformObject->getTransformArray()) {
      ImGui::Text("Instance transform (%zu instances)",
          transformObject->getTransformArray()->size());
    } else {
      // Decompose SRT from the live matrix so sliders reflect the actual
      // transform.
      auto currentMat = transformObject->getTransform();
      math::float3 sc, azelrot, tl;
      math::mat4 rot;
      math::decomposeMatrix(currentMat, sc, rot, tl);
      azelrot = math::degrees(math::matrixToAzElRoll(rot));
      math::mat3 srt(sc, azelrot, tl);

      bool doUpdate = false;

      doUpdate |= ImGui::DragFloat3("scale", &sc.x);
      doUpdate |= ImGui::SliderFloat3("rotation", &azelrot.x, 0.f, 360.f);
      doUpdate |= ImGui::DragFloat3("translation", &tl.x);

      if (doUpdate)
        transformObject->setTransform(math::mat3(sc, azelrot, tl));
    }
  } else if (auto *selectedObject = node->getObject(); selectedObject) {
    tsd::ui::buildUI_object(*selectedObject, appCore()->tsd.scene, true);
  } else if (!node->isEmpty()) {
    ImGui::Text("{unhandled '%s' node}", anari::toString(node->type()));
  } else {
    ImGui::Text("TODO: empty node");
  }

  ImGui::EndDisabled();
}

} // namespace tsd::ui::imgui