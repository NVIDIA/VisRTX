// Copyright 2024-2025 NVIDIA Corporation
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
  if (!appCore()->tsd.selectedObject && !appCore()->tsd.selectedNode) {
    ImGui::Text("{no object selected}");
    return;
  }

  ImGui::BeginDisabled(!appCore()->tsd.sceneLoadComplete);

  auto *ctx = &appCore()->tsd.ctx;

  if (!appCore()->tsd.selectedNode) {
    tsd::ui::buildUI_object(
        *appCore()->tsd.selectedObject, appCore()->tsd.ctx, true);
  } else {
    auto &selectedNode = *appCore()->tsd.selectedNode;

    if (auto *selectedObject = selectedNode->getObject(ctx); selectedObject) {
      tsd::ui::buildUI_object(*selectedObject, appCore()->tsd.ctx, true);
    } else if (selectedNode->isTransform()) {
      // Setup transform values //

      math::mat3 srt;

      auto &sc = srt[0];
      auto &azelrot = srt[1];
      auto &tl = srt[2];

      auto setSRTCache = [&]() {
        math::mat4 rot;
        math::decomposeMatrix(selectedNode->getTransform(), sc, rot, tl);
        azelrot = math::degrees(math::matrixToAzElRoll(rot));
        selectedNode->valueCache["SRT"] = srt;
      };

      const bool hasCachedSRT = selectedNode->valueCache.contains("SRT");
      if (hasCachedSRT)
        srt = selectedNode->valueCache["SRT"].getAs<math::mat3>();
      else
        setSRTCache();

      // UI widgets //

      bool doUpdate = false;

      ImGui::BeginDisabled(selectedNode->value == selectedNode->defaultValue);
      if (ImGui::Button("reset")) {
        selectedNode->value = selectedNode->defaultValue;
        doUpdate = true;
        setSRTCache();
      }
      ImGui::SameLine();
      if (ImGui::Button("set default"))
        selectedNode->defaultValue = selectedNode->value;
      ImGui::EndDisabled();

      doUpdate |= ImGui::DragFloat3("scale", &sc.x);
      doUpdate |= ImGui::SliderFloat3("rotation", &azelrot.x, 0.f, 360.f);
      doUpdate |= ImGui::DragFloat3("translation", &tl.x);

      // Handle transform update //

      if (doUpdate) {
        auto rot = math::IDENTITY_MAT4;
        rot = math::mul(rot,
            math::rotation_matrix(math::rotation_quat(
                math::float3(0.f, 1.f, 0.f), math::radians(azelrot.x))));
        rot = math::mul(rot,
            math::rotation_matrix(math::rotation_quat(
                math::float3(1.f, 0.f, 0.f), math::radians(azelrot.y))));
        rot = math::mul(rot,
            math::rotation_matrix(math::rotation_quat(
                math::float3(0.f, 0.f, 1.f), math::radians(azelrot.z))));

        selectedNode->value = math::mul(math::translation_matrix(tl),
            math::mul(rot, math::scaling_matrix(sc)));
        selectedNode->valueCache["SRT"] = srt;
        auto *layer = selectedNode.container();
        ctx->signalLayerChange(layer);
      }
    } else if (!selectedNode->isEmpty()) {
      ImGui::Text(
          "{unhandled '%s' node}", anari::toString(selectedNode->value.type()));
    } else {
      ImGui::Text("TODO: empty node");
    }
  }

  ImGui::EndDisabled();
}

} // namespace tsd::ui::imgui