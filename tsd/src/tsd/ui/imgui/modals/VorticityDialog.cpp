// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "VorticityDialog.h"
// tsd_io
#include "tsd/io/procedural/computeVorticity.hpp"
// tsd_core
#include "tsd/core/ObjectPool.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/modals/BlockingTaskModal.h"
// imgui
#include "imgui.h"
// std
#include <string>
#include <vector>

namespace tsd::ui::imgui {

VorticityDialog::VorticityDialog(Application *app) : Modal(app, "Flow Analysis")
{}

void VorticityDialog::buildUI()
{
  using namespace tsd::core;

  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;

  // Collect all SpatialFields in the scene
  std::vector<const tsd::scene::SpatialField *> fields;
  std::vector<std::string> fieldNames;

  foreach_item_const(
      scene.objectDB().field, [&](const tsd::scene::SpatialField *sf) {
        if (sf) {
          fields.push_back(sf);
          fieldNames.push_back(sf->name());
        }
      });

  // Helper to build the combo label list
  std::vector<const char *> labels;
  labels.reserve(fields.size());
  for (auto &n : fieldNames)
    labels.push_back(n.c_str());

  const int nFields = (int)fields.size();

  ImGui::Text("Select velocity component fields:");
  ImGui::Spacing();

  // Clamp indices in case fields changed
  if (m_uIdx >= nFields)
    m_uIdx = -1;
  if (m_vIdx >= nFields)
    m_vIdx = -1;
  if (m_wIdx >= nFields)
    m_wIdx = -1;

  auto comboField = [&](const char *label, int &idx) {
    const char *preview = (idx >= 0 && idx < nFields) ? labels[idx] : "(none)";
    if (ImGui::BeginCombo(label, preview)) {
      for (int i = 0; i < nFields; ++i) {
        bool selected = (i == idx);
        if (ImGui::Selectable(labels[i], selected))
          idx = i;
        if (selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
  };

  comboField("U  (x-velocity)", m_uIdx);
  comboField("V  (y-velocity)", m_vIdx);
  comboField("W  (z-velocity)", m_wIdx);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Text("Outputs to compute:");

  ImGui::Checkbox("Lambda2", &m_computeLambda2);
  ImGui::SameLine();
  ImGui::Checkbox("Q-criterion", &m_computeQ);
  ImGui::SameLine();
  ImGui::Checkbox("Vorticity magnitude", &m_computeVorticity);
  ImGui::SameLine();
  ImGui::Checkbox("Helicity", &m_computeHelicity);

  ImGui::Spacing();
  ImGui::Separator();

  if (ImGui::Button("Cancel") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();

  ImGui::SameLine();

  bool canRun = (m_uIdx >= 0 && m_vIdx >= 0 && m_wIdx >= 0 && m_uIdx != m_vIdx
      && m_uIdx != m_wIdx && m_vIdx != m_wIdx);

  ImGui::BeginDisabled(!canRun);

  if (ImGui::Button("Compute")) {
    this->hide();

    const tsd::scene::SpatialField *uField = fields[m_uIdx];
    const tsd::scene::SpatialField *vField = fields[m_vIdx];
    const tsd::scene::SpatialField *wField = fields[m_wIdx];

    tsd::io::VorticityOptions opts;
    opts.lambda2 = m_computeLambda2;
    opts.qCriterion = m_computeQ;
    opts.vorticity = m_computeVorticity;
    opts.helicity = m_computeHelicity;

    auto doCompute = [&scene, uField, vField, wField, opts]() {
      tsd::io::computeVorticity(scene, uField, vField, wField, {}, opts);
    };

    m_app->showTaskModal(doCompute, "Please Wait: Computing Flow Analysis...");
  }

  ImGui::EndDisabled();

  if (!canRun && nFields > 0) {
    ImGui::SameLine();
    ImGui::TextDisabled("(select 3 distinct fields)");
  } else if (nFields == 0) {
    ImGui::SameLine();
    ImGui::TextDisabled("(no SpatialFields in scene)");
  }
}

} // namespace tsd::ui::imgui
