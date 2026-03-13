// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CuttingPlaneDialog.h"
// tsd_core
#include "tsd/core/ObjectPool.hpp"
#include "tsd/scene/objects/Renderer.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
// imgui
#include "imgui.h"
// std
#include <cmath>

namespace tsd::ui::imgui {

CuttingPlaneDialog::CuttingPlaneDialog(Application *app)
    : Modal(app, "Cutting Plane")
{}

void CuttingPlaneDialog::buildUI()
{
  using namespace tsd::core;

  ImGui::Checkbox("Enable", &m_enabled);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Text("Preset planes:");
  if (ImGui::Button("XY")) {
    m_normal[0] = 0.f;
    m_normal[1] = 0.f;
    m_normal[2] = 1.f;
  }
  ImGui::SameLine();
  if (ImGui::Button("XZ")) {
    m_normal[0] = 0.f;
    m_normal[1] = 1.f;
    m_normal[2] = 0.f;
  }
  ImGui::SameLine();
  if (ImGui::Button("YZ")) {
    m_normal[0] = 1.f;
    m_normal[1] = 0.f;
    m_normal[2] = 0.f;
  }

  ImGui::Spacing();
  ImGui::DragFloat3("Normal", m_normal, 0.01f);
  ImGui::DragFloat("d (offset)", &m_d, 0.1f);

  ImGui::Spacing();
  ImGui::Separator();

  if (ImGui::Button("Apply")) {
    float plane[4];
    if (m_enabled) {
      float len = sqrtf(m_normal[0] * m_normal[0]
                      + m_normal[1] * m_normal[1]
                      + m_normal[2] * m_normal[2]);
      if (len < 1e-6f)
        len = 1.f;
      plane[0] = m_normal[0] / len;
      plane[1] = m_normal[1] / len;
      plane[2] = m_normal[2] / len;
      plane[3] = m_d;
    } else {
      plane[0] = plane[1] = plane[2] = 0.f;
      plane[3] = -1e30f; // disabled sentinel
    }

    auto &scene = appContext()->tsd.scene;
    foreach_item_const(scene.objectDB().renderer, [&](tsd::core::Renderer *r) {
      if (r)
        r->setParameter(tsd::core::Token("cutPlane"),
                        ANARI_FLOAT32_VEC4, plane);
    });
  }

  ImGui::SameLine();

  if (ImGui::Button("Cancel") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();
}

} // namespace tsd::ui::imgui
