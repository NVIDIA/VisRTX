// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "IsosurfaceEditor.h"
#include "tsd_ui.h"
// std
#include <algorithm>

#include "../AppCore.h"

namespace tsd_viewer {

IsosurfaceEditor::IsosurfaceEditor(AppCore *core, const char *name)
    : Window(core, name)
{}

void IsosurfaceEditor::buildUI()
{
  auto &ctx = m_core->tsd.ctx;

  tsd::Object *selectedIsosurface = nullptr;
  tsd::Object *selectedVolume = nullptr;
  tsd::Object *selectedObject = m_core->tsd.selectedObject;

  if (selectedObject != nullptr) {
    if (selectedObject->type() == ANARI_VOLUME)
      selectedVolume = selectedObject;
    else if (selectedObject->type() == ANARI_SURFACE)
      selectedObject = selectedObject->parameterValueAsObject("geometry");

    // NOTE: will get in here here if originally a surface was selected
    if (selectedObject && selectedObject->type() == ANARI_GEOMETRY
        && selectedObject->subtype() == tsd::tokens::geometry::isosurface) {
      selectedIsosurface = selectedObject;
    }
  }

  if (selectedVolume != nullptr) {
    if (ImGui::Button("add isosurface geometry from selected volume"))
      addIsosurfaceGeometryFromSelected();
    return;
  } else if (!selectedIsosurface) {
    ImGui::Text("{no isosurface object selected}");
    return;
  }

  ImGui::Text("isovalues:");

  auto *arr =
      selectedIsosurface->parameterValueAsObject<tsd::Array>("isovalue");
  if (!arr) {
    ImGui::Text("{no isovalue array object found!}");
    return;
  }

  const auto *isovalues = arr->dataAs<float>();

  for (size_t i = 0; i < arr->size(); i++) {
    ImGui::PushID(i);

    ImGui::BeginDisabled(arr->size() == 1);
    if (ImGui::Button("x")) {
      auto newArr = ctx.createArray(ANARI_FLOAT32, arr->size() - 1);
      newArr->setData(isovalues, arr->size());
      auto *v = newArr->mapAs<float>();
      std::copy(isovalues, isovalues + i, v);
      std::copy(isovalues + i + 1, isovalues + arr->size(), v + i);
      newArr->unmap();
      selectedIsosurface->setParameterObject("isovalue", *newArr);
      ctx.removeObject(*arr);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    float v = isovalues[i];
    if (ImGui::DragFloat("##isovalue", &v)) {
      arr->mapAs<float>()[i] = v;
      arr->unmap();
    }

    ImGui::PopID();
  }

  if (ImGui::Button("+")) {
    auto newArr = ctx.createArray(ANARI_FLOAT32, arr->size() + 1);
    newArr->setData(isovalues, arr->size());
    selectedIsosurface->setParameterObject("isovalue", *newArr);
    ctx.removeObject(*arr);
  }
}

void IsosurfaceEditor::addIsosurfaceGeometryFromSelected()
{
  tsd::Object *selectedObject = m_core->tsd.selectedObject;
  auto &ctx = m_core->tsd.ctx;
  auto *layer = ctx.defaultLayer();

  auto isovalue = ctx.createArray(ANARI_FLOAT32, 1);

  auto g = ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::isosurface);
  g->setName("isosurface_geometry");

  if (auto *field = selectedObject->parameterValueAsObject("value"); field)
    g->setParameterObject("field", *field);

  g->setParameterObject("isovalue", *isovalue);

  auto s = ctx.createSurface("isosurface", g, ctx.defaultMaterial());

  auto n = layer->insert_last_child(
      layer->root(), tsd::utility::Any(ANARI_SURFACE, s.index()));

  m_core->setSelectedNode(*n);
  ctx.signalLayerChange(layer);
}

} // namespace tsd_viewer
