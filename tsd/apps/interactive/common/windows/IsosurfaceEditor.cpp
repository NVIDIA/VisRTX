// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "IsosurfaceEditor.h"
#include "tsd_ui.h"
// std
#include <algorithm>

namespace tsd_viewer {

IsosurfaceEditor::IsosurfaceEditor(AppCore *state, const char *name)
    : anari_viewer::windows::Window(name, true), m_core(state)
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
    else if (selectedObject->type() == ANARI_SURFACE) {
      // Get the geometry from the selected surface for us to actually check
      if (auto *p = selectedObject->parameter("geometry"); p != nullptr)
        selectedObject = ctx.getObject(p->value());
    }

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

  auto *arr = (tsd::Array *)ctx.getObject(
      selectedIsosurface->parameter("isovalue")->value());
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

  tsd::Object *field = nullptr;
  if (auto *p = selectedObject->parameter("value"); p != nullptr)
    field = m_core->tsd.ctx.getObject(p->value());
  auto isovalue = ctx.createArray(ANARI_FLOAT32, 1);

  auto g = ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::isosurface);
  g->setName("isosurface_geometry");

  if (field)
    g->setParameterObject("field", *field);

  g->setParameterObject("isovalue", *isovalue);

  auto s = ctx.createSurface("isosurface", g, ctx.defaultMaterial());

  auto n = ctx.tree.insert_last_child(
      ctx.tree.root(), tsd::utility::Any(ANARI_SURFACE, s.index()));

  m_core->setSelectedNode(*n);
  ctx.signalInstanceTreeChange();
}

} // namespace tsd_viewer
