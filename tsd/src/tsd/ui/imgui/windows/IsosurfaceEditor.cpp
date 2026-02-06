// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "IsosurfaceEditor.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// tsd_app
#include "tsd/app/Core.h"
// std
#include <algorithm>

namespace tsd::ui::imgui {

IsosurfaceEditor::IsosurfaceEditor(Application *app, const char *name)
    : Window(app, name)
{}

void IsosurfaceEditor::buildUI()
{
  auto &scene = appCore()->tsd.scene;

  tsd::core::Object *selectedIsosurface = nullptr;
  tsd::core::Object *selectedVolume = nullptr;
  auto selectedNode = appCore()->getFirstSelected();
  tsd::core::Object *selectedObject =
      selectedNode.valid() ? (*selectedNode)->getObject() : nullptr;

  if (selectedObject != nullptr) {
    if (selectedObject->type() == ANARI_VOLUME)
      selectedVolume = selectedObject;
    else if (selectedObject->type() == ANARI_SURFACE)
      selectedObject = selectedObject->parameterValueAsObject("geometry");

    // NOTE: will get in here here if originally a surface was selected
    if (selectedObject && selectedObject->type() == ANARI_GEOMETRY
        && selectedObject->subtype()
            == tsd::core::tokens::geometry::isosurface) {
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
      selectedIsosurface->parameterValueAsObject<tsd::core::Array>("isovalue");
  if (!arr) {
    ImGui::Text("{no isovalue array object found!}");
    return;
  }

  const auto *isovalues = arr->dataAs<float>();

  for (size_t i = 0; i < arr->size(); i++) {
    ImGui::PushID(i);

    ImGui::BeginDisabled(arr->size() == 1);
    if (ImGui::Button("x")) {
      auto newArr = scene.createArray(ANARI_FLOAT32, arr->size() - 1);
      newArr->setData(isovalues, arr->size());
      auto *v = newArr->mapAs<float>();
      std::copy(isovalues, isovalues + i, v);
      std::copy(isovalues + i + 1, isovalues + arr->size(), v + i);
      newArr->unmap();
      selectedIsosurface->setParameterObject("isovalue", *newArr);
      scene.removeObject(arr);
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
    auto newArr = scene.createArray(ANARI_FLOAT32, arr->size() + 1);
    newArr->setData(isovalues, arr->size(), 0);
    selectedIsosurface->setParameterObject("isovalue", *newArr);
    scene.removeObject(arr);
  }
}

void IsosurfaceEditor::addIsosurfaceGeometryFromSelected()
{
  auto selectedNode = appCore()->getFirstSelected();
  tsd::core::Object *selectedObject =
      selectedNode.valid() ? (*selectedNode)->getObject() : nullptr;
  auto &scene = appCore()->tsd.scene;
  auto *layer = scene.defaultLayer();

  auto isovalue = scene.createArray(ANARI_FLOAT32, 1);

  auto g = scene.createObject<tsd::core::Geometry>(
      tsd::core::tokens::geometry::isosurface);
  g->setName("isosurface_geometry");

  if (auto *field = selectedObject->parameterValueAsObject("value"); field)
    g->setParameterObject("field", *field);

  g->setParameterObject("isovalue", *isovalue);

  auto s = scene.createSurface("isosurface", g, scene.defaultMaterial());

  auto n = layer->insert_last_child(layer->root(), {s});

  appCore()->setSelected(n);
  scene.signalLayerChange(layer);
}

} // namespace tsd::ui::imgui
