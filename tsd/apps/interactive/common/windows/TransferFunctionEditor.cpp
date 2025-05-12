// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferFunctionEditor.h"
#include "../tsd_ui.h"
// std
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
// tsd
#include "tsd/core/ColorMapUtil.hpp"

namespace tsd_viewer {

// Helper functions ///////////////////////////////////////////////////////////

template <typename T>
static int find_idx(const std::vector<T> &A, float p)
{
  auto found =
      std::upper_bound(A.begin(), A.end(), T(p), [](const T &a, const T &b) {
        return a.x < b.x;
      });
  return std::distance(A.begin(), found);
}

// TransferFunctionEditor definitions /////////////////////////////////////////

TransferFunctionEditor::TransferFunctionEditor(AppCore *core, const char *name)
    : Window(core->application, name, true), m_core(core)
{
  loadDefaultMaps();
  m_tfnOpacityPoints.resize(2);
  m_tfnOpacityPoints[0] = {0.f, 0.f};
  m_tfnOpacityPoints[1] = {1.f, 1.f};
  m_tfnColorPoints = &(m_tfnsColorPoints[0]);
}

TransferFunctionEditor::~TransferFunctionEditor()
{
  if (m_tfnPaletteTexture)
    SDL_DestroyTexture(m_tfnPaletteTexture);
}

void TransferFunctionEditor::buildUI()
{
  setObjectPtrsFromSelectedObject();

  if (m_volume && m_nextMap != m_currentMap) {
    m_currentMap = m_nextMap;
    m_tfnColorPoints = &(m_tfnsColorPoints[m_currentMap]);
    updateVolume();
    updateTfnPaletteTexture();
  }

  if (m_volume == nullptr) {
    ImGui::Text("{no volume selected}");
    return;
  }

  buildUI_selectColorMap();
  ImGui::Separator();
  buildUI_drawEditor();
  ImGui::Separator();
  buildUI_opacityScale();
  ImGui::Separator();
  buildUI_valueRange();
}

void TransferFunctionEditor::buildUI_selectColorMap()
{
  static std::vector<const char *> names;
  if (names.empty()) {
    names.resize(m_tfnsNames.size());
    std::transform(m_tfnsNames.begin(),
        m_tfnsNames.end(),
        names.begin(),
        [](const std::string &t) { return t.c_str(); });
  }

  int newMap = m_currentMap;
  if (ImGui::Combo("color map", &newMap, names.data(), names.size()))
    setMap(newMap);
}

void TransferFunctionEditor::buildUI_drawEditor()
{
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  float canvas_x = ImGui::GetCursorScreenPos().x;
  float canvas_y = ImGui::GetCursorScreenPos().y;
  float canvas_avail_x = ImGui::GetContentRegionAvail().x;
  float canvas_avail_y = ImGui::GetContentRegionAvail().y;
  const float mouse_x = ImGui::GetMousePos().x;
  const float mouse_y = ImGui::GetMousePos().y;
  const float scroll_x = ImGui::GetScrollX();
  const float scroll_y = ImGui::GetScrollY();
  const float margin = 10.f;
  const float width = canvas_avail_x - 2.f * margin;
  const float height = 260.f;
  const float color_len = 9.f;
  const float opacity_len = 7.f;

  // draw preview texture
  ImGui::SetCursorScreenPos(ImVec2(canvas_x + margin, canvas_y));
  ImGui::Image(reinterpret_cast<ImTextureID>(m_tfnPaletteTexture),
      ImVec2(width, height));

  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  {
    std::vector<ImVec2> polyline;
    polyline.reserve(4);
    for (int i = 0; i < m_tfnOpacityPoints.size() - 1; ++i) {
      polyline.clear();
      polyline.emplace_back(canvas_x + margin + m_tfnOpacityPoints[i].x * width,
          canvas_y + height);
      polyline.emplace_back(canvas_x + margin + m_tfnOpacityPoints[i].x * width,
          canvas_y + height - m_tfnOpacityPoints[i].y * height);
      polyline.emplace_back(
          canvas_x + margin + m_tfnOpacityPoints[i + 1].x * width + 1,
          canvas_y + height - m_tfnOpacityPoints[i + 1].y * height);
      polyline.emplace_back(
          canvas_x + margin + m_tfnOpacityPoints[i + 1].x * width + 1,
          canvas_y + height);
      draw_list->AddConvexPolyFilled(
          polyline.data(), polyline.size(), 0xc8d8d8d8);
    }
  }
  canvas_y += height + margin;
  canvas_avail_y -= height + margin;

  // draw opacity control points
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  {
    // draw circles
    for (int i = 0; i < m_tfnOpacityPoints.size(); ++i) {
      const ImVec2 pos(canvas_x + width * m_tfnOpacityPoints[i].x + margin,
          canvas_y - height * m_tfnOpacityPoints[i].y - margin);
      ImGui::SetCursorScreenPos(
          ImVec2(pos.x - opacity_len, pos.y - opacity_len));
      ImGui::InvisibleButton(("##OpacityControl-" + std::to_string(i)).c_str(),
          ImVec2(2.f * opacity_len, 2.f * opacity_len));
      ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));

      // dark bounding box
      draw_list->AddCircleFilled(pos, opacity_len, 0xFF565656);

      // white background
      draw_list->AddCircleFilled(pos, 0.8f * opacity_len, 0xFFD8D8D8);

      // highlight
      draw_list->AddCircleFilled(pos,
          0.6f * opacity_len,
          ImGui::IsItemHovered() ? 0xFF051c33 : 0xFFD8D8D8);

      // setup interaction

      // delete opacity point
      if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
        if (i > 0 && i < m_tfnOpacityPoints.size() - 1) {
          m_tfnOpacityPoints.erase(m_tfnOpacityPoints.begin() + i);
          updateVolume();
        }
      } else if (ImGui::IsItemActive()) {
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        m_tfnOpacityPoints[i].y -= delta.y / height;
        m_tfnOpacityPoints[i].y =
            std::clamp(m_tfnOpacityPoints[i].y, 0.0f, 1.0f);
        if (i > 0 && i < m_tfnOpacityPoints.size() - 1) {
          m_tfnOpacityPoints[i].x += delta.x / width;
          m_tfnOpacityPoints[i].x = std::clamp(m_tfnOpacityPoints[i].x,
              m_tfnOpacityPoints[i - 1].x,
              m_tfnOpacityPoints[i + 1].x);
        }
        updateVolume();
      } else if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Double right click button to delete point\n"
            "Left click and drag to move point");
      }
    }
  }

  // draw background interaction
  ImGui::SetCursorScreenPos(
      ImVec2(canvas_x + margin, canvas_y - height - margin));
  ImGui::InvisibleButton("##tfn_palette_opacity", ImVec2(width, height));

  // add opacity point
  if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
    const float x = std::clamp(
        (mouse_x - canvas_x - margin - scroll_x) / (float)width, 0.f, 1.f);
    const float y = std::clamp(
        -(mouse_y - canvas_y + margin - scroll_y) / (float)height, 0.f, 1.f);
    const int idx = find_idx(m_tfnOpacityPoints, x);
    tsd::OpacityPoint pt(x, y);
    m_tfnOpacityPoints.insert(m_tfnOpacityPoints.begin() + idx, pt);
    m_tfnChanged = true;
  }
}

void TransferFunctionEditor::buildUI_opacityScale()
{
  bool opacityChanged =
      ImGui::SliderFloat("opacity scale", &m_globalOpacityScale, 0.f, 10.f);

  if (ImGui::Button("reset##opacity")) {
    m_globalOpacityScale = 1.f;
    opacityChanged = true;
  }

  if (opacityChanged)
    updateVolume();
}

void TransferFunctionEditor::buildUI_valueRange()
{
  ImGui::BeginDisabled(!m_volume);

  tsd::ui::buildUI_parameter(
      *m_volume, *m_volume->parameter("valueRange"), m_core->tsd.ctx);

  if (ImGui::Button("reset##valueRange") && m_volume) {
    tsd::SpatialField *field =
        m_volume->parameterValueAsObject<tsd::SpatialField>("value");
    if (field) {
      auto valueRange = field->computeValueRange();
      m_volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
    }
  }

  ImGui::EndDisabled();
}

std::vector<tsd::float4> TransferFunctionEditor::getSampledColorsAndOpacities(
    int numSamples)
{
  std::vector<tsd::float4> sampledColorsAndOpacities;
  sampledColorsAndOpacities.reserve(numSamples);

  const float dx = 1.f / (numSamples - 1);

  for (int i = 0; i < numSamples; i++) {
    sampledColorsAndOpacities.push_back(
        tsd::float4(tsd::detail::interpolateColor(*m_tfnColorPoints, i * dx),
            tsd::detail::interpolateOpacity(m_tfnOpacityPoints, i * dx)
                * m_globalOpacityScale));
  }

  return sampledColorsAndOpacities;
}

void TransferFunctionEditor::setMap(int selection)
{
  if (m_currentMap != selection)
    m_nextMap = selection;
}

void TransferFunctionEditor::setObjectPtrsFromSelectedObject()
{
  tsd::Volume *selectedVolume = nullptr;
  tsd::Object *selectedObject = m_core->tsd.selectedObject;

  if (!selectedVolume) {
    if (selectedObject && selectedObject->type() == ANARI_VOLUME)
      selectedVolume = (tsd::Volume *)selectedObject;
  }

  if (selectedVolume == nullptr) {
    m_volume = nullptr;
    m_colorMapArray = nullptr;
    return;
  }

  if (m_volume != selectedVolume) {
    m_volume = selectedVolume;
    m_colorMapArray = m_volume->parameterValueAsObject<tsd::Array>("color");
    updateTfnPaletteTexture();
  }
}

void TransferFunctionEditor::loadDefaultMaps()
{
  std::vector<tsd::ColorPoint> colors;

  // Jet
  colors.clear();

  colors.emplace_back(0.0f, 0.f, 0.f, 1.f);
  colors.emplace_back(0.3f, 0.f, 1.f, 1.f);
  colors.emplace_back(0.6f, 1.f, 1.f, 0.f);
  colors.emplace_back(1.0f, 1.f, 0.f, 0.f);

  m_tfnsColorPoints.push_back(colors);
  m_tfnsNames.push_back("Jet");

  // Cool to warm
  colors.clear();

  colors.emplace_back(0.0f, 0.231f, 0.298f, 0.752f);
  colors.emplace_back(0.25f, 0.552f, 0.690f, 0.996f);
  colors.emplace_back(0.5f, 0.866f, 0.866f, 0.866f);
  colors.emplace_back(0.75f, 0.956f, 0.603f, 0.486f);
  colors.emplace_back(1.0f, 0.705f, 0.015f, 0.149f);

  m_tfnsColorPoints.push_back(colors);
  m_tfnsNames.push_back("Cool to Warm");

  // Viridis
  colors.clear();

  float spacing = 1.f / 15;

  colors.emplace_back(0 * spacing, 0.267004, 0.004874, 0.329415);
  colors.emplace_back(1 * spacing, 0.282656, 0.100196, 0.42216);
  colors.emplace_back(2 * spacing, 0.277134, 0.185228, 0.489898);
  colors.emplace_back(3 * spacing, 0.253935, 0.265254, 0.529983);
  colors.emplace_back(4 * spacing, 0.221989, 0.339161, 0.548752);
  colors.emplace_back(5 * spacing, 0.190631, 0.407061, 0.556089);
  colors.emplace_back(6 * spacing, 0.163625, 0.471133, 0.558148);
  colors.emplace_back(7 * spacing, 0.139147, 0.533812, 0.555298);
  colors.emplace_back(8 * spacing, 0.120565, 0.596422, 0.543611);
  colors.emplace_back(9 * spacing, 0.134692, 0.658636, 0.517649);
  colors.emplace_back(10 * spacing, 0.20803, 0.718701, 0.472873);
  colors.emplace_back(11 * spacing, 0.327796, 0.77398, 0.40664);
  colors.emplace_back(12 * spacing, 0.477504, 0.821444, 0.318195);
  colors.emplace_back(13 * spacing, 0.647257, 0.8584, 0.209861);
  colors.emplace_back(14 * spacing, 0.82494, 0.88472, 0.106217);
  colors.emplace_back(15 * spacing, 0.993248, 0.906157, 0.143936);

  m_tfnsColorPoints.push_back(colors);
  m_tfnsNames.push_back("Viridis");

  // Black body radiation
  colors.clear();

  colors.emplace_back(0.0f, 0.f, 0.f, 0.f);
  colors.emplace_back(0.3f, 1.f, 0.f, 0.f);
  colors.emplace_back(0.6f, 1.f, 1.f, 0.f);
  colors.emplace_back(1.0f, 1.f, 1.f, 1.f);

  m_tfnsColorPoints.push_back(colors);

  m_tfnsNames.push_back("Black-Body Radiation");

  // Inferno
  colors.clear();

  colors.emplace_back(0.0f, 0.f, 0.f, 0.f);
  colors.emplace_back(0.25f, 0.25f, 0.f, 0.25f);
  colors.emplace_back(0.5f, 1.f, 0.f, 0.f);
  colors.emplace_back(0.75f, 1.f, 1.f, 0.f);
  colors.emplace_back(1.0f, 1.f, 1.f, 1.f);

  m_tfnsColorPoints.push_back(colors);

  m_tfnsNames.push_back("Inferno");

  // Ice Fire
  colors.clear();

  spacing = 1.f / 16;

  colors.emplace_back(0 * spacing, 0, 0, 0);
  colors.emplace_back(1 * spacing, 0, 0.120394, 0.302678);
  colors.emplace_back(2 * spacing, 0, 0.216587, 0.524575);
  colors.emplace_back(3 * spacing, 0.0552529, 0.345022, 0.659495);
  colors.emplace_back(4 * spacing, 0.128054, 0.492592, 0.720287);
  colors.emplace_back(5 * spacing, 0.188952, 0.641306, 0.792096);
  colors.emplace_back(6 * spacing, 0.327672, 0.784939, 0.873426);
  colors.emplace_back(7 * spacing, 0.60824, 0.892164, 0.935546);
  colors.emplace_back(8 * spacing, 0.881376, 0.912184, 0.818097);
  colors.emplace_back(9 * spacing, 0.9514, 0.835615, 0.449271);
  colors.emplace_back(10 * spacing, 0.904479, 0.690486, 0);
  colors.emplace_back(11 * spacing, 0.854063, 0.510857, 0);
  colors.emplace_back(12 * spacing, 0.777096, 0.330175, 0.000885023);
  colors.emplace_back(13 * spacing, 0.672862, 0.139086, 0.00270085);
  colors.emplace_back(14 * spacing, 0.508812, 0, 0);
  colors.emplace_back(15 * spacing, 0.299413, 0.000366217, 0.000549325);
  colors.emplace_back(16 * spacing, 0.0157473, 0.00332647, 0);

  m_tfnsColorPoints.push_back(colors);

  m_tfnsNames.push_back("Ice Fire");

  // Grayscale
  colors.clear();

  colors.emplace_back(0.f, 1.f, 1.f, 1.f);
  colors.emplace_back(1.f, 1.f, 1.f, 1.f);

  m_tfnsColorPoints.push_back(colors);

  m_tfnsNames.push_back("Grayscale");
};

void TransferFunctionEditor::updateVolume()
{
  if (!m_colorMapArray) {
    tsd::logError("[tfn_editor] No color map array, cannot update volume!");
    return;
  }
  auto co = getSampledColorsAndOpacities(m_colorMapArray->size());
  auto *colorMap = m_colorMapArray->mapAs<tsd::float4>();
  std::copy(co.begin(), co.end(), colorMap);
  m_colorMapArray->unmap();
}

void TransferFunctionEditor::updateTfnPaletteTexture()
{
  if (!m_colorMapArray) {
    tsd::logError("[tfn_editor] No color map array, cannot update SDL image!");
    return;
  }
  auto width = m_colorMapArray->size();
  if (width != m_tfnPaletteWidth)
    resizeTfnPaletteTexture(width);

  std::vector<tsd::float4> palette = getSampledColorsAndOpacities(width);
  std::vector<tsd::float3> rgb(width, tsd::float3(1.f, 0.f, 0.f));
  std::transform(palette.begin(), palette.end(), rgb.begin(), [](auto &c) {
    return tsd::float3(c.x, c.y, c.z);
  });

  SDL_UpdateTexture(
      m_tfnPaletteTexture, nullptr, rgb.data(), width * sizeof(tsd::float3));
}

void TransferFunctionEditor::resizeTfnPaletteTexture(size_t width)
{
  if (m_tfnPaletteTexture)
    SDL_DestroyTexture(m_tfnPaletteTexture);
  m_tfnPaletteTexture = SDL_CreateTexture(m_core->application->sdlRenderer(),
      SDL_PIXELFORMAT_RGB96_FLOAT,
      SDL_TEXTUREACCESS_STATIC,
      int(width),
      1);
  m_tfnPaletteWidth = width;
}

} // namespace tsd_viewer
