// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferFunctionEditor.h"
// std
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace tsd_viewer {

template <typename T>
static int find_idx(const std::vector<T> &A, float p)
{
  auto found =
      std::upper_bound(A.begin(), A.end(), T(p), [](const T &a, const T &b) {
        return a.x < b.x;
      });
  return std::distance(A.begin(), found);
}

static float lerp(const float &l,
    const float &r,
    const float &pl,
    const float &pr,
    const float &p)
{
  const float dl = std::abs(pr - pl) > 0.0001f ? (p - pl) / (pr - pl) : 0.f;
  const float dr = 1.f - dl;
  return l * dr + r * dl;
}

TransferFunctionEditor::TransferFunctionEditor(
    AppCore *ctx, const char *name)
    : Window(name, true), m_core(ctx)
{
  loadDefaultMaps();

  m_tfnColorPoints = &(m_tfnsColorPoints[m_currentMap]);
  m_tfnOpacityPoints = &(m_tfnsOpacityPoints[m_currentMap]);
  m_tfnEditable = m_tfnsEditable[m_currentMap];
}

TransferFunctionEditor::~TransferFunctionEditor()
{
  if (tfnPaletteTexture)
    glDeleteTextures(1, &tfnPaletteTexture);
}

void TransferFunctionEditor::buildUI()
{
  if (m_tfnChanged) {
    updateTfnPaletteTexture();
    triggerUpdateCallback();
    m_tfnChanged = false;
  }

  std::vector<const char *> names(m_tfnsNames.size(), nullptr);
  std::transform(m_tfnsNames.begin(),
      m_tfnsNames.end(),
      names.begin(),
      [](const std::string &t) { return t.c_str(); });

  int newMap = m_currentMap;
  if (ImGui::Combo("color map", &newMap, names.data(), names.size()))
    setMap(newMap);

  ImGui::Separator();

  drawEditor();

  ImGui::Separator();

  m_tfnChanged |=
      ImGui::SliderFloat("opacity scale", &m_globalOpacityScale, 0.f, 10.f);

  if (ImGui::Button("reset##opacity")) {
    m_globalOpacityScale = 1.f;
    m_tfnChanged = true;
  }

  ImGui::Separator();

  m_tfnChanged |= ImGui::DragFloatRange2("value range",
      &m_valueRange.x,
      &m_valueRange.y,
      0.1f,
      -10000.f,
      10000.0f,
      "Min: %.7f",
      "Max: %.7f");

  if (ImGui::Button("reset##valueRange")) {
    m_valueRange = m_defaultValueRange;
    m_tfnChanged = true;
  }
}

void TransferFunctionEditor::setUpdateCallback(TFUpdateCallback cb)
{
  m_updateCallback = cb;
  triggerUpdateCallback();
}

void TransferFunctionEditor::triggerUpdateCallback()
{
  if (m_updateCallback)
    m_updateCallback(getValueRange(), getSampledColorsAndOpacities());
}

void TransferFunctionEditor::setValueRange(const tsd::float2 &vr)
{
  m_valueRange = m_defaultValueRange = vr;
  m_tfnChanged = true;
}

tsd::float2 TransferFunctionEditor::getValueRange()
{
  return m_valueRange;
}

std::vector<tsd::float4> TransferFunctionEditor::getSampledColorsAndOpacities(
    int numSamples)
{
  std::vector<tsd::float4> sampledColorsAndOpacities;

  const float dx = 1.f / (numSamples - 1);

  for (int i = 0; i < numSamples; i++) {
    sampledColorsAndOpacities.push_back(
        tsd::float4(interpolateColor(*m_tfnColorPoints, i * dx),
            interpolateOpacity(*m_tfnOpacityPoints, i * dx)
                * m_globalOpacityScale));
  }

  return sampledColorsAndOpacities;
}

void TransferFunctionEditor::loadDefaultMaps()
{
  // same opacities for all maps
  std::vector<OpacityPoint> opacities;

  opacities.emplace_back(0.f, 0.f);
  opacities.emplace_back(1.f, 1.f);

  std::vector<ColorPoint> colors;

  // Jet
  colors.clear();

  colors.emplace_back(0.0f, 0.f, 0.f, 1.f);
  colors.emplace_back(0.3f, 0.f, 1.f, 1.f);
  colors.emplace_back(0.6f, 1.f, 1.f, 0.f);
  colors.emplace_back(1.0f, 1.f, 0.f, 0.f);

  m_tfnsColorPoints.push_back(colors);
  m_tfnsOpacityPoints.push_back(opacities);

  m_tfnsEditable.push_back(false);
  m_tfnsNames.push_back("Jet");

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
  m_tfnsOpacityPoints.push_back(opacities);

  m_tfnsEditable.push_back(false);
  m_tfnsNames.push_back("Viridis");

  // Black body radiation
  colors.clear();

  colors.emplace_back(0.0f, 0.f, 0.f, 0.f);
  colors.emplace_back(0.3f, 1.f, 0.f, 0.f);
  colors.emplace_back(0.6f, 1.f, 1.f, 0.f);
  colors.emplace_back(1.0f, 1.f, 1.f, 1.f);

  m_tfnsColorPoints.push_back(colors);
  m_tfnsOpacityPoints.push_back(opacities);

  m_tfnsEditable.push_back(false);
  m_tfnsNames.push_back("Black-Body Radiation");

  // Inferno
  colors.clear();

  colors.emplace_back(0.0f, 0.f, 0.f, 0.f);
  colors.emplace_back(0.25f, 0.25f, 0.f, 0.25f);
  colors.emplace_back(0.5f, 1.f, 0.f, 0.f);
  colors.emplace_back(0.75f, 1.f, 1.f, 0.f);
  colors.emplace_back(1.0f, 1.f, 1.f, 1.f);

  m_tfnsColorPoints.push_back(colors);
  m_tfnsOpacityPoints.push_back(opacities);

  m_tfnsEditable.push_back(false);
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
  m_tfnsOpacityPoints.push_back(opacities);

  m_tfnsEditable.push_back(false);
  m_tfnsNames.push_back("Ice Fire");

  // Grayscale
  colors.clear();

  colors.emplace_back(0.f, 1.f, 1.f, 1.f);
  colors.emplace_back(1.f, 1.f, 1.f, 1.f);

  m_tfnsColorPoints.push_back(colors);
  m_tfnsOpacityPoints.push_back(opacities);

  m_tfnsEditable.push_back(false);
  m_tfnsNames.push_back("Grayscale");
};

void TransferFunctionEditor::setMap(int selection)
{
  if (m_currentMap != selection) {
    m_currentMap = selection;
    m_tfnColorPoints = &(m_tfnsColorPoints[selection]);
#if 0
    m_tfnOpacityPoints = &(m_tfnsOpacityPoints[selection]);
#endif
    m_tfnEditable = m_tfnsEditable[selection];
    m_tfnChanged = true;
  }
}

tsd::float3 TransferFunctionEditor::interpolateColor(
    const std::vector<ColorPoint> &controlPoints, float x)
{
  auto first = controlPoints.front();
  if (x <= first.x)
    return tsd::float3(first.y, first.z, first.w);

  for (uint32_t i = 1; i < controlPoints.size(); i++) {
    auto current = controlPoints[i];
    auto previous = controlPoints[i - 1];
    if (x <= current.x) {
      const float t = (x - previous.x) / (current.x - previous.x);
      return (1.0f - t) * tsd::float3(previous.y, previous.z, previous.w)
          + t * tsd::float3(current.y, current.z, current.w);
    }
  }

  auto last = controlPoints.back();
  return tsd::float3(last.x, last.y, last.z);
}

float TransferFunctionEditor::interpolateOpacity(
    const std::vector<OpacityPoint> &controlPoints, float x)

{
  auto first = controlPoints.front();
  if (x <= first.x)
    return first.y;

  for (uint32_t i = 1; i < controlPoints.size(); i++) {
    auto current = controlPoints[i];
    auto previous = controlPoints[i - 1];
    if (x <= current.x) {
      const float t = (x - previous.x) / (current.x - previous.x);
      return (1.0 - t) * previous.y + t * current.y;
    }
  }

  auto last = controlPoints.back();
  return last.y;
}

void TransferFunctionEditor::updateTfnPaletteTexture()
{
  const size_t textureWidth = 256, textureHeight = 1;

  // backup currently bound texture
  GLint prevBinding = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevBinding);

  // create transfer function palette texture if it doesn't exist
  if (!tfnPaletteTexture) {
    glGenTextures(1, &tfnPaletteTexture);
    glBindTexture(GL_TEXTURE_2D, tfnPaletteTexture);
    glTexImage2D(GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        textureWidth,
        textureHeight,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  }

  // sample the palette then upload the data
  std::vector<tsd::float4> palette = getSampledColorsAndOpacities(textureWidth);

  // save palette to texture
  glBindTexture(GL_TEXTURE_2D, tfnPaletteTexture);
  glTexImage2D(GL_TEXTURE_2D,
      0,
      GL_RGB,
      textureWidth,
      textureHeight,
      0,
      GL_RGBA,
      GL_FLOAT,
      static_cast<const void *>(palette.data()));

  // restore previously bound texture
  if (prevBinding)
    glBindTexture(GL_TEXTURE_2D, prevBinding);
}

void TransferFunctionEditor::drawEditor()
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
  ImGui::Image(
      reinterpret_cast<void *>(tfnPaletteTexture), ImVec2(width, height));

  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  {
    std::vector<ImVec2> polyline;
    polyline.reserve(4);
    for (int i = 0; i < m_tfnOpacityPoints->size() - 1; ++i) {
      polyline.clear();
      polyline.emplace_back(
          canvas_x + margin + (*m_tfnOpacityPoints)[i].x * width,
          canvas_y + height);
      polyline.emplace_back(
          canvas_x + margin + (*m_tfnOpacityPoints)[i].x * width,
          canvas_y + height - (*m_tfnOpacityPoints)[i].y * height);
      polyline.emplace_back(
          canvas_x + margin + (*m_tfnOpacityPoints)[i + 1].x * width + 1,
          canvas_y + height - (*m_tfnOpacityPoints)[i + 1].y * height);
      polyline.emplace_back(
          canvas_x + margin + (*m_tfnOpacityPoints)[i + 1].x * width + 1,
          canvas_y + height);
      draw_list->AddConvexPolyFilled(
          polyline.data(), polyline.size(), 0xc8d8d8d8);
    }
  }
  canvas_y += height + margin;
  canvas_avail_y -= height + margin;

  // draw color control points
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));

  if (m_tfnEditable) {
    // draw circle background
    draw_list->AddRectFilled(ImVec2(canvas_x + margin, canvas_y - margin),
        ImVec2(canvas_x + margin + width, canvas_y - margin + 2.5 * color_len),
        0xFF474646);

    // draw circles
    for (int i = m_tfnColorPoints->size() - 1; i >= 0; --i) {
      const ImVec2 pos(
          canvas_x + width * (*m_tfnColorPoints)[i].x + margin, canvas_y);
      ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));

      // white background
      draw_list->AddTriangleFilled(ImVec2(pos.x - 0.5f * color_len, pos.y),
          ImVec2(pos.x + 0.5f * color_len, pos.y),
          ImVec2(pos.x, pos.y - color_len),
          0xFFD8D8D8);
      draw_list->AddCircleFilled(
          ImVec2(pos.x, pos.y + 0.5f * color_len), color_len, 0xFFD8D8D8);

      // draw picker
      ImVec4 picked_color = ImColor((*m_tfnColorPoints)[i].y,
          (*m_tfnColorPoints)[i].z,
          (*m_tfnColorPoints)[i].w,
          1.f);
      ImGui::SetCursorScreenPos(
          ImVec2(pos.x - color_len, pos.y + 1.5f * color_len));
      if (ImGui::ColorEdit4(("##ColorPicker" + std::to_string(i)).c_str(),
              (float *)&picked_color,
              ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs
                  | ImGuiColorEditFlags_NoLabel
                  | ImGuiColorEditFlags_AlphaPreview
                  | ImGuiColorEditFlags_NoOptions
                  | ImGuiColorEditFlags_NoTooltip)) {
        (*m_tfnColorPoints)[i].y = picked_color.x;
        (*m_tfnColorPoints)[i].z = picked_color.y;
        (*m_tfnColorPoints)[i].w = picked_color.z;
        m_tfnChanged = true;
      }
      if (ImGui::IsItemHovered()) {
        // convert float color to char
        int cr = static_cast<int>(picked_color.x * 255);
        int cg = static_cast<int>(picked_color.y * 255);
        int cb = static_cast<int>(picked_color.z * 255);

        // setup tooltip
        ImGui::BeginTooltip();
        ImVec2 sz(
            ImGui::GetFontSize() * 4 + ImGui::GetStyle().FramePadding.y * 2,
            ImGui::GetFontSize() * 4 + ImGui::GetStyle().FramePadding.y * 2);
        ImGui::ColorButton("##PreviewColor",
            picked_color,
            ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_AlphaPreview,
            sz);
        ImGui::SameLine();
        ImGui::Text(
            "Left click to edit\n"
            "HEX: #%02X%02X%02X\n"
            "RGB: [%3d,%3d,%3d]\n(%.2f, %.2f, %.2f)",
            cr,
            cg,
            cb,
            cr,
            cg,
            cb,
            picked_color.x,
            picked_color.y,
            picked_color.z);
        ImGui::EndTooltip();
      }
    }

    for (int i = 0; i < m_tfnColorPoints->size(); ++i) {
      const ImVec2 pos(
          canvas_x + width * (*m_tfnColorPoints)[i].x + margin, canvas_y);

      // draw button
      ImGui::SetCursorScreenPos(
          ImVec2(pos.x - color_len, pos.y - 0.5 * color_len));
      ImGui::InvisibleButton(("##ColorControl-" + std::to_string(i)).c_str(),
          ImVec2(2.f * color_len, 2.f * color_len));

      // dark highlight
      ImGui::SetCursorScreenPos(ImVec2(pos.x - color_len, pos.y));
      draw_list->AddCircleFilled(ImVec2(pos.x, pos.y + 0.5f * color_len),
          0.5f * color_len,
          ImGui::IsItemHovered() ? 0xFF051C33 : 0xFFBCBCBC);

      // delete color point
      if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
        if (i > 0 && i < m_tfnColorPoints->size() - 1) {
          m_tfnColorPoints->erase(m_tfnColorPoints->begin() + i);
          m_tfnChanged = true;
        }
      }

      // drag color control point
      else if (ImGui::IsItemActive()) {
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        if (i > 0 && i < m_tfnColorPoints->size() - 1) {
          (*m_tfnColorPoints)[i].x += delta.x / width;
          (*m_tfnColorPoints)[i].x = std::clamp((*m_tfnColorPoints)[i].x,
              (*m_tfnColorPoints)[i - 1].x,
              (*m_tfnColorPoints)[i + 1].x);
        }

        m_tfnChanged = true;
      }
    }
  }

  // draw opacity control points
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  {
    // draw circles
    for (int i = 0; i < m_tfnOpacityPoints->size(); ++i) {
      const ImVec2 pos(canvas_x + width * (*m_tfnOpacityPoints)[i].x + margin,
          canvas_y - height * (*m_tfnOpacityPoints)[i].y - margin);
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
        if (i > 0 && i < m_tfnOpacityPoints->size() - 1) {
          m_tfnOpacityPoints->erase(m_tfnOpacityPoints->begin() + i);
          m_tfnChanged = true;
        }
      } else if (ImGui::IsItemActive()) {
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        (*m_tfnOpacityPoints)[i].y -= delta.y / height;
        (*m_tfnOpacityPoints)[i].y =
            std::clamp((*m_tfnOpacityPoints)[i].y, 0.0f, 1.0f);
        if (i > 0 && i < m_tfnOpacityPoints->size() - 1) {
          (*m_tfnOpacityPoints)[i].x += delta.x / width;
          (*m_tfnOpacityPoints)[i].x = std::clamp((*m_tfnOpacityPoints)[i].x,
              (*m_tfnOpacityPoints)[i - 1].x,
              (*m_tfnOpacityPoints)[i + 1].x);
        }
        m_tfnChanged = true;
      } else if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Double right click button to delete point\n"
            "Left click and drag to move point");
      }
    }
  }

  // draw background interaction
  ImGui::SetCursorScreenPos(ImVec2(canvas_x + margin, canvas_y - margin));
  ImGui::InvisibleButton("##tfn_palette_color", ImVec2(width, 2.5 * color_len));

  // add color point
  if (m_tfnEditable && ImGui::IsMouseDoubleClicked(0)
      && ImGui::IsItemHovered()) {
    const float p = std::clamp(
        (mouse_x - canvas_x - margin - scroll_x) / (float)width, 0.f, 1.f);
    const int ir = find_idx(*m_tfnColorPoints, p);
    const int il = ir - 1;
    const float pr = (*m_tfnColorPoints)[ir].x;
    const float pl = (*m_tfnColorPoints)[il].x;
    const float r =
        lerp((*m_tfnColorPoints)[il].y, (*m_tfnColorPoints)[ir].y, pl, pr, p);
    const float g =
        lerp((*m_tfnColorPoints)[il].z, (*m_tfnColorPoints)[ir].z, pl, pr, p);
    const float b =
        lerp((*m_tfnColorPoints)[il].w, (*m_tfnColorPoints)[ir].w, pl, pr, p);
    ColorPoint pt(p, r, g, b);
    m_tfnColorPoints->insert(m_tfnColorPoints->begin() + ir, pt);
    m_tfnChanged = true;
  }

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Double left click to add new color point");

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
    const int idx = find_idx(*m_tfnOpacityPoints, x);
    OpacityPoint pt(x, y);
    m_tfnOpacityPoints->insert(m_tfnOpacityPoints->begin() + idx, pt);
    m_tfnChanged = true;
  }

  // update cursors
  canvas_y += 4.f * color_len + margin;
  canvas_avail_y -= 4.f * color_len + margin;

  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
}

} // namespace tsd_viewer