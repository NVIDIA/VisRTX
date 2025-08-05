// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferFunctionEditor.h"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// tsd_app
#include "tsd/app/Core.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// std
#include <algorithm>
#include <fstream>
#include <sstream>

namespace tsd::ui::imgui {

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

TransferFunctionEditor::TransferFunctionEditor(
    Application *app, const char *name)
    : Window(app, name)
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
  static size_t lastSize = 0;

  // Refresh names list if new colormaps have been added
  if (names.size() != m_tfnsNames.size() || lastSize != m_tfnsNames.size()) {
    names.resize(m_tfnsNames.size());
    std::transform(m_tfnsNames.begin(),
        m_tfnsNames.end(),
        names.begin(),
        [](const std::string &t) { return t.c_str(); });
    lastSize = m_tfnsNames.size();
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
    tsd::core::OpacityPoint pt(x, y);
    m_tfnOpacityPoints.insert(m_tfnOpacityPoints.begin() + idx, pt);
    updateVolume();
  }
}

void TransferFunctionEditor::buildUI_opacityScale()
{
  tsd::ui::buildUI_parameter(
      *m_volume, *m_volume->parameter("opacity"), appCore()->tsd.ctx);
}

void TransferFunctionEditor::buildUI_valueRange()
{
  ImGui::BeginDisabled(!m_volume);

  tsd::ui::buildUI_parameter(
      *m_volume, *m_volume->parameter("valueRange"), appCore()->tsd.ctx);

  if (ImGui::Button("reset##valueRange") && m_volume) {
    auto *field =
        m_volume->parameterValueAsObject<tsd::core::SpatialField>("value");
    if (field) {
      auto valueRange = field->computeValueRange();
      m_volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
    }
  }

  ImGui::SameLine();
  if (ImGui::Button("Load")) {
    m_currentColormapFilename.clear();
    getTransferFunctionFilenameFromDialog(m_currentColormapFilename);
  }

  ImGui::SameLine();
  if (ImGui::Button("Save")) {
    m_saveColormapFilename.clear();
    getTransferFunctionFilenameFromDialog(m_saveColormapFilename, true);
  }

  if (!m_currentColormapFilename.empty()) {
    // Extract filename without path for display name
    size_t lastSlash = m_currentColormapFilename.find_last_of("/\\");
    std::string displayName = (lastSlash != std::string::npos)
        ? m_currentColormapFilename.substr(lastSlash + 1)
        : m_currentColormapFilename;

    // Detect file type and load accordingly
    bool loaded = false;
    if (displayName.size() > 4
        && displayName.substr(displayName.size() - 4) == ".1dt") {
      // Remove .1dt extension for display name
      displayName = displayName.substr(0, displayName.size() - 4);
      loadColormapFrom1dt(m_currentColormapFilename, displayName);
      loaded = true;
    } else if (displayName.size() > 5
        && displayName.substr(displayName.size() - 5) == ".json") {
      // Remove .json extension for display name
      displayName = displayName.substr(0, displayName.size() - 5);
      loadColormapFromParaview(m_currentColormapFilename, displayName);
      loaded = true;
    }

    // Set the loaded colormap as active and update volume immediately
    if (loaded && !m_tfnsNames.empty()) {
      // Find the index of the loaded colormap (might not be the last one if it
      // replaced an existing one)
      int loadedMapIndex = -1;
      for (size_t i = 0; i < m_tfnsNames.size(); ++i) {
        if (m_tfnsNames[i] == displayName) {
          loadedMapIndex = static_cast<int>(i);
          break;
        }
      }

      if (loadedMapIndex >= 0) {
        setMap(loadedMapIndex);

        // Force immediate update of the volume with the new colormap
        if (m_volume) {
          m_currentMap = loadedMapIndex;
          m_tfnColorPoints = &(m_tfnsColorPoints[m_currentMap]);
          updateVolume();
          updateTfnPaletteTexture();
        }
      }
    }
    m_currentColormapFilename.clear();
  }

  if (!m_saveColormapFilename.empty()) {
    // Detect file extension and save accordingly
    if (m_saveColormapFilename.size() > 5
        && m_saveColormapFilename.substr(m_saveColormapFilename.size() - 5)
            == ".json") {
      saveColormapToParaview(m_saveColormapFilename);
    } else {
      // Default to .1dt format
      saveColormapTo1dt(m_saveColormapFilename);
    }
    m_saveColormapFilename.clear();
  }

  ImGui::EndDisabled();
}

std::vector<tsd::math::float4>
TransferFunctionEditor::getSampledColorsAndOpacities(int numSamples)
{
  std::vector<tsd::math::float4> sampledColorsAndOpacities;
  sampledColorsAndOpacities.reserve(numSamples);

  const float dx = 1.f / (numSamples - 1);

  for (int i = 0; i < numSamples; i++) {
    const bool interpolateColor = m_currentMap != 0;
    tsd::math::float3 color(0.f);
    if (interpolateColor)
      color = tsd::core::detail::interpolateColor(*m_tfnColorPoints, i * dx);
    else {
      const auto co = (*m_tfnColorPoints)[i];
      color = tsd::math::float3(co.x, co.y, co.z);
    }
    const auto opacity =
        tsd::core::detail::interpolateOpacity(m_tfnOpacityPoints, i * dx);
    sampledColorsAndOpacities.push_back(tsd::math::float4(color, opacity));
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
  tsd::core::Volume *selectedVolume = nullptr;
  tsd::core::Object *selectedObject = appCore()->tsd.selectedObject;

  if (!selectedVolume) {
    if (selectedObject && selectedObject->type() == ANARI_VOLUME)
      selectedVolume = (tsd::core::Volume *)selectedObject;
  }

  if (selectedVolume == nullptr) {
    m_volume = nullptr;
    m_colorMapArray = nullptr;
    return;
  }

  if (m_volume != selectedVolume) {
    setMap(0);

    m_volume = selectedVolume;
    m_colorMapArray =
        m_volume->parameterValueAsObject<tsd::core::Array>("color");

    auto &cm = m_tfnsColorPoints[0];
    cm.resize(m_colorMapArray->size());
    auto *colorsIn = m_colorMapArray->dataAs<tsd::math::float4>();
    std::copy(colorsIn, colorsIn + m_colorMapArray->size(), cm.begin());

    // Get opacity control points from volume //

    anari::DataType type = ANARI_UNKNOWN;
    const tsd::math::float2 *opacityPoints = nullptr;
    size_t size = 0;
    m_volume->getMetadataArray(
        "opacityControlPoints", &type, (const void **)&opacityPoints, &size);
    if (type == ANARI_FLOAT32_VEC2 && size > 0) {
      tsd::core::logStatus("[tfn_editor] Receiving opacity control points");
      m_tfnOpacityPoints.resize(size);
      std::copy(
          opacityPoints, opacityPoints + size, m_tfnOpacityPoints.begin());
    } else {
      tsd::core::logWarning(
          "[tfn_editor] No metadata for opacity control points found!");
      m_tfnOpacityPoints.resize(2);
      m_tfnOpacityPoints[0] = {0.f, 0.f};
      m_tfnOpacityPoints[1] = {1.f, 1.f};
    }

    updateTfnPaletteTexture();
  }
}

void TransferFunctionEditor::loadDefaultMaps()
{
  std::vector<tsd::core::ColorPoint> colors;

  // Incoming color map
  m_tfnsNames.push_back("{from volume}");
  m_tfnsColorPoints.push_back(colors);

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

void TransferFunctionEditor::loadColormapFrom1dt(
    const std::string &filepath, const std::string &name)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    tsd::core::logError(
        ("[tfn_editor] Failed to open 1dt file: " + filepath).c_str());
    return;
  }

  std::string line;
  std::vector<tsd::core::ColorPoint> colors;
  std::vector<tsd::core::OpacityPoint> opacityPoints;

  // Read all lines and try to parse as r g b a values
  // The 1dt format is simply one color per line without a count header
  int entryIndex = 0;
  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    float r, g, b, a;

    if (!(iss >> r >> g >> b >> a)) {
      tsd::core::logError(("[tfn_editor] Failed to parse color entry at line "
          + std::to_string(entryIndex + 1))
              .c_str());
      continue;
    }

    // Store as ColorPoint: (position, r, g, b)
    // Calculate normalized position (0.0 to 1.0) based on entry index
    colors.emplace_back(
        0.0f, r, g, b); // We'll fix positions after reading all entries

    // Store alpha as OpacityPoint: (position, alpha)
    opacityPoints.emplace_back(
        0.0f, a); // We'll fix positions after reading all entries

    entryIndex++;
  }

  // Smart positioning: detect significant opacity changes to create meaningful
  // control points
  if (colors.size() > 1) {
    // First pass: evenly distribute positions
    for (size_t i = 0; i < colors.size(); ++i) {
      float position =
          static_cast<float>(i) / static_cast<float>(colors.size() - 1);
      colors[i].x = position;
      opacityPoints[i].x = position;
    }

    // Second pass: reduce control points by removing redundant ones
    // Keep points where opacity changes significantly (>5% change)
    const float minOpacityChange = 0.05f;
    std::vector<tsd::core::OpacityPoint> smartOpacityPoints;

    if (!opacityPoints.empty()) {
      smartOpacityPoints.push_back(opacityPoints[0]); // Always keep first point

      for (size_t i = 1; i < opacityPoints.size() - 1; ++i) {
        float prevOpacity = opacityPoints[i - 1].y;
        float currOpacity = opacityPoints[i].y;
        float nextOpacity = opacityPoints[i + 1].y;

        // Keep point if it's a local min/max or has significant change
        bool isLocalMinMax =
            (currOpacity > prevOpacity && currOpacity > nextOpacity)
            || (currOpacity < prevOpacity && currOpacity < nextOpacity);
        bool hasSignificantChange =
            std::abs(currOpacity - prevOpacity) > minOpacityChange;

        if (isLocalMinMax || hasSignificantChange) {
          smartOpacityPoints.push_back(opacityPoints[i]);
        }
      }

      smartOpacityPoints.push_back(
          opacityPoints.back()); // Always keep last point

      // Only use smart points if we reduced the count significantly
      if (smartOpacityPoints.size() < opacityPoints.size() * 0.8f) {
        opacityPoints = smartOpacityPoints;
        tsd::core::logStatus(
            ("[tfn_editor] Reduced opacity control points from "
                + std::to_string(colors.size()) + " to "
                + std::to_string(opacityPoints.size()) + " points")
                .c_str());
      }
    }
  } else if (colors.size() == 1) {
    colors[0].x = 0.5f; // Single color at middle position
    opacityPoints[0].x = 0.5f;
  }

  file.close();

  if (colors.empty()) {
    tsd::core::logError(
        "[tfn_editor] No valid color entries found in 1dt file");
    return;
  }

  // Check if colormap with this name already exists and replace it
  int existingIndex = -1;
  for (size_t i = 0; i < m_tfnsNames.size(); ++i) {
    if (m_tfnsNames[i] == name) {
      existingIndex = static_cast<int>(i);
      break;
    }
  }

  int newMapIndex;
  if (existingIndex >= 0) {
    // Replace existing colormap
    m_tfnsColorPoints[existingIndex] = colors;
    newMapIndex = existingIndex;
    tsd::core::logStatus(
        ("[tfn_editor] Replaced existing colormap '" + name + "'").c_str());
  } else {
    // Add new colormap
    m_tfnsColorPoints.push_back(colors);
    m_tfnsNames.push_back(name);
    newMapIndex = m_tfnsColorPoints.size() - 1;
  }

  // Update opacity control points with alpha values from the file
  if (!opacityPoints.empty()) {
    m_tfnOpacityPoints = opacityPoints;

    // Set the loaded colormap as active
    m_currentMap = newMapIndex;
    m_nextMap = newMapIndex;
    m_tfnColorPoints = &(m_tfnsColorPoints[newMapIndex]);

    if (m_volume) {
      updateVolume();
      updateTfnPaletteTexture();
    }
  }

  tsd::core::logStatus(
      ("[tfn_editor] Successfully loaded colormap '" + name + "' from "
          + filepath + " (" + std::to_string(colors.size()) + " colors)")
          .c_str());
}

void TransferFunctionEditor::loadColormapFromParaview(
    const std::string &filepath, const std::string &name)
{
  const size_t colorDepth = 4;
  std::ifstream file(filepath);
  if (!file.is_open()) {
    tsd::core::logError(
        ("[tfn_editor] Failed to open Paraview JSON file: " + filepath)
            .c_str());
    return;
  }

  // Read entire file content
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  std::string jsonContent = buffer.str();

  // Basic JSON parsing for Paraview colormap format
  // Look for "RGBPoints" array
  size_t rgbPointsPos = jsonContent.find("\"RGBPoints\"");
  if (rgbPointsPos == std::string::npos) {
    tsd::core::logError(
        "[tfn_editor] No RGBPoints found in Paraview JSON file");
    return;
  }

  // Also look for "Points" array which might contain alpha values
  size_t pointsPos = jsonContent.find("\"Points\"");
  bool hasAlphaPoints = (pointsPos != std::string::npos);

  // Find the opening bracket of the RGBPoints array
  size_t arrayStart = jsonContent.find("[", rgbPointsPos);
  if (arrayStart == std::string::npos) {
    tsd::core::logError(
        "[tfn_editor] Invalid RGBPoints array format in Paraview JSON file");
    return;
  }

  // Find the closing bracket of the RGBPoints array
  int bracketCount = 0;
  size_t arrayEnd = arrayStart;
  for (size_t i = arrayStart; i < jsonContent.length(); ++i) {
    if (jsonContent[i] == '[')
      bracketCount++;
    else if (jsonContent[i] == ']') {
      bracketCount--;
      if (bracketCount == 0) {
        arrayEnd = i;
        break;
      }
    }
  }

  if (arrayEnd == arrayStart) {
    tsd::core::logError(
        "[tfn_editor] Could not find end of RGBPoints array in Paraview JSON file");
    return;
  }

  // Extract the array content
  std::string arrayContent =
      jsonContent.substr(arrayStart + 1, arrayEnd - arrayStart - 1);

  // Parse the numbers from the array
  std::vector<float> values;
  std::stringstream ss(arrayContent);
  std::string token;

  while (std::getline(ss, token, ',')) {
    // Remove whitespace and newlines
    token.erase(0, token.find_first_not_of(" \t\n\r"));
    token.erase(token.find_last_not_of(" \t\n\r") + 1);

    if (!token.empty()) {
      try {
        float val = std::stof(token);
        values.push_back(val);
      } catch (const std::exception &e) {
        // Skip non-numeric tokens
        continue;
      }
    }
  }

  // RGBPoints should have groups of 4 values: [value, r, g, b, value, r, g, b,
  // ...]
  if (values.size() % colorDepth != 0) {
    tsd::core::logError("[tfn_editor] Invalid RGBPoints format");
    return;
  }

  if (values.empty()) {
    tsd::core::logError(
        "[tfn_editor] No valid color entries found in Paraview JSON file");
    return;
  }

  std::vector<tsd::core::ColorPoint> colors;
  colors.reserve(values.size() / colorDepth);
  std::vector<tsd::core::OpacityPoint> opacityPoints;
  opacityPoints.reserve(values.size() / colorDepth);

  // Find min/max data values for normalization
  float minVal = values[0];
  float maxVal = values[0];
  for (size_t i = 0; i < values.size(); i += colorDepth) {
    minVal = std::min(minVal, values[i]);
    maxVal = std::max(maxVal, values[i]);
  }

  float range = maxVal - minVal;
  if (range == 0.0f)
    range = 1.0f; // Avoid division by zero

  // Convert to ColorPoint format
  const auto dx = 1.0f / (values.size() - colorDepth);
  for (size_t i = 0; i < values.size(); i += colorDepth) {
    float dataValue = values[i];
    float r = values[i + 1];
    float g = values[i + 2];
    float b = values[i + 3];

    // Normalize data value to [0, 1] range
    float normalizedValue = (dataValue - minVal) / range;

    // Store as ColorPoint: (position, r, g, b)
    colors.emplace_back(normalizedValue, r, g, b);

    // For Paraview files, assume full opacity (alpha = 1.0)
    opacityPoints.emplace_back(dx * i, normalizedValue);
  }

  // Check if colormap with this name already exists and replace it
  int existingIndex = -1;
  for (size_t i = 0; i < m_tfnsNames.size(); ++i) {
    if (m_tfnsNames[i] == name) {
      existingIndex = static_cast<int>(i);
      break;
    }
  }

  int newMapIndex;
  if (existingIndex >= 0) {
    // Replace existing colormap
    m_tfnsColorPoints[existingIndex] = colors;
    newMapIndex = existingIndex;
    tsd::core::logStatus(
        ("[tfn_editor] Replaced existing colormap '" + name + "'").c_str());
  } else {
    // Add new colormap
    m_tfnsColorPoints.push_back(colors);
    m_tfnsNames.push_back(name);
    newMapIndex = m_tfnsColorPoints.size() - 1;
  }

  // Update opacity control points (set to full opacity for Paraview colormaps)
  if (!opacityPoints.empty()) {
    m_tfnOpacityPoints = opacityPoints;

    // Set the loaded colormap as active
    m_currentMap = newMapIndex;
    m_nextMap = newMapIndex;
    m_tfnColorPoints = &(m_tfnsColorPoints[newMapIndex]);

    if (m_volume) {
      updateVolume();
      updateTfnPaletteTexture();
    }
  }

  tsd::core::logStatus(
      ("[tfn_editor] Successfully loaded Paraview colormap '" + name + "' from "
          + filepath + " (" + std::to_string(colors.size()) + " colors)")
          .c_str());
}

void TransferFunctionEditor::updateVolume()
{
  if (!m_colorMapArray) {
    tsd::core::logError(
        "[tfn_editor] No color map array, cannot update volume!");
    return;
  }
  auto co = getSampledColorsAndOpacities(m_colorMapArray->size());
  auto *colorMap = m_colorMapArray->mapAs<tsd::math::float4>();
  std::copy(co.begin(), co.end(), colorMap);
  m_colorMapArray->unmap();

  m_volume->setMetadataArray("opacityControlPoints",
      ANARI_FLOAT32_VEC2,
      m_tfnOpacityPoints.data(),
      m_tfnOpacityPoints.size());
}

void TransferFunctionEditor::updateTfnPaletteTexture()
{
  if (!m_colorMapArray) {
    tsd::core::logError(
        "[tfn_editor] No color map array, cannot update SDL image!");
    return;
  }
  auto width = m_colorMapArray->size();
  if (width != m_tfnPaletteWidth)
    resizeTfnPaletteTexture(width);

  std::vector<tsd::math::float4> palette = m_currentMap == 0
      ? *m_tfnColorPoints
      : getSampledColorsAndOpacities(width);
  std::vector<tsd::math::float3> rgb(width, tsd::math::float3(1.f, 0.f, 0.f));
  std::transform(palette.begin(), palette.end(), rgb.begin(), [&](auto &c) {
    return tsd::math::float3(c.x, c.y, c.z);
  });

  SDL_UpdateTexture(m_tfnPaletteTexture,
      nullptr,
      rgb.data(),
      width * sizeof(tsd::math::float3));
}

void TransferFunctionEditor::resizeTfnPaletteTexture(size_t width)
{
  if (m_tfnPaletteTexture)
    SDL_DestroyTexture(m_tfnPaletteTexture);
  m_tfnPaletteTexture = SDL_CreateTexture(m_app->sdlRenderer(),
      SDL_PIXELFORMAT_RGB96_FLOAT,
      SDL_TEXTUREACCESS_STATIC,
      int(width),
      1);
  m_tfnPaletteWidth = width;
}

void TransferFunctionEditor::saveColormapTo1dt(const std::string &filepath)
{
  if (!m_volume) {
    tsd::core::logError(
        "[tfn_editor] No volume selected, cannot save colormap");
    return;
  }

  // Get the current colormap data
  auto colorsAndOpacities = getSampledColorsAndOpacities(256);

  if (colorsAndOpacities.empty()) {
    tsd::core::logError("[tfn_editor] No colormap data to save");
    return;
  }

  std::ofstream file(filepath);
  if (!file.is_open()) {
    tsd::core::logError(
        ("[tfn_editor] Failed to open file for writing: " + filepath).c_str());
    return;
  }

  // Write colors with interpolated opacity at control point positions
  // This preserves the sparse nature of opacity control points
  for (const auto &opacityPoint : m_tfnOpacityPoints) {
    float position = opacityPoint.x;
    float opacity = opacityPoint.y;

    // Get color at this position
    tsd::math::float3 color(0.f);
    if (m_currentMap == 0) {
      // Direct color mapping
      if (!m_tfnColorPoints->empty()) {
        // Find closest color point or interpolate
        if (position <= m_tfnColorPoints->front().x) {
          auto &cp = m_tfnColorPoints->front();
          color = tsd::math::float3(cp.y, cp.z, cp.w);
        } else if (position >= m_tfnColorPoints->back().x) {
          auto &cp = m_tfnColorPoints->back();
          color = tsd::math::float3(cp.y, cp.z, cp.w);
        } else {
          color =
              tsd::core::detail::interpolateColor(*m_tfnColorPoints, position);
        }
      }
    } else {
      // Interpolated color mapping
      color = tsd::core::detail::interpolateColor(*m_tfnColorPoints, position);
    }

    file << color.x << " " << color.y << " " << color.z << " " << opacity
         << std::endl;
  }

  file.close();

  tsd::core::logStatus(
      ("[tfn_editor] Successfully saved colormap to " + filepath + " ("
          + std::to_string(colorsAndOpacities.size()) + " colors)")
          .c_str());
}

void TransferFunctionEditor::saveColormapToParaview(const std::string &filepath)
{
  if (!m_volume) {
    tsd::core::logError(
        "[tfn_editor] No volume selected, cannot save colormap");
    return;
  }

  if (m_tfnOpacityPoints.empty()) {
    tsd::core::logError("[tfn_editor] No opacity control points to save");
    return;
  }

  std::ofstream file(filepath);
  if (!file.is_open()) {
    tsd::core::logError(
        ("[tfn_editor] Failed to open file for writing: " + filepath).c_str());
    return;
  }

  // Extract basename from filepath for the colormap name
  size_t lastSlash = filepath.find_last_of("/\\");
  std::string basename = (lastSlash != std::string::npos)
      ? filepath.substr(lastSlash + 1)
      : filepath;

  // Remove .json extension if present
  if (basename.size() > 5 && basename.substr(basename.size() - 5) == ".json") {
    basename = basename.substr(0, basename.size() - 5);
  }

  // Write Paraview JSON format (array containing one colormap object)
  file << "[\n";
  file << "\t{\n";
  file << "\t\t\"ColorSpace\" : \"RGB\",\n";
  file << "\t\t\"Creator\" : \"VisRTX\",\n";
  file << "\t\t\"DefaultMap\" : false,\n";
  file << "\t\t\"Name\" : \"" << basename << "\",\n";
  file << "\t\t\"NanColor\" : \n";
  file << "\t\t[\n";
  file << "\t\t\t0.0,\n";
  file << "\t\t\t0.0,\n";
  file << "\t\t\t0.0\n";
  file << "\t\t],\n";
  file << "\t\t\"RGBPoints\" : \n";
  file << "\t\t[\n";

  // Save colors and alpha values at opacity control point positions
  for (size_t i = 0; i < m_tfnOpacityPoints.size(); ++i) {
    const auto &opacityPoint = m_tfnOpacityPoints[i];
    float position = opacityPoint.x;
    float opacity = opacityPoint.y;

    // Get color at this opacity control point position
    tsd::math::float3 color(0.f);
    if (m_currentMap == 0) {
      // Direct color mapping
      if (!m_tfnColorPoints->empty()) {
        // Find closest color point or interpolate
        if (position <= m_tfnColorPoints->front().x) {
          auto &cp = m_tfnColorPoints->front();
          color = tsd::math::float3(cp.y, cp.z, cp.w);
        } else if (position >= m_tfnColorPoints->back().x) {
          auto &cp = m_tfnColorPoints->back();
          color = tsd::math::float3(cp.y, cp.z, cp.w);
        } else {
          color =
              tsd::core::detail::interpolateColor(*m_tfnColorPoints, position);
        }
      }
    } else {
      // Interpolated color mapping
      color = tsd::core::detail::interpolateColor(*m_tfnColorPoints, position);
    }

    file << "\t\t\t" << opacity << ",\n";
    file << "\t\t\t" << color.x << ",\n";
    file << "\t\t\t" << color.y << ",\n";
    file << "\t\t\t" << color.z << ",\n";

    if (i < m_tfnOpacityPoints.size() - 1) {
      file << ",";
    }
    file << "\n";
  }

  file << "\t\t]\n";
  file << "\t}\n";
  file << "]\n";

  file.close();

  tsd::core::logStatus(
      ("[tfn_editor] Successfully saved Paraview colormap to " + filepath + " ("
          + std::to_string(m_tfnOpacityPoints.size()) + " control points)")
          .c_str());
}

void TransferFunctionEditor::getTransferFunctionFilenameFromDialog(
    std::string &filenameOut, bool save)
{
  auto fileDialogCb =
      [](void *userdata, const char *const *filelist, int filter) {
        std::string &out = *(std::string *)userdata;
        if (!filelist) {
          tsd::core::logError("SDL DIALOG ERROR: %s\n", SDL_GetError());
          return;
        }

        if (*filelist)
          out = *filelist;
      };

  // Define file filters for transfer function formats
  SDL_DialogFileFilter filters[] = {
      {"JSON files", "json"}, {"1dt files", "1dt"}};

  if (save) {
    SDL_ShowSaveFileDialog(
        fileDialogCb, &filenameOut, m_app->sdlWindow(), filters, 2, nullptr);
  } else {
    SDL_ShowOpenFileDialog(fileDialogCb,
        &filenameOut,
        m_app->sdlWindow(),
        filters,
        2,
        nullptr,
        false);
  }
}

} // namespace tsd::ui::imgui
