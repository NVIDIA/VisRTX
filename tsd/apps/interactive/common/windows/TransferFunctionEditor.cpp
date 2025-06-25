// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferFunctionEditor.h"
#include "../AppCore.h"
#include "../tsd_ui.h"
// std
#include <algorithm>
#include <fstream>
#include <sstream>
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
    : Window(core, name)
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

  // Add Load File button
  ImGui::SameLine();
  if (ImGui::Button("Load")) {
    m_currentColormapFilename.clear();
    m_core->getFilenameFromDialog(m_currentColormapFilename);
  }
    
  if (!m_currentColormapFilename.empty()) { 
    // Extract filename without path for display name
    size_t lastSlash = m_currentColormapFilename.find_last_of("/\\");
    std::string displayName = (lastSlash != std::string::npos) 
        ? m_currentColormapFilename.substr(lastSlash + 1) 
        : m_currentColormapFilename;
    
    // Detect file type and load accordingly
    bool loaded = false;
    if (displayName.size() > 4 && displayName.substr(displayName.size() - 4) == ".1dt") {
      // Remove .1dt extension for display name
      displayName = displayName.substr(0, displayName.size() - 4);
      loadColormapFrom1dt(m_currentColormapFilename, displayName);
      loaded = true;
    } else if (displayName.size() > 5 && displayName.substr(displayName.size() - 5) == ".json") {
      // Remove .json extension for display name
      displayName = displayName.substr(0, displayName.size() - 5);
      loadColormapFromParaview(m_currentColormapFilename, displayName);
      loaded = true;
    } 
    
    // Set the newly loaded colormap as active and update volume immediately
    if (loaded && !m_tfnsNames.empty()) {
      int newMapIndex = m_tfnsNames.size() - 1;
      setMap(newMapIndex);
      
      // Force immediate update of the volume with the new colormap
      if (m_volume) {
        m_currentMap = newMapIndex;
        m_tfnColorPoints = &(m_tfnsColorPoints[m_currentMap]);
        updateVolume();
        updateTfnPaletteTexture();
      }
    }
    m_currentColormapFilename.clear();
  }
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
    updateVolume();
  }
}

void TransferFunctionEditor::buildUI_opacityScale()
{
  tsd::ui::buildUI_parameter(
      *m_volume, *m_volume->parameter("opacity"), m_core->tsd.ctx);
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
    const bool interpolateColor = m_currentMap != 0;
    tsd::float3 color(0.f);
    if (interpolateColor)
      color = tsd::detail::interpolateColor(*m_tfnColorPoints, i * dx);
    else {
      auto co = (*m_tfnColorPoints)[i];
      color = tsd::float3(co.x, co.y, co.z);
    }
    auto opacity = tsd::detail::interpolateOpacity(m_tfnOpacityPoints, i * dx);
    sampledColorsAndOpacities.push_back(tsd::float4(color, opacity));
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
    setMap(0);

    m_volume = selectedVolume;
    m_colorMapArray = m_volume->parameterValueAsObject<tsd::Array>("color");

    auto &cm = m_tfnsColorPoints[0];
    cm.resize(m_colorMapArray->size());
    auto *colorsIn = m_colorMapArray->dataAs<tsd::float4>();
    std::copy(colorsIn, colorsIn + m_colorMapArray->size(), cm.begin());

    // Get opacity control points from volume //

    anari::DataType type = ANARI_UNKNOWN;
    const tsd::float2 *opacityPoints = nullptr;
    size_t size = 0;
    m_volume->getMetadataArray(
        "opacityControlPoints", &type, (const void **)&opacityPoints, &size);
    if (type == ANARI_FLOAT32_VEC2 && size > 0) {
      tsd::logStatus("[tfn_editor] Receiving opacity control points");
      m_tfnOpacityPoints.resize(size);
      std::copy(
          opacityPoints, opacityPoints + size, m_tfnOpacityPoints.begin());
    } else {
      tsd::logWarning(
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
  std::vector<tsd::ColorPoint> colors;

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
    tsd::logError(
        ("[tfn_editor] Failed to open 1dt file: " + filepath).c_str());
    return;
  }

  std::string line;
  std::vector<tsd::ColorPoint> colors;

  // Read the first line to get the number of entries
  if (!std::getline(file, line)) {
    tsd::logError(
        "[tfn_editor] Failed to read number of entries from 1dt file");
    file.close();
    return;
  }

  int numEntries = 0;
  try {
    numEntries = std::stoi(line);
  } catch (const std::exception &e) {
    tsd::logError(("[tfn_editor] Invalid number of entries in 1dt file: "
        + std::string(e.what()))
            .c_str());
    file.close();
    return;
  }

  if (numEntries <= 0) {
    tsd::logError(("[tfn_editor] Invalid number of entries in 1dt file: "
        + std::to_string(numEntries))
            .c_str());
    file.close();
    return;
  }

  colors.reserve(numEntries);

  // Read each color entry
  int entryIndex = 0;
  while (std::getline(file, line) && entryIndex < numEntries) {
    std::istringstream iss(line);
    float r, g, b, a;

    if (!(iss >> r >> g >> b >> a)) {
      tsd::logError(("[tfn_editor] Failed to parse color entry at line "
          + std::to_string(entryIndex + 2))
              .c_str());
      continue;
    }

    // Calculate normalized position (0.0 to 1.0) based on entry index
    float position =
        static_cast<float>(entryIndex) / static_cast<float>(numEntries - 1);

    // Store as ColorPoint: (position, r, g, b)
    colors.emplace_back(position, r, g, b);
    entryIndex++;
  }

  file.close();

  if (colors.empty()) {
    tsd::logError("[tfn_editor] No valid color entries found in 1dt file");
    return;
  }

  // Add the loaded colormap to the available options
  m_tfnsColorPoints.push_back(colors);
  m_tfnsNames.push_back(name);

  tsd::logStatus(
      ("[tfn_editor] Successfully loaded colormap '" + name + "' from "
          + filepath + " (" + std::to_string(colors.size()) + " colors)")
          .c_str());
}

void TransferFunctionEditor::loadColormapFromParaview(
    const std::string &filepath, const std::string &name)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    tsd::logError(
        ("[tfn_editor] Failed to open Paraview JSON file: " + filepath).c_str());
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
    tsd::logError("[tfn_editor] No RGBPoints found in Paraview JSON file");
    return;
  }

  // Find the opening bracket of the RGBPoints array
  size_t arrayStart = jsonContent.find("[", rgbPointsPos);
  if (arrayStart == std::string::npos) {
    tsd::logError("[tfn_editor] Invalid RGBPoints array format in Paraview JSON file");
    return;
  }

  // Find the closing bracket of the RGBPoints array
  int bracketCount = 0;
  size_t arrayEnd = arrayStart;
  for (size_t i = arrayStart; i < jsonContent.length(); ++i) {
    if (jsonContent[i] == '[') bracketCount++;
    else if (jsonContent[i] == ']') {
      bracketCount--;
      if (bracketCount == 0) {
        arrayEnd = i;
        break;
      }
    }
  }
  
  if (arrayEnd == arrayStart) {
    tsd::logError("[tfn_editor] Could not find end of RGBPoints array in Paraview JSON file");
    return;
  }

  // Extract the array content
  std::string arrayContent = jsonContent.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
  
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

  // RGBPoints should have groups of 4 values: [value, r, g, b, value, r, g, b, ...]
  if (values.size() % 4 != 0) {
    tsd::logError("[tfn_editor] Invalid RGBPoints format - values should be in groups of 4");
    return;
  }

  if (values.empty()) {
    tsd::logError("[tfn_editor] No valid color entries found in Paraview JSON file");
    return;
  }

  std::vector<tsd::ColorPoint> colors;
  colors.reserve(values.size() / 4);

  // Find min/max data values for normalization
  float minVal = values[0];
  float maxVal = values[0];
  for (size_t i = 0; i < values.size(); i += 4) {
    minVal = std::min(minVal, values[i]);
    maxVal = std::max(maxVal, values[i]);
  }
  
  float range = maxVal - minVal;
  if (range == 0.0f) range = 1.0f; // Avoid division by zero

  // Convert to ColorPoint format
  for (size_t i = 0; i < values.size(); i += 4) {
    float dataValue = values[i];
    float r = values[i + 1];
    float g = values[i + 2];  
    float b = values[i + 3];
    
    // Normalize data value to [0, 1] range
    float normalizedValue = (dataValue - minVal) / range;
    
    // Store as ColorPoint: (position, r, g, b)
    colors.emplace_back(normalizedValue, r, g, b);
  }

  // Add the loaded colormap to the available options
  m_tfnsColorPoints.push_back(colors);
  m_tfnsNames.push_back(name);

  tsd::logStatus(
      ("[tfn_editor] Successfully loaded Paraview colormap '" + name + "' from "
          + filepath + " (" + std::to_string(colors.size()) + " colors)")
          .c_str());
}

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

  m_volume->setMetadataArray("opacityControlPoints",
      ANARI_FLOAT32_VEC2,
      m_tfnOpacityPoints.data(),
      m_tfnOpacityPoints.size());
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

  std::vector<tsd::float4> palette = m_currentMap == 0
      ? *m_tfnColorPoints
      : getSampledColorsAndOpacities(width);
  std::vector<tsd::float3> rgb(width, tsd::float3(1.f, 0.f, 0.f));
  std::transform(palette.begin(), palette.end(), rgb.begin(), [&](auto &c) {
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
