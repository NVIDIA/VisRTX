// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "WeightedPointsControls.h"
#include "WeightedPointsOctree.h"
#include <tsd/ui/imgui/tsd_ui_imgui.h>
#include <tsd/core/Logging.hpp>
#include <tsd/core/TSDMath.hpp>
#include <tsd/scene/objects/Array.hpp>
#include <tsd/scene/objects/SpatialField.hpp>
#include <tsd/scene/objects/Volume.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

namespace tsd::demo {

WeightedPointsControls::WeightedPointsControls(
    tsd::ui::imgui::Application *app,
    const char *name,
    const std::string &pdbPath)
    : Window(app, name)
    , m_pdbPath(pdbPath)
    , m_usePDB(!pdbPath.empty())
{}

void WeightedPointsControls::buildUI()
{
  if (!m_sceneCreated) {
    m_sceneCreated = true;
    createScene();
  }

  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Point Cloud Source")) {
    bool changed = false;

    if (ImGui::Checkbox("Use PDB File", &m_usePDB))
      changed = true;

    if (m_usePDB) {
      char buf[512] = {};
      std::strncpy(buf, m_pdbPath.c_str(), sizeof(buf) - 1);
      if (ImGui::InputText("PDB Path",
              buf,
              sizeof(buf),
              ImGuiInputTextFlags_EnterReturnsTrue)) {
        m_pdbPath = buf;
        changed = true;
      }
    } else {
      changed |= ImGui::SliderInt("Points", &m_numPoints, 100, 50000);
    }

    if (changed || ImGui::Button("Regenerate"))
      createScene();
  }

  ImGui::Spacing();

  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Field Parameters")) {
    bool sigmaChanged = ImGui::DragFloat(
        "Sigma (0 = auto)", &m_sigmaOverride, 0.005f, 0.f, 10.f, "%.4f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip(
          "Gaussian kernel width. 0 = auto-compute from point spacing.");
    }

    bool cutoffChanged = ImGui::DragFloat(
        "Cutoff (0 = auto)", &m_cutoffOverride, 0.02f, 0.f, 100.f, "%.3f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("LOD cutoff distance. 0 = auto (5x sigma).");
    }

    if (sigmaChanged || cutoffChanged) {
      if (m_field) {
        auto &scene = appContext()->tsd.scene;
        if (m_sigmaOverride > 0.f)
          m_field->setParameter("sigma", m_sigmaOverride);
        if (m_cutoffOverride > 0.f)
          m_field->setParameter("cutoff", m_cutoffOverride);

        bool needFullRebuild = (sigmaChanged && m_sigmaOverride <= 0.f)
            || (cutoffChanged && m_cutoffOverride <= 0.f);
        if (needFullRebuild)
          rebuildField();
        else
          scene.signalLayerStructureChanged(scene.defaultLayer());
      }
    }
  }

  ImGui::Spacing();

  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Lighting")) {
    if (m_light)
      tsd::ui::buildUI_object(*m_light, appContext()->tsd.scene, true);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::TextWrapped(
      "Weighted-points field: octree of Gaussian blobs with "
      "distance-based LOD. Each leaf stores a weighted centroid; "
      "internal nodes aggregate subtree weights for fast far-field "
      "approximation.");
}

void WeightedPointsControls::createScene()
{
  auto &scene = appContext()->tsd.scene;
  auto *layer = scene.defaultLayer();
  layer->clear();

  generatePoints();

  const tsd::core::Token wpToken("weightedPoints");
  m_field = scene.createObject<tsd::scene::SpatialField>(wpToken);
  m_field->setName("weighted_points_field");
  m_field->setParameter("domainMin", tsd::math::float3(0.f, 0.f, 0.f));
  m_field->setParameter("domainMax", tsd::math::float3(1.f, 1.f, 1.f));
  m_field->setParameter("sigma", 1.f);
  m_field->setParameter("cutoff", 1.f);

  m_volume = scene.createObject<tsd::scene::Volume>(
      tsd::scene::tokens::volume::transferFunction1D);
  m_volume->setName("weighted_points_volume");
  m_volume->setParameterObject("value", *m_field);

  tsd::math::float2 valueRange(0.f, 1.f);
  m_volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
  m_volume->setParameter("unitDistance", 1.f);

  constexpr size_t TF_SIZE = 256;
  auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, TF_SIZE);
  auto *tfData = colorArray->mapAs<tsd::math::float4>();
  for (size_t i = 0; i < TF_SIZE; i++) {
    float t = i / (float)(TF_SIZE - 1);

    // Turbo-inspired colormap: deep blue → cyan → green → yellow → red → maroon
    float r, g, b;
    if (t < 0.15f) {
      float s = t / 0.15f;
      r = 0.19f + 0.11f * s;
      g = 0.07f + 0.31f * s;
      b = 0.56f + 0.44f * s;
    } else if (t < 0.35f) {
      float s = (t - 0.15f) / 0.2f;
      r = 0.30f - 0.22f * s;
      g = 0.38f + 0.52f * s;
      b = 1.00f - 0.30f * s;
    } else if (t < 0.55f) {
      float s = (t - 0.35f) / 0.2f;
      r = 0.08f + 0.62f * s;
      g = 0.90f + 0.10f * s;
      b = 0.70f - 0.60f * s;
    } else if (t < 0.75f) {
      float s = (t - 0.55f) / 0.2f;
      r = 0.70f + 0.30f * s;
      g = 1.00f - 0.45f * s;
      b = 0.10f - 0.05f * s;
    } else {
      float s = (t - 0.75f) / 0.25f;
      r = 1.00f - 0.40f * s;
      g = 0.55f - 0.50f * s;
      b = 0.05f + 0.05f * s;
    }

    float a = t;
    tfData[i] = tsd::math::float4(r, g, b, a);
  }
  colorArray->unmap();
  m_volume->setParameterObject("color", *colorArray);

  layer->root()->insert_first_child({layer, m_volume});

  rebuildField();

  auto light = scene.createObject<tsd::scene::Light>(
      tsd::scene::tokens::light::directional);
  light->setName("mainLight");
  light->setParameter("direction", tsd::math::float2(0.f, 240.f));
  light->setParameter("irradiance", 20.0f);
  m_light = light.data();
  layer->root()->insert_last_child({layer, light});

  scene.signalLayerStructureChanged(layer);
}

void WeightedPointsControls::generatePoints()
{
  if (m_usePDB)
    m_rawPoints = loadPDB(m_pdbPath);
  else
    m_rawPoints = generateRandomUniform();
}

void WeightedPointsControls::rebuildField()
{
  if (!m_field || !m_volume || m_rawPoints.empty())
    return;

  auto &scene = appContext()->tsd.scene;

  const size_t n = m_rawPoints.size() / 4;

  std::vector<WeightedPoint> points;
  points.reserve(n);
  for (size_t i = 0; i < n; i++) {
    const float *p = m_rawPoints.data() + i * 4;
    points.push_back({p[0], p[1], p[2], p[3]});
  }

  WeightedPointsOctree octree;
  octree.build(points, /*maxPointsPerLeaf=*/8, /*maxDepth=*/12);

  const auto &flatValues = octree.flatValues();
  const auto &flatIndices = octree.flatIndices();
  int numNodes = octree.numNodes();

  if (numNodes == 0) {
    tsd::core::logStatus("[customField demo] octree produced no nodes");
    return;
  }

  const float *bmin = octree.boundsMin();
  const float *bmax = octree.boundsMax();
  float domainMin[3], domainMax[3];
  for (int c = 0; c < 3; c++) {
    float extent = bmax[c] - bmin[c];
    float pad = std::max(0.1f, extent * 0.05f);
    domainMin[c] = bmin[c] - pad;
    domainMax[c] = bmax[c] + pad;
  }

  float sigma = m_sigmaOverride;
  if (sigma <= 0.f) {
    // Estimate typical inter-atom spacing via sampled nearest-neighbor distances
    size_t nSamp = std::min(n, (size_t)200);
    size_t sStep = std::max((size_t)1, n / nSamp);
    std::vector<float> nnDists;
    nnDists.reserve(nSamp);
    for (size_t i = 0; i < n; i += sStep) {
      const float *p = m_rawPoints.data() + i * 4;
      float best = std::numeric_limits<float>::max();
      for (size_t j = 0; j < n; j++) {
        if (j == i)
          continue;
        const float *q = m_rawPoints.data() + j * 4;
        float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best)
          best = d2;
      }
      if (best < std::numeric_limits<float>::max())
        nnDists.push_back(std::sqrt(best));
    }
    std::sort(nnDists.begin(), nnDists.end());
    float medianNN = nnDists.empty()
        ? 1.f
        : nnDists[nnDists.size() / 2];
    sigma = medianNN * 3.f;
  }

  float cutoff = m_cutoffOverride;
  if (cutoff <= 0.f) {
    float dx = bmax[0] - bmin[0];
    float dy = bmax[1] - bmin[1];
    float dz = bmax[2] - bmin[2];
    float diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
    cutoff = diagonal * 0.4f;
  }

  m_field->setParameter(
      "domainMin", tsd::math::float3(domainMin[0], domainMin[1], domainMin[2]));
  m_field->setParameter(
      "domainMax", tsd::math::float3(domainMax[0], domainMax[1], domainMax[2]));
  m_field->setParameter("sigma", sigma);
  m_field->setParameter("cutoff", cutoff);

  auto valuesArray = scene.createArray(ANARI_FLOAT32, flatValues.size());
  valuesArray->setData(flatValues.data());
  m_field->setParameterObject("values", *valuesArray);

  auto indicesArray = scene.createArray(ANARI_INT32, flatIndices.size());
  indicesArray->setData(flatIndices.data());
  m_field->setParameterObject("indices", *indicesArray);

  float inv2s2 = 1.f / (2.f * sigma * sigma);
  float maxFieldVal = 0.f;
  size_t nSamples = std::min(n, (size_t)50);
  size_t step = std::max((size_t)1, n / nSamples);
  for (size_t i = 0; i < n; i += step) {
    const float *p = m_rawPoints.data() + i * 4;
    float val = 0.f;
    for (size_t j = 0; j < n; j++) {
      const float *q = m_rawPoints.data() + j * 4;
      float ddx = p[0] - q[0], ddy = p[1] - q[1], ddz = p[2] - q[2];
      float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
      val += q[3] * std::exp(-d2 * inv2s2);
    }
    if (val > maxFieldVal)
      maxFieldVal = val;
  }

  float rangeMax = (maxFieldVal > 0.f) ? maxFieldVal : 100.f;
  tsd::math::float2 valueRange(0.f, rangeMax);
  m_volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  scene.signalLayerStructureChanged(scene.defaultLayer());

  tsd::core::logStatus(
      "[customField demo] %zu points -> %d nodes, "
      "sigma=%.4f, cutoff=%.3f, range=[0, %.2f]",
      n, numNodes, sigma, cutoff, rangeMax);
}

// ---------------------------------------------------------------------------

std::vector<float> WeightedPointsControls::generateRandomUniform()
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> pos_dist(-1.f, 1.f);
  std::uniform_real_distribution<float> weight_dist(0.1f, 1.5f);

  std::vector<float> points;
  points.reserve(m_numPoints * 4);

  for (int i = 0; i < m_numPoints; i++) {
    points.push_back(pos_dist(rng));
    points.push_back(pos_dist(rng));
    points.push_back(pos_dist(rng));
    points.push_back(weight_dist(rng));
  }
  return points;
}

std::vector<float> WeightedPointsControls::loadPDB(const std::string &path)
{
  std::vector<float> points;

  FILE *fp = std::fopen(path.c_str(), "r");
  if (!fp) {
    tsd::core::logWarning("[PDB] failed to open '%s'", path.c_str());
    return points;
  }

  char line[256];
  while (std::fgets(line, sizeof(line), fp)) {
    if (std::strncmp(line, "ATOM  ", 6) != 0
        && std::strncmp(line, "HETATM", 6) != 0)
      continue;

    size_t len = std::strlen(line);
    if (len < 54)
      continue;

    char xbuf[9] = {}, ybuf[9] = {}, zbuf[9] = {};
    std::memcpy(xbuf, line + 30, 8);
    std::memcpy(ybuf, line + 38, 8);
    std::memcpy(zbuf, line + 46, 8);

    float x = std::strtof(xbuf, nullptr);
    float y = std::strtof(ybuf, nullptr);
    float z = std::strtof(zbuf, nullptr);

    float w = 1.f;
    if (len >= 60) {
      char obuf[7] = {};
      std::memcpy(obuf, line + 54, 6);
      float occ = std::strtof(obuf, nullptr);
      if (occ > 0.f)
        w = occ;
    }

    points.push_back(x);
    points.push_back(y);
    points.push_back(z);
    points.push_back(w);
  }

  std::fclose(fp);

  tsd::core::logStatus(
      "[PDB] loaded %zu atoms from '%s'", points.size() / 4, path.c_str());

  return points;
}

} // namespace tsd::demo
