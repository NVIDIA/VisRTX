// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "InstancingControls.h"
// tsd_ui_imgui
#include <tsd/ui/imgui/tsd_ui_imgui.h>
// tsd_io
#include <tsd/io/procedural.hpp>
// std
#include <random>

namespace tsd::demo {

InstancingControls::InstancingControls(
    tsd::ui::imgui::Application *app, const char *name)
    : Window(app, name)
{
  createScene();
}

void InstancingControls::buildUI()
{
  if (ImGui::Button("clear scene"))
    appCore()->tsd.scene.removeAllObjects();

  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Instancing")) {
    ImGui::Checkbox("particles", &m_addSpheres);
    ImGui::InputFloat("particle radius", &m_particleRadius);
    ImGui::Checkbox("mesh instances", &m_addInstances);
    ImGui::DragInt("# instances", &m_numInstances, 1, 10000000);
    ImGui::DragFloat("spacing", &m_spacing, 1.f, 10000000.f);
    if (ImGui::Button("update scene"))
      createScene();
  }

  ImGui::NewLine();

  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Lighting")) {
    tsd::ui::buildUI_object(*m_light, appCore()->tsd.scene, true);
  }
}

void InstancingControls::createScene()
{
  auto &scene = appCore()->tsd.scene;
  auto *layer = scene.defaultLayer();

  // Clear out previous scene //

  scene.removeAllObjects();

  // Default (global) material //

  auto mat = scene.createObject<tsd::core::Material>(
      tsd::core::tokens::material::matte);
  mat->setName("default_material");
  mat->setParameter("color", "color");

  // Generate geometry //

  if (m_addSpheres)
    generateSpheres();
  if (m_addInstances)
    generateInstances();

  // Add light //

  auto light = scene.createObject<tsd::core::Light>(
      tsd::core::tokens::light::directional);
  light->setName("mainLight");
  light->setParameter("direction", tsd::math::float2(0.f, 240.f));
  m_light = light.data();

  layer->root()->insert_first_child({light});

  // Finally update instancing in RenderIndexes //

  scene.signalLayerChange(layer);
}

void InstancingControls::generateSpheres()
{
  auto &scene = appCore()->tsd.scene;

  // Generate geometry //

  auto spheres = scene.createObject<tsd::core::Geometry>(
      tsd::core::tokens::geometry::sphere);

  spheres->setName("random_spheres_geometry");
  spheres->setParameter("radius", m_particleRadius);

  std::mt19937 rng;
  rng.seed(1);
#if 0
  std::normal_distribution<float> vert_dist(0.f, m_spacing);
#else
  std::uniform_real_distribution<float> vert_dist(
      -m_spacing / 2.f, m_spacing / 2.f);
#endif

  const uint32_t numSpheres = m_numInstances;
  auto positionArray = scene.createArray(ANARI_FLOAT32_VEC3, numSpheres);

  auto *positions = positionArray->mapAs<tsd::math::float3>();
  for (uint32_t i = 0; i < numSpheres; i++)
    positions[i] =
        tsd::math::float3(vert_dist(rng), vert_dist(rng), vert_dist(rng));
  positionArray->unmap();

  spheres->setParameterObject("vertex.position", *positionArray);
  spheres->setParameter("color", tsd::math::float4(1.f, 0.f, 0.f, 1.f));

  // Populate material with sampler for colormapping //

  auto surface =
      scene.createSurface("random_spheres", spheres, scene.defaultMaterial());

  scene.defaultLayer()->root()->insert_last_child({surface});
}

void InstancingControls::generateInstances()
{
  auto &scene = appCore()->tsd.scene;

  // Setup transforms //

  size_t numXfms = size_t(m_numInstances);
  auto xfmArray = scene.createArray(ANARI_FLOAT32_MAT4, numXfms);

  auto xfmArrayNode =
      scene.defaultLayer()->root()->insert_last_child({xfmArray});

  std::mt19937 rng;
  rng.seed(0);

#if 0
  std::normal_distribution<float> pos_dist(0.f, m_spacing);
#else
  std::uniform_real_distribution<float> pos_dist(
      -m_spacing / 2.f, m_spacing / 2.f);
#endif
  auto *xfms = xfmArray->mapAs<tsd::math::mat4>();
  std::for_each(xfms, xfms + numXfms, [&](auto &xfm) {
    xfm = anari::math::translation_matrix(
        tsd::math::float3(pos_dist(rng), pos_dist(rng), pos_dist(rng)));
  });
  xfmArray->unmap();

  // Setup randomized per-instance colors //

  auto attrArray = scene.createArray(ANARI_FLOAT32_VEC3, numXfms);

  std::uniform_real_distribution<float> col_dist(0.1f, 0.9f);
  auto *attrs = attrArray->mapAs<tsd::math::float3>();
  std::for_each(attrs, attrs + numXfms, [&](auto &attr) {
    attr = tsd::math::float3(col_dist(rng), col_dist(rng), col_dist(rng));
  });
  attrArray->unmap();

  (*xfmArrayNode)
      ->setInstanceParameter(
          "color", tsd::core::Any(ANARI_ARRAY1D, attrArray.index()));

  // Generate mesh //

  tsd::io::generate_monkey(scene, xfmArrayNode);
}

} // namespace tsd::demo