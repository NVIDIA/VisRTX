// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "InstancingControls.h"
// tsd_viewer
#include "tsd_ui.h"
// std
#include <random>

namespace tsd_viewer {

using namespace tsd::literals;

// InstancingControls definitions /////////////////////////////////////////////

InstancingControls::InstancingControls(AppContext *state, const char *name)
    : anari_viewer::windows::Window(name, true), m_context(state)
{
  createScene();
}

void InstancingControls::buildUI()
{
  if (ImGui::Button("clear scene"))
    m_context->tsd.ctx.removeAllObjects();

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
    tsd::ui::buildUI_object(*m_light, m_context->tsd.ctx, true);
  }
}

void InstancingControls::createScene()
{
  auto &ctx = m_context->tsd.ctx;

  // Clear out previous scene //

  ctx.removeAllObjects();

  // Default (global) material //

  auto mat = ctx.createObject<tsd::Material>(tsd::tokens::material::matte);
  mat->setName("default_material");
  mat->setParameter(tsd::Token("color"), "color");

  // Generate geometry //

  if (m_addSpheres)
    generateSpheres();
  if (m_addInstances)
    generateInstances();

  // Add light //

  auto light = ctx.createObject<tsd::Light>(tsd::tokens::light::directional);
  light->setName("mainLight");
  light->setParameter("direction", tsd::float2(0.f, 240.f));
  m_light = light.data();

  ctx.tree.insert_first_child(
      ctx.tree.root(), tsd::utility::Any(ANARI_LIGHT, light.index()));

  // Finally update instancing in RenderIndexes //

  ctx.signalInstanceTreeChange();
}

void InstancingControls::generateSpheres()
{
  auto &ctx = m_context->tsd.ctx;

  // Generate geometry //

  auto spheres = ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);

  spheres->setName("random_spheres_geometry");
  spheres->setParameter("radius"_t, m_particleRadius);

  std::mt19937 rng;
  rng.seed(1);
#if 0
  std::normal_distribution<float> vert_dist(0.f, m_spacing);
#else
  std::uniform_real_distribution<float> vert_dist(
      -m_spacing / 2.f, m_spacing / 2.f);
#endif

  const uint32_t numSpheres = m_numInstances;
  auto positionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numSpheres);

  auto *positions = positionArray->mapAs<tsd::float3>();
  for (uint32_t i = 0; i < numSpheres; i++)
    positions[i] = tsd::float3(vert_dist(rng), vert_dist(rng), vert_dist(rng));
  positionArray->unmap();

  spheres->setParameterObject("vertex.position"_t, *positionArray);
  spheres->setParameter("color"_t, tsd::float4(1.f, 0.f, 0.f, 1.f));

  // Populate material with sampler for colormapping //

  auto surface =
      ctx.createSurface("random_spheres", spheres, ctx.defaultMaterial());

  ctx.tree.insert_last_child(
      ctx.tree.root(), tsd::utility::Any(ANARI_SURFACE, surface.index()));
}

void InstancingControls::generateInstances()
{
  auto &ctx = m_context->tsd.ctx;

  // Setup transforms //

  size_t numXfms = size_t(m_numInstances);
  auto xfmArray = ctx.createArray(ANARI_FLOAT32_MAT4, numXfms);

  auto xfmArrayNode = ctx.tree.insert_last_child(
      ctx.tree.root(), tsd::utility::Any(ANARI_ARRAY1D, xfmArray.index()));

  std::mt19937 rng;
  rng.seed(0);

#if 0
  std::normal_distribution<float> pos_dist(0.f, m_spacing);
#else
  std::uniform_real_distribution<float> pos_dist(
      -m_spacing / 2.f, m_spacing / 2.f);
#endif
  auto *xfms = xfmArray->mapAs<tsd::mat4>();
  std::for_each(xfms, xfms + numXfms, [&](auto &xfm) {
    xfm = anari::math::translation_matrix(
        tsd::float3(pos_dist(rng), pos_dist(rng), pos_dist(rng)));
  });
  xfmArray->unmap();

  // Setup randomized per-instance colors //

  auto attrArray = ctx.createArray(ANARI_FLOAT32_VEC3, numXfms);

  std::uniform_real_distribution<float> col_dist(0.1f, 0.9f);
  auto *attrs = attrArray->mapAs<tsd::float3>();
  std::for_each(attrs, attrs + numXfms, [&](auto &attr) {
    attr = tsd::float3(col_dist(rng), col_dist(rng), col_dist(rng));
  });
  attrArray->unmap();

  (*xfmArrayNode)->customParameters["color"] =
      tsd::utility::Any(ANARI_ARRAY1D, attrArray.index());

  // Generate mesh //

  tsd::generate_monkey(ctx, xfmArrayNode);
}

} // namespace tsd_viewer