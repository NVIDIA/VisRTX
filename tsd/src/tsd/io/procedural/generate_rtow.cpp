// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"

double drand48()
{
  constexpr static uint64_t m = 1ULL << 48;
  constexpr static uint64_t a = 0x5DEECE66DULL;
  constexpr static uint64_t c = 0xBULL;
  thread_local static uint64_t x = 0;

  x = (a * x + c) & (m - 1ULL);

  return double(x) / m;
}

namespace tsd::io {

inline SurfaceRef makeSphere(Context &ctx, float3 pos, float radius, int ID)
{
  auto sphere = ctx.createObject<Geometry>(tokens::geometry::sphere);

  auto positionArray = ctx.createArray(ANARI_FLOAT32_VEC3, 1);
  auto radiusArray = ctx.createArray(ANARI_FLOAT32, 1);

  auto *P = positionArray->mapAs<float3>();
  auto *R = radiusArray->mapAs<float>();

  P[0] = pos;
  R[0] = radius;

  positionArray->unmap();
  radiusArray->unmap();

  auto name = std::string("sphere_") + std::to_string(ID);

  sphere->setParameterObject("vertex.position"_t, *positionArray);
  sphere->setParameterObject("vertex.radius"_t, *radiusArray);
  sphere->setName((name + "_geometry").c_str());

  return ctx.createSurface(name.c_str(), sphere);
}

inline MaterialRef makeDielectric(Context &ctx, float ior, int ID)
{
  auto material = ctx.createObject<Material>(tokens::material::physicallyBased);
  material->setParameter("baseColor"_t, float3(1.0f, 1.0f, 1.0f));
  material->setParameter("ior"_t, ior);
  material->setName((std::string("dielectric") + std::to_string(ID)).c_str());
  return material;
}

inline MaterialRef makeLambertian(Context &ctx, float3 color, int ID)
{
  auto material = ctx.createObject<Material>(tokens::material::matte);
  material->setParameter("color"_t, color);
  material->setName((std::string("matte_") + std::to_string(ID)).c_str());
  return material;
}

inline MaterialRef makeMetal(Context &ctx, float3 refl, int ID)
{
  auto material = ctx.createObject<Material>(tokens::material::physicallyBased);
  material->setParameter("baseColor"_t, refl);
  material->setParameter("metallic"_t, 0.9f);
  material->setParameter("roughness"_t, 0.4f);
  material->setParameter("ior"_t, 0.5f);
  material->setName((std::string("metal") + std::to_string(ID)).c_str());
  return material;
}

void generate_rtow(Context &ctx, LayerNodeRef location)
{
  if (!location)
    location = ctx.defaultLayer()->root();

  auto rtow_root = ctx.insertChildNode(location);

  int i = 0;

  auto hugeSphere = makeSphere(ctx, float3(0, -1000, 0), 1000, i);
  hugeSphere->setMaterial(makeLambertian(ctx, float3(0.5f, 0.5f, 0.5f), i));
  i++;

  ctx.insertChildObjectNode(rtow_root, hugeSphere);

  for (int a = -11; a < 11; ++a) {
    for (int b = -11; b < 11; ++b) {
      float choose_mat = drand48();
      float3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
      if (length(center - float3(4, 0.2, 0)) > 0.9) {
        auto sphere = makeSphere(ctx, center, 0.2, i);
        if (choose_mat < 0.8) // diffuse
        {
          sphere->setMaterial(makeLambertian(ctx,
              float3(float(drand48() * drand48()),
                  float(drand48() * drand48()),
                  float(drand48() * drand48())),
              i));
        } else if (choose_mat < 0.95) // metal
        {
          sphere->setMaterial(makeMetal(ctx,
              float3(0.5f * (1.0f + (float)(drand48())),
                  0.5f * (1.0f + (float)(drand48())),
                  0.5f * (1.0f + (float)(drand48()))),
              i));
        } else {
          sphere->setMaterial(makeDielectric(ctx, 1.5f, i));
        }
        ctx.insertChildObjectNode(rtow_root, sphere);
        i++;
      }
    }
  }

  auto sphere1 = makeSphere(ctx, float3(0, 1, 0), 1.5f, i);
  sphere1->setMaterial(makeDielectric(ctx, 1.5f, i));
  ctx.insertChildObjectNode(rtow_root, sphere1);
  i++;

  auto sphere2 = makeSphere(ctx, float3(-4, 1, 0), 1.0f, i);
  sphere2->setMaterial(makeLambertian(ctx, float3(0.5f, 0.2f, 0.1f), i));
  ctx.insertChildObjectNode(rtow_root, sphere2);
  i++;

  auto sphere3 = makeSphere(ctx, float3(4, 1, 0), 1.0f, i);
  sphere3->setMaterial(makeMetal(ctx, float3(0.7f, 0.6f, 0.5f), i));
  ctx.insertChildObjectNode(rtow_root, sphere3);
  i++;
}

} // namespace tsd
