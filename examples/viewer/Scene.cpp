/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "Scene.h"
// glm
#include <glm/glm.hpp>
#include "glm/ext/matrix_transform.hpp"
// anari
#include <anari/anari_cpp/ext/glm.h>
// VisRTX
#include "anari/ext/visrtx/visrtx.h"
// tiny_obj_loader
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
// stb_image
#include "stb_image.h"
// match3D
#include <match3D/match3D.h>
// std
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <unordered_map>

// Helper functions  //////////////////////////////////////////////////////////

static void anari_free(const void * /*user_data*/, const void *ptr)
{
  std::free(const_cast<void *>(ptr));
}

static anari::Array2D makeTextureData(anari::Device d, int dim)
{
#if 0
  auto *data = new glm::vec3[dim * dim];

  for (int h = 0; h < dim; h++) {
    for (int w = 0; w < dim; w++) {
      bool even = h & 1;
      if (even)
        data[h * dim + w] = w & 1 ? glm::vec3(.7f) : glm::vec3(.3f);
      else
        data[h * dim + w] = w & 1 ? glm::vec3(.3f) : glm::vec3(.7f);
    }
  }

  return anariNewArray2D(
      d, data, &anari_free, nullptr, ANARI_FLOAT32_VEC3, dim, dim);
#else
  using texel = std::array<uint8_t, 3>;
  auto *data = new texel[dim * dim];

  auto makeTexel = [](uint8_t v) -> texel { return {v, v, v}; };

  for (int h = 0; h < dim; h++) {
    for (int w = 0; w < dim; w++) {
      bool even = h & 1;
      if (even)
        data[h * dim + w] = w & 1 ? makeTexel(200) : makeTexel(50);
      else
        data[h * dim + w] = w & 1 ? makeTexel(50) : makeTexel(200);
    }
  }

  return anariNewArray2D(
      d, data, &anari_free, nullptr, ANARI_UFIXED8_VEC3, dim, dim);
#endif
}

static anari::Surface makePlane(anari::Device d, const box3 &bounds)
{
  std::vector<glm::vec3> vertices;
  float f = glm::length(bounds[1] - bounds[0]);
  vertices.emplace_back(
      bounds[0].x - f, bounds[0].y - (0.1f * f), bounds[1].z + f);
  vertices.emplace_back(
      bounds[1].x + f, bounds[0].y - (0.1f * f), bounds[1].z + f);
  vertices.emplace_back(
      bounds[1].x + f, bounds[0].y - (0.1f * f), bounds[0].z - f);
  vertices.emplace_back(
      bounds[0].x - f, bounds[0].y - (0.1f * f), bounds[0].z - f);

  std::vector<glm::vec2> texcoords = {
      //
      {0.f, 0.f},
      {0.f, 1.f},
      {1.f, 1.f},
      {1.f, 0.f}
      //
  };

  auto geom = anari::newObject<anari::Geometry>(d, "quad");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, vertices.data(), vertices.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.attribute0",
      anari::newArray1D(d, texcoords.data(), texcoords.size()));
  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto tex = anari::newObject<anari::Sampler>(d, "image2D");
  anari::setAndReleaseParameter(d, tex, "image", makeTextureData(d, 8));
  anari::setParameter(d, tex, "inAttribute", "attribute0");
  anari::setParameter(d, tex, "wrapMode1", "clampToEdge");
  anari::setParameter(d, tex, "wrapMode2", "clampToEdge");
  anari::setParameter(d, tex, "filter", "nearest");
  anari::commitParameters(d, tex);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setAndReleaseParameter(d, mat, "color", tex);
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  return surface;
}

static anari::Instance makePlaneInstance(anari::Device d, const box3 &bounds)
{
  auto surface = makePlane(d, bounds);

  auto group = anari::newObject<anari::Group>(d);
  anari::setAndReleaseParameter(
      d, group, "surface", anari::newArray1D(d, &surface));
  anari::commitParameters(d, group);

  anari::release(d, surface);

  auto inst = anari::newObject<anari::Instance>(d);
  anari::setAndReleaseParameter(d, inst, "group", group);
  anari::commitParameters(d, inst);

  return inst;
}

static std::vector<anari::Instance> makeGridOfInstances(
    anari::Device d, anari::Surface surface, anari::Volume volume)
{
  auto group = anari::newObject<anari::Group>(d);
  if (surface) {
    anari::setAndReleaseParameter(
        d, group, "surface", anari::newArray1D(d, &surface));
  }
  if (volume) {
    anari::setAndReleaseParameter(
        d, group, "volume", anari::newArray1D(d, &volume));
  }

  anari::commitParameters(d, group);

  std::vector<anari::Instance> instances;

  for (int x = 1; x < 4; x++) {
    for (int y = 1; y < 4; y++) {
      for (int z = 1; z < 4; z++) {
        auto inst = anari::newObject<anari::Instance>(d);
        auto tl = glm::translate(glm::mat4(1.f), 4.f * glm::vec3(x, y, z));
        auto rot_x = glm::rotate(glm::mat4(1.f), float(x), glm::vec3(1, 0, 0));
        auto rot_y = glm::rotate(glm::mat4(1.f), float(y), glm::vec3(0, 1, 0));
        auto rot_z = glm::rotate(glm::mat4(1.f), float(z), glm::vec3(0, 0, 1));

        { // NOTE: exercise anari::setParameter with C-array type
          glm::mat4x3 _xfm = tl * rot_x * rot_y * rot_z;
          float xfm[12];
          std::memcpy(xfm, &_xfm, sizeof(_xfm));
          anari::setParameter(d, inst, "transform", xfm);
        }

        anari::setParameter(d, inst, "group", group);
        anari::commitParameters(d, inst);
        instances.push_back(inst);
      }
    }
  }

  anari::release(d, group);

  return instances;
}

///////////////////////////////////////////////////////////////////////////////
// Cylinders scene ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static ScenePtr generateCylinders(anari::Device d, CylindersConfig config)
{
  auto world = anari::newObject<anari::World>(d);

  std::mt19937 rng;
  if (config.useRandomSeed)
    rng.seed(std::random_device()());
  else
    rng.seed(0);
  std::uniform_real_distribution<float> dist(0.f, 1.f);

  std::vector<glm::vec3> positions(2 * config.numCylinders);
  std::vector<glm::uvec2> indices(config.numCylinders);

  for (auto &s : positions) {
    s.x = dist(rng);
    s.y = dist(rng);
    s.z = dist(rng);
  }

  for (int i = 0; i < config.numCylinders; i++)
    indices[i] = glm::uvec2(2 * i) + glm::uvec2(0, 1);

  auto geom = anari::newObject<anari::Geometry>(d, "cylinder");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, positions.data(), positions.size()));
  anari::setParameter(d, geom, "radius", config.radius);
  anari::setParameter(d, geom, "caps", config.caps ? "both" : "none");

  std::vector<glm::vec4> colors(2 * config.numCylinders);

  for (auto &s : colors) {
    s.x = dist(rng);
    s.y = dist(rng);
    s.z = dist(rng);
    s.w = 1.f;
  }

  anari::setAndReleaseParameter(d,
      geom,
      "vertex.color",
      anari::newArray1D(d, colors.data(), colors.size()));

  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "transparentMatte");
  anari::setParameter(d, mat, "color", "color");
  anari::setParameter(d, mat, "opacity", config.opacity);
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  anari::setAndReleaseParameter(
      d, world, "surface", anari::newArray1D(d, &surface));

  anari::release(d, surface);

  return std::make_unique<Scene>(d, world);
}

///////////////////////////////////////////////////////////////////////////////
// Cones scene ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static ScenePtr generateCones(anari::Device d, ConesConfig config)
{
  auto world = anari::newObject<anari::World>(d);

  std::mt19937 rng;
  if (config.useRandomSeed)
    rng.seed(std::random_device()());
  else
    rng.seed(0);
  std::uniform_real_distribution<float> dist(0.f, 1.f);

  std::vector<glm::vec3> positions(2 * config.numCones);
  std::vector<glm::uvec2> indices(config.numCones);

  for (auto &s : positions) {
    s.x = dist(rng);
    s.y = dist(rng);
    s.z = dist(rng);
  }

  for (int i = 0; i < config.numCones; i++)
    indices[i] = glm::uvec2(2 * i) + glm::uvec2(0, 1);

  std::vector<glm::vec2> radii(config.numCones);
  std::fill(radii.begin(), radii.end(), glm::vec2(0.125f, 0.f));

  auto geom = anari::newObject<anari::Geometry>(d, "cone");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, positions.data(), positions.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.radius",
      anari::newArray1D(d, (float *)radii.data(), radii.size() * 2));
  anari::setParameter(d, geom, "caps", config.caps ? "caps" : "none");

  std::vector<glm::vec4> colors(2 * config.numCones);

  for (auto &s : colors) {
    s.x = dist(rng);
    s.y = dist(rng);
    s.z = dist(rng);
    s.w = 1.f;
  }

  anari::setAndReleaseParameter(d,
      geom,
      "primitive.color",
      anari::newArray1D(d, colors.data(), colors.size()));

  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "transparentMatte");
  anari::setParameter(d, mat, "color", "color");
  anari::setParameter(d, mat, "opacity", config.opacity);
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  anari::setAndReleaseParameter(
      d, world, "surface", anari::newArray1D(d, &surface));

  anari::release(d, surface);

  return std::make_unique<Scene>(d, world);
}

static ScenePtr generateCurves(anari::Device d, CurvesConfig config)
{
  auto world = anari::newObject<anari::World>(d);

  // This code is adapted from the OSPRay 'streamlines' example:
  //   https://github.com/ospray/ospray/blob/fdda0889f9143a8b20f26389c22d1691f1a6a527/apps/common/ospray_testing/builders/Streamlines.cpp

  std::vector<glm::vec3> positions;
  std::vector<float> radii;
  std::vector<unsigned int> indices;
  std::vector<glm::vec4> colors;

  auto addPoint = [&](const glm::vec4 &p) {
    positions.emplace_back(p.x, p.y, p.z);
    radii.push_back(p.w);
  };

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> radDist(0.5f, 1.5f);
  std::uniform_real_distribution<float> stepDist(0.001f, 0.1f);
  std::uniform_real_distribution<float> sDist(0, 360);
  std::uniform_real_distribution<float> dDist(360, 720);
  std::uniform_real_distribution<float> freqDist(0.5f, 1.5f);

  // create multiple lines
  int numLines = 100;
  for (int l = 0; l < numLines; l++) {
    int dStart = sDist(rng);
    int dEnd = dDist(rng);
    float radius = radDist(rng);
    float h = 0;
    float hStep = stepDist(rng);
    float f = freqDist(rng);

    float r = (720 - dEnd) / 360.f;
    glm::vec4 c(r, 1 - r, 1 - r / 2, 1.f);

    // spiral up with changing radius of curvature
    for (int d = dStart; d < dStart + dEnd; d += 10, h += hStep) {
      glm::vec3 p, q;
      float startRadius, endRadius;

      p.x = radius * std::sin(d * M_PI / 180.f);
      p.y = h - 2;
      p.z = radius * std::cos(d * M_PI / 180.f);
      startRadius = 0.015f * std::sin(f * d * M_PI / 180) + 0.02f;

      q.x = (radius - 0.05f) * std::sin((d + 10) * M_PI / 180.f);
      q.y = h + hStep - 2;
      q.z = (radius - 0.05f) * std::cos((d + 10) * M_PI / 180.f);
      endRadius = 0.015f * std::sin(f * (d + 10) * M_PI / 180) + 0.02f;
      if (d == dStart) {
        const auto rim = glm::mix(q, p, 1.f + endRadius / length(q - p));
        const auto cap = glm::mix(p, rim, 1.f + startRadius / length(rim - p));
        addPoint(glm::vec4(cap, 0.f));
        addPoint(glm::vec4(rim, 0.f));
        addPoint(glm::vec4(p, startRadius));
        addPoint(glm::vec4(q, endRadius));
        indices.push_back(positions.size() - 4);
        colors.push_back(c);
        colors.push_back(c);
      } else if (d + 10 < dStart + dEnd && d + 20 > dStart + dEnd) {
        const auto rim = glm::mix(p, q, 1.f + startRadius / length(p - q));
        const auto cap = glm::mix(q, rim, 1.f + endRadius / length(rim - q));
        addPoint(glm::vec4(p, startRadius));
        addPoint(glm::vec4(q, endRadius));
        addPoint(glm::vec4(rim, 0.f));
        addPoint(glm::vec4(cap, 0.f));
        indices.push_back(positions.size() - 7);
        indices.push_back(positions.size() - 6);
        indices.push_back(positions.size() - 5);
        indices.push_back(positions.size() - 4);
        colors.push_back(c);
        colors.push_back(c);
      } else if ((d != dStart && d != dStart + 10) && d + 20 < dStart + dEnd) {
        addPoint(glm::vec4(p, startRadius));
        indices.push_back(positions.size() - 4);
      }
      colors.push_back(c);
      radius -= 0.05f;
    }
  }

  auto geom = anari::newObject<anari::Geometry>(d, "curve");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, positions.data(), positions.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.radius",
      anari::newArray1D(d, radii.data(), radii.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.color",
      anari::newArray1D(d, colors.data(), colors.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "primitive.index",
      anari::newArray1D(d, indices.data(), indices.size()));
  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", "color");
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  anari::setAndReleaseParameter(
      d, world, "surface", anari::newArray1D(d, &surface));

  anari::release(d, surface);

  return std::make_unique<Scene>(d, world);
}

///////////////////////////////////////////////////////////////////////////////
// Spheres scene //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static ScenePtr generateSpheres(anari::Device d, SpheresConfig config)
{
  auto world = anari::newObject<anari::World>(d);

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> vert_dist(0.f, 1.0f);
  std::uniform_real_distribution<float> color_dist(0.f, 1.f);

  auto positionArray =
      anari::newArray1D(d, ANARI_FLOAT32_VEC3, config.numSpheres);
  auto distArray = anari::newArray1D(d, ANARI_FLOAT32, config.numSpheres);

  float maxDistance = 0.f;
  {
    auto *bp = anari::map<glm::vec3>(d, positionArray);
    auto *ep = bp + config.numSpheres;
    std::for_each(bp, ep, [&](glm::vec3 &p) {
      p.x = vert_dist(rng);
      p.y = vert_dist(rng);
      p.z = vert_dist(rng);
      maxDistance = std::max(maxDistance, glm::length(p));
    });
    auto *bd = anari::map<float>(d, distArray);
    std::transform(bp, ep, bd, [](glm::vec3 &p) { return glm::length(p); });
    anari::unmap(d, positionArray);
    anari::unmap(d, distArray);
  }

  auto colorArray = anari::newArray1D(d, ANARI_FLOAT32_VEC4, config.numSpheres);
  {
    auto *b = anari::map<glm::vec4>(d, colorArray);
    auto *e = b + config.numSpheres;
    std::for_each(b, e, [&](glm::vec4 &c) {
      c.x = color_dist(rng);
      c.y = color_dist(rng);
      c.z = color_dist(rng);
      c.w = 1.f;
    });
    anari::unmap(d, colorArray);
  }

  visrtx::Features features = visrtx::getInstanceFeatures(d, d);
  const bool haveColorMapSampler = features.VISRTX_SAMPLER_COLOR_MAP;

  auto geom = anari::newObject<anari::Geometry>(d, "sphere");
  anari::setParameter(d, geom, "vertex.position", positionArray);
  anari::setParameter(d, geom, "primitive.color", colorArray);
  anari::setParameter(d, geom, "primitive.attribute0", distArray);
  anari::setParameter(d, geom, "radius", config.radius);
  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "transparentMatte");
  anari::Sampler sampler{nullptr};
  if (haveColorMapSampler) {
    sampler = anari::newObject<anari::Sampler>(d, "colorMap");
    anari::setParameter(d, sampler, "inAttribute", "attribute0");
    anari::setParameter(d, sampler, "valueRange", glm::vec2(0.f, maxDistance));
    std::vector<glm::vec3> tf = {glm::vec3(1.f, 1.f, 0.f),
        glm::vec3(0.f, 1.f, 0.f),
        glm::vec3(0.f, 0.f, 1.f)};
    std::vector<float> tfPos = {
        maxDistance * 0.0f, maxDistance * 0.25f, maxDistance * 0.75f};
    anari::setAndReleaseParameter(
        d, sampler, "color", anari::newArray1D(d, tf.data(), tf.size()));
    anari::setAndReleaseParameter(d,
        sampler,
        "color.position",
        anari::newArray1D(d, tfPos.data(), tfPos.size()));
    anari::commitParameters(d, sampler);
    anari::setParameter(d, mat, "color", sampler);
  } else {
    anari::setParameter(d, mat, "color", "color");
  }
  anari::setParameter(d, mat, "opacity", config.opacity);
  anari::commitParameters(d, mat);
  anari::setParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  anari::setAndReleaseParameter(
      d, world, "surface", anari::newArray1D(d, &surface));

  anari::release(d, surface);

  float radius = config.radius;
  int capacity = config.numSpheres;
  int count = config.numSpheres;
  bool useColorMap = true;

  auto retval = std::make_unique<Scene>(
      d,
      world,
      [=]() mutable {
        if (ImGui::DragFloat("radius##spheres", &radius, 0.001f, 0.001f, 1.f)) {
          anari::setParameter(d, geom, "radius", radius);
          anari::commitParameters(d, geom);
        }

        if (ImGui::SliderInt("count##spheres", &count, 1, capacity)) {
          anari::setParameter(d, positionArray, "end", size_t(count));
          anari::setParameter(d, colorArray, "end", size_t(count));
          anari::setParameter(d, distArray, "end", size_t(count));
          anari::commitParameters(d, positionArray);
          anari::commitParameters(d, colorArray);
          anari::commitParameters(d, distArray);
        }

        if (haveColorMapSampler) {
          if (ImGui::RadioButton("gradient colors", useColorMap)) {
            anari::setParameter(d, mat, "color", sampler);
            anari::commitParameters(d, mat);
            useColorMap = true;
          }
          if (ImGui::RadioButton("random colors", !useColorMap)) {
            anari::setParameter(d, mat, "color", "color");
            anari::commitParameters(d, mat);
            useColorMap = false;
          }
        }

        if (ImGui::Button("randomize positions")) {
          rng.seed(std::random_device{}());
          auto *b = anari::map<glm::vec3>(d, positionArray);
          auto *e = b + config.numSpheres;
          std::for_each(b, e, [&](glm::vec3 &p) {
            p.x = vert_dist(rng);
            p.y = vert_dist(rng);
            p.z = vert_dist(rng);
          });
          anari::unmap(d, positionArray);
        }

        ImGui::BeginDisabled(useColorMap);

        if (ImGui::Button("randomize colors")) {
          rng.seed(std::random_device{}());
          auto *b = anari::map<glm::vec4>(d, colorArray);
          auto *e = b + config.numSpheres;
          std::for_each(b, e, [&](glm::vec4 &c) {
            c.x = color_dist(rng);
            c.y = color_dist(rng);
            c.z = color_dist(rng);
            c.w = 1.f;
          });
          anari::unmap(d, colorArray);
        }

        ImGui::EndDisabled();
      },
      [=]() {
        anari::release(d, geom);
        anari::release(d, mat);
        anari::release(d, sampler);
        anari::release(d, positionArray);
        anari::release(d, colorArray);
        anari::release(d, distArray);
      });

  return retval;
}

///////////////////////////////////////////////////////////////////////////////
// Noise volume scene /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static ScenePtr generateNoiseVolume(ANARIDevice d, NoiseVolumeConfig config)
{
  auto world = anari::newObject<anari::World>(d);

  const auto volumeDims = size_t(config.size);
  glm::uvec3 dims(volumeDims);

  static std::mt19937 rng;
  rng.seed(0);
  static std::normal_distribution<float> dist(0.f, 1.0f);

  auto voxelArray =
      anari::newArray3D(d, ANARI_UINT8, volumeDims, volumeDims, volumeDims);

  auto *voxelsBegin = anari::map<uint8_t>(d, voxelArray);
  auto *voxelsEnd = voxelsBegin + (volumeDims * volumeDims * volumeDims);

  std::for_each(voxelsBegin, voxelsEnd, [&](auto &v) { v = dist(rng) * 255; });

  anari::unmap(d, voxelArray);

  auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
  anari::setParameter(d, field, "origin", glm::vec3(-1.f));
  anari::setParameter(d, field, "spacing", glm::vec3(2.f / volumeDims));
  anari::setParameter(d, field, "data", voxelArray);
  anari::commitParameters(d, field);

  auto volume = anari::newObject<anari::Volume>(d, "scivis");
  anari::setAndReleaseParameter(d, volume, "field", field);
  anari::setParameter(d, volume, "densityScale", config.density);

  {
    std::vector<glm::vec3> colors;
    std::vector<float> opacities;

    colors.emplace_back(0.f, 0.f, 1.f);
    colors.emplace_back(0.f, 1.f, 0.f);
    colors.emplace_back(1.f, 0.f, 0.f);

    opacities.emplace_back(0.f);
    opacities.emplace_back(1.f);

    anari::setAndReleaseParameter(
        d, volume, "color", anari::newArray1D(d, colors.data(), colors.size()));
    anari::setAndReleaseParameter(d,
        volume,
        "opacity",
        anari::newArray1D(d, opacities.data(), opacities.size()));
    anari::setParameter(d, volume, "valueRange", glm::vec2(0.f, 255.f));
  }

  anari::commitParameters(d, volume);

  anari::Array1D instanceArray{nullptr};
  int count = 1;

  if (config.instanceVolume) {
    auto instances = makeGridOfInstances(d, nullptr, volume);
    instanceArray = anari::newArray1D(d, ANARI_INSTANCE, instances.size());
    auto *insts = anari::map<void>(d, instanceArray);
    std::memcpy(
        insts, instances.data(), sizeof(ANARIInstance) * instances.size());
    anari::unmap(d, instanceArray);
    count = int(instances.size());
    anari::setParameter(d, world, "instance", instanceArray);

    for (auto i : instances)
      anari::release(d, i);
  } else {
    anari::setAndReleaseParameter(
        d, world, "volume", anari::newArray1D(d, &volume));
  }

  anari::release(d, volume);

  if (config.addPlane && config.instanceVolume) {
    anari::commitParameters(d, world);
    box3 bounds;
    anari::getProperty(d, world, "bounds", bounds, ANARI_WAIT);
    auto planeSurface = makePlane(d, bounds);
    anari::setParameter(
        d, world, "surface", anari::newArray1D(d, &planeSurface));
    anari::release(d, planeSurface);
  }

  int capacity = count;
  auto retval = std::make_unique<Scene>(
      d,
      world,
      [&, d, voxelArray, count, capacity, instanceArray, volumeDims]() mutable {
        if (ImGui::Button("regen data")) {
          rng.seed(std::random_device{}());
          auto *b = anari::map<float>(d, voxelArray);
          auto *e = b + (volumeDims * volumeDims * volumeDims);
          std::for_each(b, e, [&](auto &v) { v = dist(rng); });
          anari::unmap(d, voxelArray);
        }

        if (instanceArray
            && ImGui::SliderInt("count##instances", &count, 1, capacity)) {
          anari::setParameter(d, instanceArray, "end", size_t(count));
          anari::commitParameters(d, instanceArray);
        }
      },
      [=]() {
        anari::release(d, voxelArray);
        anari::release(d, instanceArray);
      });

  return retval;
}

///////////////////////////////////////////////////////////////////////////////
// Gravity volume scene ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct Point
{
  glm::vec3 center;
  float weight;
};

static std::vector<Point> generatePoints(int numPoints)
{
  // create random number distributions for point center and weight
  std::mt19937 gen(0);

  std::uniform_real_distribution<float> centerDistribution(-1.f, 1.f);
  std::uniform_real_distribution<float> weightDistribution(0.1f, 0.3f);

  // populate the points
  std::vector<Point> points(numPoints);

  for (auto &p : points) {
    p.center.x = centerDistribution(gen);
    p.center.y = centerDistribution(gen);
    p.center.z = centerDistribution(gen);

    p.weight = weightDistribution(gen);
  }

  return points;
}

static std::vector<float> generateVoxels(
    const std::vector<Point> &points, glm::ivec3 dims)
{
  // get world coordinate in [-1.f, 1.f] from logical coordinates in [0,
  // volumeDimension)
  auto logicalToWorldCoordinates = [&](int i, int j, int k) {
    return glm::vec3(-1.f + float(i) / float(dims.x - 1) * 2.f,
        -1.f + float(j) / float(dims.y - 1) * 2.f,
        -1.f + float(k) / float(dims.z - 1) * 2.f);
  };

  // generate voxels
  std::vector<float> voxels(size_t(dims.x) * size_t(dims.y) * size_t(dims.z));

  for (int k = 0; k < dims.z; k++) {
    for (int j = 0; j < dims.y; j++) {
      for (int i = 0; i < dims.x; i++) {
        // index in array
        size_t index =
            size_t(k) * dims.z * dims.y + size_t(j) * dims.x + size_t(i);

        // compute volume value
        float value = 0.f;

        for (auto &p : points) {
          glm::vec3 pointCoordinate = logicalToWorldCoordinates(i, j, k);
          const float distance = glm::length(pointCoordinate - p.center);

          // contribution proportional to weighted inverse-square distance
          // (i.e. gravity)
          value += p.weight / (distance * distance);
        }

        voxels[index] = value;
      }
    }
  }

  return voxels;
}

static ScenePtr generateGravityVolume(
    anari::Device d, GravityVolumeConfig config)
{
  auto world = anari::newObject<anari::World>(d);

  const bool withVolume = config.withVolume;
  const bool withGeometry = config.withBodies;
  const int volumeDims = config.size;
  const int numPoints = config.numWells;
  const auto voxelRange = glm::vec2(0.f, 10.f);

  auto points = generatePoints(numPoints);

  if (withVolume) {
    auto voxels = generateVoxels(points, glm::ivec3(volumeDims));

    auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
    anari::setParameter(d, field, "origin", glm::vec3(-1.f));
    anari::setParameter(d, field, "spacing", glm::vec3(2.f / volumeDims));
    anari::setAndReleaseParameter(d,
        field,
        "data",
        anari::newArray3D(
            d, voxels.data(), volumeDims, volumeDims, volumeDims));
    anari::commitParameters(d, field);

    auto volume = anariNewVolume(d, "scivis");
    anari::setAndReleaseParameter(d, volume, "field", field);
    anari::setParameter(d, volume, "densityScale", config.density);

    {
      std::vector<glm::vec3> colors;
      std::vector<float> opacities;

      colors.emplace_back(0.f, 0.f, 1.f);
      colors.emplace_back(0.f, 1.f, 0.f);
      colors.emplace_back(1.f, 0.f, 0.f);

      opacities.emplace_back(0.f);
      opacities.emplace_back(1.f);

      anari::setAndReleaseParameter(d,
          volume,
          "color",
          anari::newArray1D(d, colors.data(), colors.size()));
      anari::setAndReleaseParameter(d,
          volume,
          "opacity",
          anari::newArray1D(d, opacities.data(), opacities.size()));
      anariSetParameter(
          d, volume, "valueRange", ANARI_FLOAT32_BOX1, &voxelRange);
    }

    anari::commitParameters(d, volume);

    anari::setAndReleaseParameter(
        d, world, "volume", anari::newArray1D(d, &volume));
    anari::release(d, volume);
  }

  if (withGeometry) {
    std::vector<glm::vec3> positions(numPoints);
    std::transform(
        points.begin(), points.end(), positions.begin(), [](const Point &p) {
          return p.center;
        });

    auto geom = anari::newObject<anari::Geometry>(d, "sphere");
    anari::setAndReleaseParameter(d,
        geom,
        "vertex.position",
        anari::newArray1D(d, positions.data(), positions.size()));
    anari::setParameter(d, geom, "radius", 0.05f);
    anari::commitParameters(d, geom);

    auto mat = anari::newObject<anari::Material>(d, "matte");
    anari::commitParameters(d, mat);

    auto surface = anari::newObject<anari::Surface>(d);
    anari::setAndReleaseParameter(d, surface, "geometry", geom);
    anari::setAndReleaseParameter(d, surface, "material", mat);
    anari::commitParameters(d, surface);

    anari::setAndReleaseParameter(
        d, world, "surface", anari::newArray1D(d, &surface));
    anari::release(d, surface);
  }

  return std::make_unique<Scene>(d, world);
}

///////////////////////////////////////////////////////////////////////////////
// Obj file scene /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static std::string pathOf(const std::string &filename)
{
#ifdef _WIN32
  const char path_sep = '\\';
#else
  const char path_sep = '/';
#endif

  size_t pos = filename.find_last_of(path_sep);
  if (pos == std::string::npos)
    return "";
  return filename.substr(0, pos + 1);
}

using TextureCache = std::unordered_map<std::string, anari::Sampler>;

static void loadTexture(anari::Device d,
    anari::Material m,
    std::string filename,
    TextureCache &cache)
{
  std::transform(
      filename.begin(), filename.end(), filename.begin(), [](char c) {
        return c == '\\' ? '/' : c;
      });

  anari::Sampler colorTex = cache[filename];
  anari::Sampler opacityTex = cache[filename + "_opacity"];

  if (!colorTex) {
    int width, height, n;
    stbi_set_flip_vertically_on_load(1);
    float *data = stbi_loadf(filename.c_str(), &width, &height, &n, 0);

    if (!data || n < 1) {
      if (!data)
        printf("failed to load texture '%s'\n", filename.c_str());
      else
        printf(
            "texture '%s' with %i channels not loaded\n", filename.c_str(), n);
      return;
    }

    colorTex = anari::newObject<anari::Sampler>(d, "image2D");

    int texelType = ANARI_FLOAT32_VEC4;
    if (n == 3)
      texelType = ANARI_FLOAT32_VEC3;
    else if (n == 2)
      texelType = ANARI_FLOAT32_VEC2;
    else if (n == 1)
      texelType = ANARI_FLOAT32;

    if (texelType == ANARI_FLOAT32_VEC4) {
      opacityTex = anari::newObject<anari::Sampler>(d, "image2D");

      auto colorArray = anari::newArray2D(d, ANARI_FLOAT32_VEC3, width, height);
      auto opacityArray = anari::newArray2D(d, ANARI_FLOAT32, width, height);

      auto *colors = anari::map<glm::vec3>(d, colorArray);
      auto *opacities = anari::map<float>(d, opacityArray);

      for (size_t i = 0; i < size_t(width) * size_t(height); i++) {
        auto *texel = data + (i * 4);
        colors[i] = glm::vec3(texel[0], texel[1], texel[2]);
        opacities[i] = texel[3];
      }

      anari::unmap(d, colorArray);
      anari::unmap(d, opacityArray);

      anari::setAndReleaseParameter(d, colorTex, "image", colorArray);

      anari::setAndReleaseParameter(d, opacityTex, "image", opacityArray);
      anari::setParameter(d, opacityTex, "inAttribute", "attribute0");
      anari::setParameter(d, opacityTex, "wrapMode1", "repeat");
      anari::setParameter(d, opacityTex, "wrapMode2", "repeat");
      anari::setParameter(d, opacityTex, "filter", "bilinear");
      anari::commitParameters(d, opacityTex);

      free(data);
    } else {
      auto array = anariNewArray2D(
          d, data, &anari_free, nullptr, texelType, width, height);
      anari::setAndReleaseParameter(d, colorTex, "image", array);
    }

    anari::setParameter(d, colorTex, "inAttribute", "attribute0");
    anari::setParameter(d, colorTex, "wrapMode1", "repeat");
    anari::setParameter(d, colorTex, "wrapMode2", "repeat");
    anari::setParameter(d, colorTex, "filter", "bilinear");
    anari::commitParameters(d, colorTex);
  }

  cache[filename] = colorTex;
  anari::setAndReleaseParameter(d, m, "color", colorTex);
  if (opacityTex) {
    cache[filename + "_opacity"] = opacityTex;
    anari::setAndReleaseParameter(d, m, "opacity", opacityTex);
  }
}

struct OBJData
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
};

static anari::World loadObj(
    anari::Device d, const OBJData &objdata, const std::string &basePath)
{
  visrtx::Features features = visrtx::getInstanceFeatures(d, d);
  const bool attributeIndexing = features.VISRTX_TRIANGLE_ATTRIBUTE_INDEXING;

  auto world = anari::newObject<anari::World>(d);

  auto inf = std::numeric_limits<float>::infinity();

  std::vector<ANARIMaterial> materials;

  auto defaultMaterial = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, defaultMaterial, "color", glm::vec3(0.f, 1.f, 0.f));
  anari::commitParameters(d, defaultMaterial);

  TextureCache cache;

  for (auto &mat : objdata.materials) {
    auto m = anari::newObject<anari::Material>(d, "transparentMatte");

    anari::setParameter(d, m, "color", ANARI_FLOAT32_VEC3, &mat.diffuse[0]);
    anari::setParameter(d, m, "opacity", ANARI_FLOAT32, &mat.dissolve);

    if (!mat.diffuse_texname.empty())
      loadTexture(d, m, basePath + mat.diffuse_texname, cache);

#if 0
    if (!mat.alpha_texname.empty()) {
      auto opacityTexture = loadTexture(d, basePath + mat.alpha_texname, cache);
      if (opacityTexture)
        anari::setParameter(d, m, "opacity", opacityTexture);
    }
#endif

    anari::commitParameters(d, m);
    materials.push_back(m);
  }

  for (auto &t : cache)
    anari::release(d, t.second);

  std::vector<anari::Surface> meshes;

  auto &v = objdata.attrib.vertices;
  auto &t = objdata.attrib.texcoords;
  auto &n = objdata.attrib.normals;
  anari::Array1D positionArray =
      anari::newArray1D(d, (glm::vec3 *)v.data(), v.size() / 3);
  anari::Array1D texcoordArray = !t.empty()
      ? anari::newArray1D(d, (glm::vec2 *)t.data(), t.size() / 2)
      : nullptr;
  anari::Array1D normalsArray = !n.empty()
      ? anari::newArray1D(d, (glm::vec3 *)n.data(), n.size() / 3)
      : nullptr;

  std::vector<glm::uvec3> vi;
  std::vector<glm::uvec3> vti;
  std::vector<glm::uvec3> vni;

  for (auto &shape : objdata.shapes) {
    auto numSrcIndices = shape.mesh.indices.size();

    vi.clear();
    vti.clear();
    vni.clear();

    size_t numIndices = shape.mesh.indices.size();

    vi.reserve(numIndices);
    vti.reserve(numIndices);
    vni.reserve(numIndices);

    for (size_t i = 0; i < numIndices; i += 3) {
      const auto &i0 = shape.mesh.indices[i + 0];
      const auto &i1 = shape.mesh.indices[i + 1];
      const auto &i2 = shape.mesh.indices[i + 2];

      vi.emplace_back(i0.vertex_index, i1.vertex_index, i2.vertex_index);
      vti.emplace_back(i0.texcoord_index, i1.texcoord_index, i2.texcoord_index);
      vni.emplace_back(i0.normal_index, i1.normal_index, i2.normal_index);
    }

    auto geom = anari::newObject<anari::Geometry>(d, "triangle");

    anari::setParameter(d, geom, "vertex.position", positionArray);
    anari::setAndReleaseParameter(
        d, geom, "primitive.index", anari::newArray1D(d, vi.data(), vi.size()));

    if (attributeIndexing && texcoordArray) {
      anari::setAndReleaseParameter(d,
          geom,
          "vertex.attribute0.index",
          anari::newArray1D(d, vti.data(), vti.size()));
      anari::setParameter(d, geom, "vertex.attribute0", texcoordArray);
    }

    if (attributeIndexing && normalsArray) {
      anari::setAndReleaseParameter(d,
          geom,
          "vertex.normal.index",
          anari::newArray1D(d, vni.data(), vni.size()));
      anari::setParameter(d, geom, "vertex.normal", normalsArray);
    }

    anari::commitParameters(d, geom);

    auto surface = anari::newObject<anari::Surface>(d);

    int matID =
        !shape.mesh.material_ids.empty() ? shape.mesh.material_ids[0] : -1;
    auto mat = matID < 0 ? defaultMaterial : materials[matID];
    anari::setParameter(d, surface, "material", mat);
    anari::setParameter(d, surface, "geometry", geom);

    anari::commitParameters(d, surface);
    anari::release(d, geom);

    meshes.push_back(surface);
  }

  anari::release(d, positionArray);
  anari::release(d, texcoordArray);
  anari::release(d, normalsArray);

  anari::setAndReleaseParameter(
      d, world, "surface", anari::newArray1D(d, meshes.data(), meshes.size()));

  for (auto &m : meshes)
    anari::release(d, m);
  for (auto &m : materials)
    anari::release(d, m);
  anari::release(d, defaultMaterial);

  return world;
}

static ScenePtr loadObjFile(anari::Device d, ObjFileConfig config)
{
  static bool loaded = false;
  static OBJData objdata;

  std::string warn;
  std::string err;
  std::string basePath = pathOf(config.filename);

  if (!loaded) {
    printf("LOADING OBJ FILE: %s\n", config.filename.c_str());
    auto retval = tinyobj::LoadObj(&objdata.attrib,
        &objdata.shapes,
        &objdata.materials,
        &warn,
        &err,
        config.filename.c_str(),
        basePath.c_str(),
        true);

    if (!retval)
      throw std::runtime_error("failed to open/parse obj file!");

    loaded = true;
    printf("DONE!\n");
  }

  printf("constructing ANARIWorld from loaded .obj file\n");
  fflush(stdout);
  auto world = loadObj(d, objdata, basePath);
  fflush(stdout);
  printf("DONE!\n");

  return std::make_unique<Scene>(d, world);
}

///////////////////////////////////////////////////////////////////////////////
// Scene definitions //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

Scene::Scene(
    anari::Device d, anari::World w, UICallback cb, CleanupCallback ccb)
    : m_device(d), m_world(w), m_ui(cb), m_cleanup(ccb)
{
  anari::retain(d, d);
  anari::retain(d, w);
}

Scene::~Scene()
{
  if (m_cleanup)
    m_cleanup();
  anari::release(m_device, m_world);
  anari::release(m_device, m_device);
}

anari::World Scene::world() const
{
  return m_world;
}

void Scene::buildUI()
{
  if (m_ui)
    m_ui();
}

///////////////////////////////////////////////////////////////////////////////
// Scene generation dispatch //////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ScenePtr generateScene(anari::Device d, SceneConfig config)
{
  ScenePtr retval;

  bool addPlane = false;

  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, SpheresConfig>)
          retval = generateSpheres(d, arg);
        else if constexpr (std::is_same_v<T, CylindersConfig>)
          retval = generateCylinders(d, arg);
        else if constexpr (std::is_same_v<T, ConesConfig>)
          retval = generateCones(d, arg);
        else if constexpr (std::is_same_v<T, CurvesConfig>)
          retval = generateCurves(d, arg);
        else if constexpr (std::is_same_v<T, NoiseVolumeConfig>) {
          retval = generateNoiseVolume(d, arg);
          if (arg.instanceVolume && arg.addPlane)
            arg.addPlane = false;
        } else if constexpr (std::is_same_v<T, GravityVolumeConfig>)
          retval = generateGravityVolume(d, arg);
        else if constexpr (std::is_same_v<T, ObjFileConfig>)
          retval = loadObjFile(d, arg);

        addPlane = arg.addPlane;
      },
      config);

  auto world = retval->world();

  if (addPlane) {
    anari::commitParameters(d, world);
    box3 bounds;
    anari::getProperty(d, world, "bounds", bounds, ANARI_WAIT);
    auto planeInst = makePlaneInstance(d, bounds);
    anari::setAndReleaseParameter(
        d, world, "instance", anari::newArray1D(d, &planeInst));
    anari::release(d, planeInst);
  }

  anari::release(d, world);
  return retval;
}
