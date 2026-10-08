/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Generated tangent frames, per ADR 0010 and the normal-maps effort: where no
// tangents are authored, the frame follows attribute0, with T along +dP/du and
// the bitangent along +dP/dv, and corners that share a vertex only share a
// tangent when their normals and texture coordinates match. Pins: a mesh with
// a hard crease (face-varying normals) and a UV seam (a second island rotated
// 90 degrees) shades a tilted normal map identically with generated tangents
// and with authored tangents set to the expected frame, for tilts along both
// +X and +Y. Quads get the same frame from their two-triangle split. A no-map
// render must differ, so the light actually sees the tilt. Runs against
// `physicallyBased`, and `physicallyBasedMDL` when MDL support is compiled in.

// anari_cpp
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using uvec3 = std::array<unsigned int, 3>;
using uvec4 = std::array<unsigned int, 4>;
using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
}

static constexpr uvec2 IMAGE_SIZE = {256, 256};
static constexpr int PIXEL_SAMPLES = 64;
static constexpr int TILES = 16; // per side
static constexpr float TILT = 0.6f; // radians off the surface normal

// The fold: the right face is rotated up 30 degrees about the shared edge
// (the z axis through the origin).
static const float FOLD_C = std::cos(0.5236f);
static const float FOLD_S = std::sin(0.5236f);

enum class Mesh
{
  TRIANGLES, // indexed, face-varying normals and attribute0 (crease + seam)
  QUADS, // two separate quads with vertex normals and attribute0
};

enum class Frame
{
  NO_MAP, // no normal map at all
  GENERATED, // normal map, no authored tangents
  AUTHORED, // normal map, tangents authored to the expected frame
  ROTATED, // normal map, authored tangents rotated 90 degrees: must differ
};

// Per face: positions of its 4 corners (counter-clockwise seen from above),
// its normal, texture coordinates, and the expected tangent (T, w), where w
// makes w * cross(N, T) point along +dP/dv; `rotated` is that frame turned 90
// degrees about N.
struct Face
{
  std::array<vec3, 4> p;
  vec3 n;
  std::array<vec2, 4> uv;
  vec4 tangent;
  vec4 rotated;
};

static std::array<Face, 2> faces()
{
  const float c = FOLD_C, s = FOLD_S;
  // Left face, flat: u along +x, v along +z. cross(N, T) = -z, so w = -1.
  Face left{{vec3{-1.f, 0.f, -1.f},
                vec3{-1.f, 0.f, 1.f},
                vec3{0.f, 0.f, 1.f},
                vec3{0.f, 0.f, -1.f}},
      vec3{0.f, 1.f, 0.f},
      {vec2{0.f, 0.f}, vec2{0.f, 1.f}, vec2{1.f, 1.f}, vec2{1.f, 0.f}},
      vec4{1.f, 0.f, 0.f, -1.f},
      vec4{0.f, 0.f, 1.f, -1.f}};
  // Right face, folded: a separate UV island rotated 90 degrees, u along +z
  // and v along the fold direction (c, s, 0). cross(N, T) = (c, s, 0), w = +1.
  Face right{{vec3{0.f, 0.f, -1.f},
                 vec3{0.f, 0.f, 1.f},
                 vec3{c, s, 1.f},
                 vec3{c, s, -1.f}},
      vec3{-s, c, 0.f},
      {vec2{2.f, 0.f}, vec2{3.f, 0.f}, vec2{3.f, 1.f}, vec2{2.f, 1.f}},
      vec4{0.f, 0.f, 1.f, 1.f},
      vec4{c, s, 0.f, 1.f}};
  return {left, right};
}

static anari::Geometry makeTriangles(ANARIDevice device, Frame frame)
{
  // Six shared positions; the two faces meet along x = 0. Every per-corner
  // value is face-varying, so the shared vertices carry two different normals
  // and two different texture coordinates.
  const auto f = faces();
  const std::array<vec3, 6> pos = {
      f[0].p[0], f[0].p[1], f[0].p[2], f[0].p[3], f[1].p[2], f[1].p[3]};
  // Face corner index -> position index.
  const std::array<std::array<unsigned, 4>, 2> faceVerts = {
      std::array<unsigned, 4>{0, 1, 2, 3}, std::array<unsigned, 4>{3, 2, 4, 5}};

  std::vector<uvec3> idx;
  std::vector<vec3> n;
  std::vector<vec2> uv;
  std::vector<vec4> t;
  for (int fi = 0; fi < 2; fi++) {
    for (const auto &tri :
        {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
      idx.push_back(uvec3{
          faceVerts[fi][tri[0]], faceVerts[fi][tri[1]], faceVerts[fi][tri[2]]});
      for (int k : tri) {
        n.push_back(f[fi].n);
        uv.push_back(f[fi].uv[k]);
        t.push_back(frame == Frame::ROTATED ? f[fi].rotated : f[fi].tangent);
      }
    }
  }

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 6);
  anari::setParameterArray1D(
      device, geom, "primitive.index", idx.data(), idx.size());
  anari::setParameterArray1D(
      device, geom, "faceVarying.normal", n.data(), n.size());
  anari::setParameterArray1D(
      device, geom, "faceVarying.attribute0", uv.data(), uv.size());
  if (frame == Frame::AUTHORED || frame == Frame::ROTATED)
    anari::setParameterArray1D(
        device, geom, "faceVarying.tangent", t.data(), t.size());
  anari::commitParameters(device, geom);
  return geom;
}

static anari::Geometry makeQuads(ANARIDevice device, Frame frame)
{
  const auto f = faces();
  std::vector<vec3> pos, n;
  std::vector<vec2> uv;
  std::vector<vec4> t;
  std::vector<uvec4> idx;
  for (unsigned fi = 0; fi < 2; fi++) {
    for (int k = 0; k < 4; k++) {
      pos.push_back(f[fi].p[k]);
      n.push_back(f[fi].n);
      uv.push_back(f[fi].uv[k]);
      t.push_back(frame == Frame::ROTATED ? f[fi].rotated : f[fi].tangent);
    }
    idx.push_back(uvec4{4 * fi, 4 * fi + 1, 4 * fi + 2, 4 * fi + 3});
  }

  auto geom = anari::newObject<anari::Geometry>(device, "quad");
  anari::setParameterArray1D(
      device, geom, "vertex.position", pos.data(), pos.size());
  anari::setParameterArray1D(
      device, geom, "primitive.index", idx.data(), idx.size());
  anari::setParameterArray1D(device, geom, "vertex.normal", n.data(), n.size());
  anari::setParameterArray1D(
      device, geom, "vertex.attribute0", uv.data(), uv.size());
  if (frame == Frame::AUTHORED || frame == Frame::ROTATED)
    anari::setParameterArray1D(
        device, geom, "vertex.tangent", t.data(), t.size());
  anari::commitParameters(device, geom);
  return geom;
}

// Mean luminance per tile of the image.
static std::vector<double> render(ANARIDevice device,
    const char *subtype,
    Mesh mesh,
    Frame frame,
    const vec3 &tangentSpaceNormal)
{
  auto geom = mesh == Mesh::TRIANGLES ? makeTriangles(device, frame)
                                      : makeQuads(device, frame);

  auto mat = anari::newObject<anari::Material>(device, subtype);
  anari::setParameter(device, mat, "baseColor", vec3{0.8f, 0.8f, 0.8f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (frame != Frame::NO_MAP) {
    // A constant map holding the tangent-space normal itself (the spec's
    // sampler value; no texel decode needed for a float image).
    auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
    anari::setParameter(device, sampler, "inAttribute", "attribute0");
    anari::setParameter(device, sampler, "filter", "nearest");
    const std::array<vec3, 4> img = {tangentSpaceNormal,
        tangentSpaceNormal,
        tangentSpaceNormal,
        tangentSpaceNormal};
    anari::setParameterArray2D(device, sampler, "image", img.data(), 2, 2);
    anari::commitParameters(device, sampler);
    anari::setAndReleaseParameter(device, mat, "normal", sampler);
  }
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);

  // Light from a generic direction, so tilts along any tangent-plane axis of
  // either face change the shading.
  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", vec3{-0.8f, -0.5f, -0.25f});
  anari::setParameter(device, light, "irradiance", 2.f);
  anari::commitParameters(device, light);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 3.f, 0.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -1.f, 0.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 0.f, 1.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto fr = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, fr, "size", IMAGE_SIZE);
  anari::setParameter(device, fr, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, fr, "world", world);
  anari::setAndReleaseParameter(device, fr, "camera", camera);
  anari::setAndReleaseParameter(device, fr, "renderer", renderer);
  anari::commitParameters(device, fr);

  anari::render(device, fr);
  anari::wait(device, fr);

  std::vector<double> tiles(TILES * TILES, 0.0);
  auto fb = anari::map<vec4>(device, fr, "channel.color");
  const uint32_t tw = IMAGE_SIZE[0] / TILES, th = IMAGE_SIZE[1] / TILES;
  for (uint32_t y = 0; y < IMAGE_SIZE[1]; ++y) {
    for (uint32_t x = 0; x < IMAGE_SIZE[0]; ++x) {
      const vec4 &p = fb.data[y * IMAGE_SIZE[0] + x];
      tiles[(y / th) * TILES + x / tw] +=
          (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) / (tw * th);
    }
  }
  anari::unmap(device, fr, "channel.color");
  anari::release(device, fr);
  return tiles;
}

// Largest relative difference over the tiles the reference lights.
static double maxTileRelErr(
    const std::vector<double> &a, const std::vector<double> &ref)
{
  const double peak = *std::max_element(ref.begin(), ref.end());
  if (peak <= 0.0) // a black reference: any light in `a` is infinitely off
    return *std::max_element(a.begin(), a.end()) > 0.0 ? INFINITY : 0.0;
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); i++) {
    if (ref[i] < 0.05 * peak)
      continue;
    worst = std::max(worst, std::abs(a[i] - ref[i]) / ref[i]);
  }
  return worst;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const std::array<const char *, 2> subtypes = {"physicallyBased",
#ifdef VISRTX_TEST_MDL_WRAPPER
      "physicallyBasedMDL"
#else
      nullptr
#endif
  };
  const std::array<std::pair<const char *, vec3>, 2> tilts = {
      std::pair<const char *, vec3>{
          "+x", vec3{std::sin(TILT), 0.f, std::cos(TILT)}},
      std::pair<const char *, vec3>{
          "+y", vec3{0.f, std::sin(TILT), std::cos(TILT)}}};

  bool ok = true;
  for (const char *subtype : subtypes) {
    if (!subtype)
      continue;
    for (Mesh mesh : {Mesh::TRIANGLES, Mesh::QUADS}) {
      const char *meshName = mesh == Mesh::TRIANGLES ? "triangles" : "quads";
      const auto flat =
          render(device, subtype, mesh, Frame::NO_MAP, vec3{0.f, 0.f, 1.f});
      for (const auto &[tiltName, ts] : tilts) {
        const auto generated =
            render(device, subtype, mesh, Frame::GENERATED, ts);
        const auto authored =
            render(device, subtype, mesh, Frame::AUTHORED, ts);
        const auto rotated = render(device, subtype, mesh, Frame::ROTATED, ts);

        const double sameErr = maxTileRelErr(generated, authored);
        const double mapErr = maxTileRelErr(authored, flat);
        const double rotatedErr = maxTileRelErr(generated, rotated);
        printf(
            "%s/%s/tilt %s: generated vs authored maxTileRelErr=%f, "
            "authored vs no map=%f, generated vs rotated=%f\n",
            subtype,
            meshName,
            tiltName,
            sameErr,
            mapErr,
            rotatedErr);
        if (rotatedErr < 0.1) {
          fprintf(stderr,
              "FAIL: %s/%s/tilt %s: generated matches tangents rotated 90 "
              "degrees (%f); authored tangents aren't being used\n",
              subtype,
              meshName,
              tiltName,
              rotatedErr);
          ok = false;
        }
        if (mapErr < 0.1) {
          fprintf(stderr,
              "FAIL: %s/%s/tilt %s: the normal map barely changes the image "
              "(%f); the test can't see the frame\n",
              subtype,
              meshName,
              tiltName,
              mapErr);
          ok = false;
        }
        if (sameErr > 0.03) {
          fprintf(stderr,
              "FAIL: %s/%s/tilt %s: generated tangents don't match the "
              "expected frame (maxTileRelErr=%f > 0.03)\n",
              subtype,
              meshName,
              tiltName,
              sameErr);
          ok = false;
        }
      }
    }
  }

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("generated tangent frames passed\n");
  return 0;
}
