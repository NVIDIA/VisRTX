// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/HDRImage.h"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/importers/detail/pbrt/PbrtParser.hpp"
// stb
#include "stb_image.h"
// ply
#include "tinyply.h"
// std
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>

namespace tsd::io {

using namespace tsd::core;
using namespace tsd::math;

namespace {

constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.f;
constexpr float PI = 3.14159265358979323846f;

// Helpers /////////////////////////////////////////////////////////////////////

mat4 pbrtTransformToMat4(const pbrt::Transform &t)
{
  mat4 m;
  std::memcpy(&m, t.m, sizeof(float) * 16);
  return m;
}

float3 getFloat3(const pbrt::ParamList &p, const std::string &name, float3 def)
{
  auto &v = p.getFloats(name);
  return v.size() >= 3 ? float3(v[0], v[1], v[2]) : def;
}

// Defined later — shared with the lights/area-emitters path so that
// `"spectrum reflectance" [λ v λ v …]`, `"blackbody"`, etc. resolve
// identically for materials, lights, and any other RGB triple lookup.
static float3 resolveEmissionColor(const pbrt::ParamList &params,
    const std::string &name,
    float3 fallback);

float3 getRgb(
    const pbrt::ParamList &p, const std::string &name, float3 def = float3(1.f))
{
  return resolveEmissionColor(p, name, def);
}

float3 transformPoint(const mat4 &m, const float3 &p)
{
  float4 r = linalg::mul(m, float4(p.x, p.y, p.z, 1.f));
  return float3(r.x, r.y, r.z);
}

float3 safeNormalize(const float3 &v, const float3 &fallback)
{
  float len = length(v);
  return len > 1e-12f ? v / len : fallback;
}

// HDRI helpers ////////////////////////////////////////////////////////////////

// Clarberg (2008) equal-area sphere mapping (inverse direction): unit S^2
// vector to (u, v) ∈ [0, 1]^2. Center of the square is +Z, edge midpoints
// are ±X / ±Y, corners are -Z. Matches the VisRTX gpu_math.h
// sphereToEqualAreaSquare so the conversion can round-trip.
float2 sphereToEqualAreaSquare(const float3 &d)
{
  const float x = std::fabs(d.x), y = std::fabs(d.y), z = std::fabs(d.z);
  const float r = std::sqrt(std::max(0.f, 1.f - z));
  const float a = std::max(x, y), b = std::min(x, y);
  const float bb = (a == 0.f) ? 0.f : b / a;
  const float phi = std::atan(bb) * (2.f / PI);
  float vc = phi * r;
  float uc = r - vc;
  if (x < y)
    std::swap(uc, vc);
  if (d.z < 0.f) {
    const float tmp = uc;
    uc = 1.f - vc;
    vc = 1.f - tmp;
  }
  uc = std::copysign(uc, d.x);
  vc = std::copysign(vc, d.y);
  return float2((uc + 1.f) * 0.5f, (vc + 1.f) * 0.5f);
}

// Bilinear sample of a row-major float3 image at normalized (u, v) ∈ [0, 1].
// (u, v) = (0, 0) reads pixel (0, 0); wraps in u, clamps in v.
float3 bilinearWrapU(
    const std::vector<float3> &img, int W, int H, float u, float v)
{
  // Wrap u (longitude is periodic), clamp v (latitude is bounded).
  u = u - std::floor(u);
  v = std::clamp(v, 0.f, 1.f);
  const float fx = u * W - 0.5f;
  const float fy = v * H - 0.5f;
  const int x0 = int(std::floor(fx));
  const int y0 = std::clamp(int(std::floor(fy)), 0, H - 1);
  const int y1 = std::clamp(y0 + 1, 0, H - 1);
  const int x0w = ((x0 % W) + W) % W;
  const int x1w = ((x0 + 1) % W + W) % W;
  const float tx = fx - std::floor(fx);
  const float ty = std::clamp(fy - std::floor(fy), 0.f, 1.f);
  const float3 &c00 = img[y0 * W + x0w];
  const float3 &c10 = img[y0 * W + x1w];
  const float3 &c01 = img[y1 * W + x0w];
  const float3 &c11 = img[y1 * W + x1w];
  const float3 c0 = c00 * (1.f - tx) + c10 * tx;
  const float3 c1 = c01 * (1.f - tx) + c11 * tx;
  return c0 * (1.f - ty) + c1 * ty;
}

// Convert a square equal-area HDRI to equirectangular (2:1 aspect).
// Both buffers follow HDRImage's vertically-flipped layout: row 0 is the
// bottom of the conceptual file. The output convention is +Z up: row 0
// corresponds to the -Z pole, the last row to the +Z pole.
void convertEqualAreaToEquirectangular(const std::vector<float3> &in,
    int inSize,
    std::vector<float3> &out,
    int &outW,
    int &outH)
{
  outW = 2 * inSize;
  outH = inSize;
  out.resize(size_t(outW) * outH);

  for (int j = 0; j < outH; ++j) {
    // PBRT image v (0 at top of file = +Z pole). Output is flipped, so
    // buffer row j corresponds to PBRT v = 1 - (j + 0.5) / outH.
    const float pbrtV = 1.f - (j + 0.5f) / float(outH);
    const float theta = PI * pbrtV;
    const float st = std::sin(theta);
    const float ct = std::cos(theta);
    for (int i = 0; i < outW; ++i) {
      const float u = (i + 0.5f) / float(outW);
      const float phi = 2.f * PI * u;
      const float3 d(st * std::cos(phi), st * std::sin(phi), ct);
      const float2 eq = sphereToEqualAreaSquare(d);
      // Input buffer is also vertically flipped, so PBRT v_pbrt maps to
      // input buffer v_buf = 1 - v_pbrt.
      const float3 c = bilinearWrapU(in, inSize, inSize, eq.x, 1.f - eq.y);
      out[size_t(j) * outW + i] = c;
    }
  }
}

// Shape converters ////////////////////////////////////////////////////////////

// Reverse triangle winding in-place. Triangles are uint3 indices.
void reverseTriangleWinding(uint3 *idx, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    std::swap(idx[i].y, idx[i].z);
}

static GeometryRef buildTriangleMesh(Scene &scene, const pbrt::Shape &shape)
{
  auto &params = shape.params;
  auto &P = params.getFloats("P");
  auto &indices = params.getInts("indices");

  if (P.size() < 3) {
    logWarning("[import_PBRT] trianglemesh: no vertices");
    return {};
  }

  const size_t numVertices = P.size() / 3;
  auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);

  auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, numVertices);
  auto *outPos = posArr->mapAs<float3>();
  std::memcpy(outPos, P.data(), numVertices * sizeof(float3));
  posArr->unmap();
  geom->setParameterObject("vertex.position", *posArr);

  if (!indices.empty()) {
    const size_t numTris = indices.size() / 3;
    auto idxArr = scene.createArray(ANARI_UINT32_VEC3, numTris);
    auto *outIdx = idxArr->mapAs<uint3>();
    for (size_t i = 0; i < numTris; i++)
      outIdx[i] = uint3(indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]);
    if (shape.reverseOrientation)
      reverseTriangleWinding(outIdx, numTris);
    idxArr->unmap();
    geom->setParameterObject("primitive.index", *idxArr);
  }

  auto &N = params.getFloats("N");
  if (N.size() >= 3) {
    const size_t numNormals = N.size() / 3;
    auto nArr = scene.createArray(ANARI_FLOAT32_VEC3, numNormals);
    auto *outN = nArr->mapAs<float3>();
    if (shape.reverseOrientation) {
      for (size_t i = 0; i < numNormals; ++i)
        outN[i] = float3(-N[3 * i], -N[3 * i + 1], -N[3 * i + 2]);
    } else {
      std::memcpy(outN, N.data(), numNormals * sizeof(float3));
    }
    nArr->unmap();
    geom->setParameterObject("vertex.normal", *nArr);
  }

  auto &uv = params.getFloats("uv");
  if (uv.size() >= 2) {
    const size_t numUV = uv.size() / 2;
    auto uvArr = scene.createArray(ANARI_FLOAT32_VEC2, numUV);
    auto *outUV = uvArr->mapAs<float2>();
    std::memcpy(outUV, uv.data(), numUV * sizeof(float2));
    for (size_t i = 0; i < numUV; i++)
      outUV[i].y = 1.f - outUV[i].y;
    uvArr->unmap();
    geom->setParameterObject("vertex.attribute0", *uvArr);
  }

  geom->setName("trianglemesh");
  return geom;
}

static GeometryRef buildSphere(Scene &scene, const pbrt::Shape &shape)
{
  const float radius = shape.params.getFloat("radius", 1.f);

  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);

  auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, 1);
  auto *outPos = posArr->mapAs<float3>();
  outPos[0] = float3(0.f);
  posArr->unmap();
  geom->setParameterObject("vertex.position", *posArr);

  auto radArr = scene.createArray(ANARI_FLOAT32, 1);
  auto *outRad = radArr->mapAs<float>();
  outRad[0] = radius;
  radArr->unmap();
  geom->setParameterObject("vertex.radius", *radArr);

  geom->setName("sphere");
  return geom;
}

static GeometryRef buildCylinder(Scene &scene, const pbrt::Shape &shape)
{
  const float radius = shape.params.getFloat("radius", 1.f);
  const float zmin = shape.params.getFloat("zmin", -1.f);
  const float zmax = shape.params.getFloat("zmax", 1.f);

  auto geom = scene.createObject<Geometry>(tokens::geometry::cylinder);

  auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, 2);
  auto *outPos = posArr->mapAs<float3>();
  outPos[0] = float3(0.f, 0.f, zmin);
  outPos[1] = float3(0.f, 0.f, zmax);
  posArr->unmap();
  geom->setParameterObject("vertex.position", *posArr);

  auto radArr = scene.createArray(ANARI_FLOAT32, 2);
  auto *outRad = radArr->mapAs<float>();
  outRad[0] = radius;
  outRad[1] = radius;
  radArr->unmap();
  geom->setParameterObject("vertex.radius", *radArr);

  geom->setName("cylinder");
  return geom;
}

static GeometryRef buildDisk(Scene &scene, const pbrt::Shape &shape)
{
  // PBRT disk is a planar circle in the z = height plane, centered at the
  // origin. Approximate as triangle fan (or annulus quad strip).
  const float radius = shape.params.getFloat("radius", 1.f);
  const float height = shape.params.getFloat("height", 0.f);
  const float innerRadius = shape.params.getFloat("innerradius", 0.f);
  const float phimax = shape.params.getFloat("phimax", 360.f) * DEG_TO_RAD;

  constexpr int kSegments = 64;
  const bool isAnnulus = innerRadius > 0.f;
  const bool reverse = shape.reverseOrientation;

  auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);

  if (isAnnulus) {
    auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, 2 * (kSegments + 1));
    auto *outPos = posArr->mapAs<float3>();
    for (int i = 0; i <= kSegments; ++i) {
      const float t = phimax * float(i) / float(kSegments);
      const float c = std::cos(t), s = std::sin(t);
      outPos[2 * i + 0] = float3(innerRadius * c, innerRadius * s, height);
      outPos[2 * i + 1] = float3(radius * c, radius * s, height);
    }
    posArr->unmap();
    geom->setParameterObject("vertex.position", *posArr);

    auto idxArr = scene.createArray(ANARI_UINT32_VEC3, 2 * kSegments);
    auto *idx = idxArr->mapAs<uint3>();
    for (int i = 0; i < kSegments; ++i) {
      const uint32_t a = 2 * i, b = a + 1, c = a + 2, d = a + 3;
      idx[2 * i + 0] = uint3(a, b, d);
      idx[2 * i + 1] = uint3(a, d, c);
    }
    if (reverse)
      reverseTriangleWinding(idx, 2 * kSegments);
    idxArr->unmap();
    geom->setParameterObject("primitive.index", *idxArr);
  } else {
    auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, kSegments + 1);
    auto *outPos = posArr->mapAs<float3>();
    outPos[0] = float3(0.f, 0.f, height);
    for (int i = 0; i < kSegments; ++i) {
      const float t = phimax * float(i) / float(kSegments - 1);
      outPos[i + 1] =
          float3(radius * std::cos(t), radius * std::sin(t), height);
    }
    posArr->unmap();
    geom->setParameterObject("vertex.position", *posArr);

    auto idxArr = scene.createArray(ANARI_UINT32_VEC3, kSegments - 1);
    auto *idx = idxArr->mapAs<uint3>();
    for (int i = 0; i < kSegments - 1; ++i)
      idx[i] = uint3(0, uint32_t(i + 1), uint32_t(i + 2));
    if (reverse)
      reverseTriangleWinding(idx, kSegments - 1);
    idxArr->unmap();
    geom->setParameterObject("primitive.index", *idxArr);
  }

  geom->setName("disk");
  return geom;
}

static GeometryRef buildPlyMesh(
    Scene &scene, const pbrt::Shape &shape, const std::string &basePath)
{
  auto plyFile = shape.params.getString("filename", "");
  if (plyFile.empty()) {
    logWarning("[import_PBRT] plymesh: missing filename");
    return {};
  }

  std::string fullPath;
  try {
    fullPath = pbrt::resolveScenePath(basePath, plyFile);
  } catch (const std::exception &e) {
    logWarning("[import_PBRT] plymesh: %s", e.what());
    return {};
  }
  std::ifstream stream(fullPath, std::ios::binary);
  if (!stream) {
    logWarning("[import_PBRT] plymesh: cannot open '%s'", fullPath.c_str());
    return {};
  }

  tinyply::PlyFile file;
  file.parse_header(stream);

  std::shared_ptr<tinyply::PlyData> vertices, normals, texcoords, faces;

  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (...) {
  }
  try {
    normals =
        file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
  } catch (...) {
  }
  try {
    texcoords = file.request_properties_from_element("vertex", {"u", "v"});
  } catch (...) {
    try {
      texcoords = file.request_properties_from_element("vertex", {"s", "t"});
    } catch (...) {
    }
  }
  try {
    faces = file.request_properties_from_element("face", {"vertex_indices"}, 3);
  } catch (...) {
    try {
      faces = file.request_properties_from_element("face", {"vertex_index"}, 3);
    } catch (...) {
    }
  }

  file.read(stream);

  if (!vertices || vertices->t != tinyply::Type::FLOAT32) {
    logWarning("[import_PBRT] plymesh: no FLOAT32 vertex data in '%s'",
        fullPath.c_str());
    return {};
  }

  auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);

  auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, vertices->count);
  posArr->setData(vertices->buffer.get());
  geom->setParameterObject("vertex.position", *posArr);

  if (normals && normals->t == tinyply::Type::FLOAT32) {
    auto nArr = scene.createArray(ANARI_FLOAT32_VEC3, normals->count);
    if (shape.reverseOrientation) {
      auto *src = reinterpret_cast<const float3 *>(normals->buffer.get());
      auto *dst = nArr->mapAs<float3>();
      for (size_t i = 0; i < normals->count; ++i)
        dst[i] = -src[i];
      nArr->unmap();
    } else {
      nArr->setData(normals->buffer.get());
    }
    geom->setParameterObject("vertex.normal", *nArr);
  }

  if (texcoords && texcoords->t == tinyply::Type::FLOAT32) {
    auto uvArr = scene.createArray(ANARI_FLOAT32_VEC2, texcoords->count);
    auto *outUV = uvArr->mapAs<float2>();
    std::memcpy(
        outUV, texcoords->buffer.get(), texcoords->count * sizeof(float2));
    for (size_t i = 0; i < texcoords->count; i++)
      outUV[i].y = 1.f - outUV[i].y;
    uvArr->unmap();
    geom->setParameterObject("vertex.attribute0", *uvArr);
  }

  if (faces) {
    if (faces->t != tinyply::Type::UINT32 && faces->t != tinyply::Type::INT32) {
      logWarning(
          "[import_PBRT] plymesh '%s': unsupported face index type, skipping",
          fullPath.c_str());
    } else {
      auto idxArr = scene.createArray(ANARI_UINT32_VEC3, faces->count);
      auto *idx = idxArr->mapAs<uint3>();
      std::memcpy(idx, faces->buffer.get(), faces->count * sizeof(uint3));
      if (shape.reverseOrientation)
        reverseTriangleWinding(idx, faces->count);
      idxArr->unmap();
      geom->setParameterObject("primitive.index", *idxArr);
    }
  }

  geom->setName(plyFile.c_str());
  return geom;
}

// Evaluate a cubic Bezier curve at parameter t (0..1) for control points
// P0..P3.
float3 evalBezier3(const float3 &p0,
    const float3 &p1,
    const float3 &p2,
    const float3 &p3,
    float t)
{
  const float u = 1.f - t;
  return u * u * u * p0 + 3.f * u * u * t * p1 + 3.f * u * t * t * p2
      + t * t * t * p3;
}

// Evaluate a uniform cubic B-spline segment at t (0..1) for control points
// P0..P3 (the segment between knots; converts to Bezier basis internally).
float3 evalBspline3(const float3 &p0,
    const float3 &p1,
    const float3 &p2,
    const float3 &p3,
    float t)
{
  const float u = 1.f - t;
  const float b0 = u * u * u / 6.f;
  const float b1 = (3.f * t * t * t - 6.f * t * t + 4.f) / 6.f;
  const float b2 = (-3.f * t * t * t + 3.f * t * t + 3.f * t + 1.f) / 6.f;
  const float b3 = t * t * t / 6.f;
  return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3;
}

// PBRT v4 'curve' shape: a single cubic curve segment with widths at the
// endpoints. Tessellate into a polyline of N+1 vertices. ANARI 'curve'
// geometry takes vertex.position + vertex.radius + primitive.index where
// each index marks the start of a (i, i+1) segment.
static GeometryRef buildCurve(Scene &scene, const pbrt::Shape &shape)
{
  const auto &params = shape.params;
  const auto &P = params.getFloats("P");
  if (P.size() < 12) {
    logWarning(
        "[import_PBRT] curve: needs 4 control points (got %zu)", P.size() / 3);
    return {};
  }

  const float3 cp[4] = {float3(P[0], P[1], P[2]),
      float3(P[3], P[4], P[5]),
      float3(P[6], P[7], P[8]),
      float3(P[9], P[10], P[11])};

  const auto basis = params.getString("basis", "bezier");
  const float w0 = params.has("width0") ? params.getFloat("width0")
                                        : params.getFloat("width", 1.f);
  const float w1 = params.has("width1") ? params.getFloat("width1")
                                        : params.getFloat("width", 1.f);

  const auto curveType = params.getString("type", "cylinder");
  if (curveType == "ribbon") {
    logWarning("[import_PBRT] curve type 'ribbon' approximated as cylinder");
  }

  constexpr int kSegments = 16;
  const int numVerts = kSegments + 1;

  auto geom = scene.createObject<Geometry>(tokens::geometry::curve);

  auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, numVerts);
  auto *pos = posArr->mapAs<float3>();
  auto radArr = scene.createArray(ANARI_FLOAT32, numVerts);
  auto *rad = radArr->mapAs<float>();
  for (int i = 0; i < numVerts; ++i) {
    const float t = float(i) / float(kSegments);
    pos[i] = (basis == "bspline") ? evalBspline3(cp[0], cp[1], cp[2], cp[3], t)
                                  : evalBezier3(cp[0], cp[1], cp[2], cp[3], t);
    rad[i] = 0.5f * ((1.f - t) * w0 + t * w1);
  }
  posArr->unmap();
  radArr->unmap();
  geom->setParameterObject("vertex.position", *posArr);
  geom->setParameterObject("vertex.radius", *radArr);

  auto idxArr = scene.createArray(ANARI_UINT32, kSegments);
  auto *idx = idxArr->mapAs<uint32_t>();
  for (int i = 0; i < kSegments; ++i)
    idx[i] = uint32_t(i);
  idxArr->unmap();
  geom->setParameterObject("primitive.index", *idxArr);

  geom->setName("curve");
  return geom;
}

// PBRT `bilinearmesh`: 4-vertex bilinear patches indexed in groups of 4
// (or implicit `0..N-1` if `indices` is absent). Triangulate each quad into
// (a,b,c) + (a,c,d) and emit as a regular triangle mesh.
static GeometryRef buildBilinearMesh(Scene &scene, const pbrt::Shape &shape)
{
  auto &params = shape.params;
  auto &P = params.getFloats("P");
  auto &indices = params.getInts("indices");
  if (P.size() < 12) {
    logWarning("[import_PBRT] bilinearmesh: not enough vertices");
    return {};
  }

  const size_t numVertices = P.size() / 3;
  std::vector<int> patchIndices;
  if (indices.empty()) {
    if (numVertices % 4 != 0) {
      logWarning("[import_PBRT] bilinearmesh: vertex count not multiple of 4");
      return {};
    }
    patchIndices.resize(numVertices);
    for (size_t i = 0; i < numVertices; ++i)
      patchIndices[i] = int(i);
  } else {
    if (indices.size() % 4 != 0) {
      logWarning("[import_PBRT] bilinearmesh: index count not multiple of 4");
      return {};
    }
    patchIndices = indices;
  }

  const size_t numPatches = patchIndices.size() / 4;
  auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);

  auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, numVertices);
  auto *outPos = posArr->mapAs<float3>();
  std::memcpy(outPos, P.data(), numVertices * sizeof(float3));
  posArr->unmap();
  geom->setParameterObject("vertex.position", *posArr);

  auto idxArr = scene.createArray(ANARI_UINT32_VEC3, numPatches * 2);
  auto *outIdx = idxArr->mapAs<uint3>();
  for (size_t p = 0; p < numPatches; ++p) {
    const uint32_t a = uint32_t(patchIndices[p * 4 + 0]);
    const uint32_t b = uint32_t(patchIndices[p * 4 + 1]);
    const uint32_t c = uint32_t(patchIndices[p * 4 + 2]);
    const uint32_t d = uint32_t(patchIndices[p * 4 + 3]);
    outIdx[p * 2 + 0] = uint3(a, b, c);
    outIdx[p * 2 + 1] = uint3(a, c, d);
  }
  if (shape.reverseOrientation)
    reverseTriangleWinding(outIdx, numPatches * 2);
  idxArr->unmap();
  geom->setParameterObject("primitive.index", *idxArr);

  geom->setName("bilinearmesh");
  return geom;
}

// PBRT `loopsubdiv`: Loop subdivision surface defined by a base triangle
// mesh + level count. We render the base mesh as-is (no subdivision) — the
// silhouette will be coarser than PBRT's reference but shading is correct.
static GeometryRef buildLoopSubdiv(Scene &scene, const pbrt::Shape &shape)
{
  auto geom = buildTriangleMesh(scene, shape);
  if (geom)
    geom->setName("loopsubdiv");
  return geom;
}

static GeometryRef buildShapeGeometry(
    Scene &scene, const pbrt::Shape &shape, const std::string &basePath)
{
  if (shape.type == "trianglemesh")
    return buildTriangleMesh(scene, shape);
  if (shape.type == "sphere")
    return buildSphere(scene, shape);
  if (shape.type == "cylinder")
    return buildCylinder(scene, shape);
  if (shape.type == "disk")
    return buildDisk(scene, shape);
  if (shape.type == "plymesh")
    return buildPlyMesh(scene, shape, basePath);
  if (shape.type == "curve")
    return buildCurve(scene, shape);
  if (shape.type == "bilinearmesh")
    return buildBilinearMesh(scene, shape);
  if (shape.type == "loopsubdiv")
    return buildLoopSubdiv(scene, shape);
  logWarning("[import_PBRT] unsupported shape type: %s", shape.type.c_str());
  return {};
}

// Wrap a (geom, mat) into a Surface and place it under a transform node
// holding the shape's local transform.
static SurfaceRef makeShapeSurface(
    Scene &scene, const pbrt::Shape &shape, GeometryRef geom, MaterialRef mat)
{
  if (shape.reverseOrientation && shape.type != "trianglemesh"
      && shape.type != "plymesh" && shape.type != "disk") {
    logWarning(
        "[import_PBRT] ReverseOrientation on '%s' shape is not supported",
        shape.type.c_str());
  }
  return scene.createSurface(geom->name().c_str(), geom, mat);
}

static void convertShape(Scene &scene,
    const pbrt::Shape &shape,
    MaterialRef mat,
    LayerNodeRef parent,
    const std::string &basePath)
{
  auto geom = buildShapeGeometry(scene, shape, basePath);
  if (!geom)
    return;
  auto surface = makeShapeSurface(scene, shape, geom, mat);
  auto xfmNode = scene.insertChildTransformNode(
      parent, pbrtTransformToMat4(shape.objectToWorld));
  scene.insertChildObjectNode(xfmNode, surface);
}

// Material, light, camera converters //////////////////////////////////////////

// Result of resolving a PBRT texture chain (imagemap, scale, mix, constant).
// A `Sampler` represents `tint * sample + offset` baked into the sampler's
// outTransform/outOffset so the device performs the affine combination at
// sample time — no CPU re-allocation. A `Constant` is a final rgb value
// that will overwrite the material's parameter directly.
struct BakedTexture
{
  enum class Kind
  {
    None,
    Constant,
    Sampler
  } kind{Kind::None};
  float3 constant{1.f};
  SamplerRef sampler;
  float3 tint{1.f};
  float3 offset{0.f};
};

namespace {
inline float3 cwMul(const float3 &x, const float3 &y)
{
  return float3(x.x * y.x, x.y * y.y, x.z * y.z);
}
inline float3 cwLerp(const float3 &a, const float3 &b, const float3 &t)
{
  return float3(a.x * (1.f - t.x) + b.x * t.x,
      a.y * (1.f - t.y) + b.y * t.y,
      a.z * (1.f - t.z) + b.z * t.z);
}
} // namespace

// Multiplicative combinator (PBRT `scale`):
//   `(o1 + t1*S1) * c2  =  (o1*c2) + (t1*c2)*S1`
static BakedTexture combineMul(const BakedTexture &a, const BakedTexture &b)
{
  if (a.kind == BakedTexture::Kind::None)
    return b;
  if (b.kind == BakedTexture::Kind::None)
    return a;
  if (a.kind == BakedTexture::Kind::Constant
      && b.kind == BakedTexture::Kind::Constant) {
    BakedTexture out;
    out.kind = BakedTexture::Kind::Constant;
    out.constant = cwMul(a.constant, b.constant);
    return out;
  }
  if (a.kind == BakedTexture::Kind::Sampler
      && b.kind == BakedTexture::Kind::Sampler) {
    logWarning(
        "[import_PBRT] 'scale' of two image textures not supported, "
        "using first with second's tint applied");
    BakedTexture out = a;
    out.tint = cwMul(a.tint, b.tint);
    out.offset = cwMul(a.offset, b.tint);
    return out;
  }
  const BakedTexture &samp = (a.kind == BakedTexture::Kind::Sampler) ? a : b;
  const BakedTexture &con = (a.kind == BakedTexture::Kind::Sampler) ? b : a;
  BakedTexture out = samp;
  out.tint = cwMul(samp.tint, con.constant);
  out.offset = cwMul(samp.offset, con.constant);
  return out;
}

// Linear-interpolation combinator (PBRT `mix`):
//   result = lerp(a, b, amount), per-channel.
//   Sampler{t,o,S} expands to (o + t*S); insert into the lerp formula and
//   re-bin into a sampler's affine.
static BakedTexture combineMix(
    const BakedTexture &a, const BakedTexture &b, const float3 &amount)
{
  if (a.kind == BakedTexture::Kind::None)
    return b;
  if (b.kind == BakedTexture::Kind::None)
    return a;
  const float3 ia(1.f - amount.x, 1.f - amount.y, 1.f - amount.z);
  if (a.kind == BakedTexture::Kind::Constant
      && b.kind == BakedTexture::Kind::Constant) {
    BakedTexture out;
    out.kind = BakedTexture::Kind::Constant;
    out.constant = cwLerp(a.constant, b.constant, amount);
    return out;
  }
  if (a.kind == BakedTexture::Kind::Sampler
      && b.kind == BakedTexture::Kind::Sampler) {
    logWarning(
        "[import_PBRT] 'mix' of two image textures not supported, "
        "using first with weighted tint applied");
    BakedTexture out = a;
    out.tint = cwMul(a.tint, ia);
    out.offset = cwMul(a.offset, ia);
    return out;
  }
  // Exactly one Sampler, one Constant.
  if (a.kind == BakedTexture::Kind::Sampler) {
    // (1-α)(oa + ta*S) + α*cb = ((1-α)oa + α*cb) + ((1-α)ta)*S
    BakedTexture out = a;
    out.tint = cwMul(a.tint, ia);
    out.offset = float3(a.offset.x * ia.x + b.constant.x * amount.x,
        a.offset.y * ia.y + b.constant.y * amount.y,
        a.offset.z * ia.z + b.constant.z * amount.z);
    return out;
  }
  // b is Sampler, a is Constant: (1-α)*ca + α*(ob + tb*S) = ((1-α)ca + α*ob) +
  // (α*tb)*S
  BakedTexture out = b;
  out.tint = cwMul(b.tint, amount);
  out.offset = float3(a.constant.x * ia.x + b.offset.x * amount.x,
      a.constant.y * ia.y + b.offset.y * amount.y,
      a.constant.z * ia.z + b.offset.z * amount.z);
  return out;
}

// Apply PBRT v4's UV-coordinate transform (`uscale`, `vscale`, `udelta`,
// `vdelta`) to a sampler via its `inTransform`/`inOffset`. PBRT samples the
// image at `(us*u + ud, vs*v + vd)` in its v-up convention. Our importer has
// already flipped each vertex's v to ANARI's v-down convention, and the
// image is also stored top-down, so the equivalent fetch on our side is
// `(us*u_a + ud, vs*v_a + (1 - vs - vd))` — the v-flip cancels into the
// constant offset.
static void applyPbrtUvTransform(
    SamplerRef &sampler, const pbrt::ParamList &params)
{
  const float us = params.getFloat("uscale", 1.f);
  const float vs = params.getFloat("vscale", 1.f);
  const float ud = params.getFloat("udelta", 0.f);
  const float vd = params.getFloat("vdelta", 0.f);
  if (us == 1.f && vs == 1.f && ud == 0.f && vd == 0.f)
    return;
  mat4 m{float4(us, 0.f, 0.f, 0.f),
      float4(0.f, vs, 0.f, 0.f),
      float4(0.f, 0.f, 1.f, 0.f),
      float4(0.f, 0.f, 0.f, 1.f)};
  sampler->setParameter("inTransform", m);
  sampler->setParameter("inOffset", float4(ud, 1.f - vs - vd, 0.f, 0.f));
}

static BakedTexture bakeTexture(Scene &scene,
    const pbrt::Scene &pbrtScene,
    const std::string &textureName,
    const std::string &basePath,
    TextureCache &texCache);

// Resolve a PBRT texture-or-constant slot ("rgb tex1" / "float tex1" /
// "texture tex1"). Used by both `scale` and `mix`.
static BakedTexture bakeTextureSlot(Scene &scene,
    const pbrt::ParamList &params,
    const std::string &paramName,
    const pbrt::Scene &pbrtScene,
    const std::string &basePath,
    TextureCache &texCache)
{
  auto it = params.values.find(paramName);
  if (it == params.values.end()) {
    BakedTexture out;
    out.kind = BakedTexture::Kind::Constant;
    out.constant = float3(1.f);
    return out;
  }
  if (auto *floats = std::get_if<std::vector<float>>(&it->second)) {
    BakedTexture out;
    out.kind = BakedTexture::Kind::Constant;
    if (floats->size() >= 3)
      out.constant = float3((*floats)[0], (*floats)[1], (*floats)[2]);
    else if (!floats->empty())
      out.constant = float3((*floats)[0]);
    return out;
  }
  if (auto *strings = std::get_if<std::vector<std::string>>(&it->second);
      strings && !strings->empty()) {
    return bakeTexture(scene, pbrtScene, (*strings)[0], basePath, texCache);
  }
  return {};
}

// Read a PBRT `mix` amount: float (broadcast to rgb) or per-channel rgb.
// Texture-driven amounts can't be baked statically; warn and use 0.5.
static float3 bakeMixAmount(const pbrt::ParamList &params)
{
  auto it = params.values.find("amount");
  if (it == params.values.end())
    return float3(0.5f);
  if (auto *floats = std::get_if<std::vector<float>>(&it->second)) {
    if (floats->size() >= 3)
      return float3((*floats)[0], (*floats)[1], (*floats)[2]);
    if (!floats->empty())
      return float3((*floats)[0]);
    return float3(0.5f);
  }
  logWarning(
      "[import_PBRT] 'mix' texture-driven amount not supported, "
      "using 0.5");
  return float3(0.5f);
}

static BakedTexture bakeTexture(Scene &scene,
    const pbrt::Scene &pbrtScene,
    const std::string &textureName,
    const std::string &basePath,
    TextureCache &texCache)
{
  auto texIt = pbrtScene.textures.find(textureName);
  if (texIt == pbrtScene.textures.end()) {
    logWarning("[import_PBRT] texture '%s' not found", textureName.c_str());
    return {};
  }
  const auto &texDef = texIt->second;

  if (texDef.implType == "imagemap") {
    auto filename = texDef.params.getString("filename");
    if (filename.empty())
      return {};
    std::string fullPath;
    try {
      fullPath = pbrt::resolveScenePath(basePath, filename);
    } catch (const std::exception &e) {
      logWarning(
          "[import_PBRT] texture '%s': %s", textureName.c_str(), e.what());
      return {};
    }
    // PBRT splits image textures by colorType: "spectrum" is sRGB color
    // data, "float" is linear scalar data (roughness, masks, bumps).
    const bool isLinear = (texDef.colorType == "float");
    auto sampler = importTexture(scene, fullPath, texCache, isLinear);
    if (!sampler)
      return {};
    applyPbrtUvTransform(sampler, texDef.params);
    // Per the ANARI sampler spec, a fetched texel is completed to four
    // components with the missing first three defaulting to 0. A 1-channel
    // grayscale image bound to a color slot therefore reads `baseColor.xyz
    // = (v, 0, 0)` — a red tint on what should be a neutral mask (e.g.
    // villa's `LampMask.png` for `Verre.001`). For a "spectrum" texture,
    // broadcast .x into .xyz via the sampler's outTransform so the
    // grayscale value lands in every colour channel; downstream tint via
    // applyAffineToSampler composes on top.
    if (texDef.colorType != "float") {
      if (auto *image = sampler->parameterValueAsObject<Array>("image");
          image && image->elementType() == ANARI_FLOAT32) {
        mat4 broadcast{float4(1.f, 1.f, 1.f, 0.f),
            float4(0.f, 0.f, 0.f, 0.f),
            float4(0.f, 0.f, 0.f, 0.f),
            float4(0.f, 0.f, 0.f, 1.f)};
        sampler->setParameter("outTransform", broadcast);
      }
    }
    BakedTexture out;
    out.kind = BakedTexture::Kind::Sampler;
    out.sampler = sampler;
    return out;
  }

  if (texDef.implType == "constant") {
    BakedTexture out;
    out.kind = BakedTexture::Kind::Constant;
    auto &v = texDef.params.getFloats("value");
    if (v.size() >= 3)
      out.constant = float3(v[0], v[1], v[2]);
    else if (!v.empty())
      out.constant = float3(v[0]);
    return out;
  }

  if (texDef.implType == "scale") {
    // PBRT v4 'scale' takes `tex` and `scale` (each may be float, rgb, or
    // texture-ref) — not `tex1` / `tex2`. With the old keys, every `scale`
    // chain in `crown.pbrt` silently resolved to Constant(1) and the
    // referenced imagemaps never made it into the scene.
    auto a = bakeTextureSlot(
        scene, texDef.params, "tex", pbrtScene, basePath, texCache);
    auto b = bakeTextureSlot(
        scene, texDef.params, "scale", pbrtScene, basePath, texCache);
    return combineMul(a, b);
  }

  if (texDef.implType == "mix") {
    auto a = bakeTextureSlot(
        scene, texDef.params, "tex1", pbrtScene, basePath, texCache);
    auto b = bakeTextureSlot(
        scene, texDef.params, "tex2", pbrtScene, basePath, texCache);
    return combineMix(a, b, bakeMixAmount(texDef.params));
  }

  logWarning("[import_PBRT] unsupported texture type '%s' for '%s'",
      texDef.implType.c_str(),
      textureName.c_str());
  return {};
}

// Bake a per-channel affine (`tint * sample + offset`) onto a sampler via its
// outTransform/outOffset, so the device handles modulation at sample time.
// The composition is `result = diag(tint) * (cur_outTransform * texel +
// cur_outOffset) + offset`, so any leaf-side outTransform (e.g. the
// grayscale-to-RGB broadcast set in `bakeTexture`) survives downstream tint.
static void applyAffineToSampler(
    SamplerRef &sampler, const float3 &tint, const float3 &offset)
{
  const bool tintIsIdentity = tint.x == 1.f && tint.y == 1.f && tint.z == 1.f;
  const bool offsetIsZero =
      offset.x == 0.f && offset.y == 0.f && offset.z == 0.f;
  if (tintIsIdentity && offsetIsZero)
    return;
  const mat4 identity{float4(1.f, 0.f, 0.f, 0.f),
      float4(0.f, 1.f, 0.f, 0.f),
      float4(0.f, 0.f, 1.f, 0.f),
      float4(0.f, 0.f, 0.f, 1.f)};
  mat4 cur = sampler->parameterValueAs<mat4>("outTransform").value_or(identity);
  float4 curOff =
      sampler->parameterValueAs<float4>("outOffset").value_or(float4(0.f));
  mat4 diag{float4(tint.x, 0.f, 0.f, 0.f),
      float4(0.f, tint.y, 0.f, 0.f),
      float4(0.f, 0.f, tint.z, 0.f),
      float4(0.f, 0.f, 0.f, 1.f)};
  mat4 m = linalg::mul(diag, cur);
  float4 o =
      linalg::mul(diag, curOff) + float4(offset.x, offset.y, offset.z, 0.f);
  sampler->setParameter("outTransform", m);
  sampler->setParameter("outOffset", o);
}

static void resolveTexture(Scene &scene,
    MaterialRef mat,
    const std::string &paramName,
    const std::string &texParamName,
    const pbrt::MaterialDef &matDef,
    const pbrt::Scene &pbrtScene,
    const std::string &basePath,
    TextureCache &texCache,
    anari::DataType paramType = ANARI_FLOAT32_VEC3)
{
  auto texName = matDef.params.getString(texParamName);
  if (texName.empty())
    return;

  auto baked = bakeTexture(scene, pbrtScene, texName, basePath, texCache);
  switch (baked.kind) {
  case BakedTexture::Kind::None:
    return;
  case BakedTexture::Kind::Constant:
    // The texture's value IS the parameter — overwrite. PBRT scenes pick
    // either `"rgb foo" […]` or `"texture foo" "name"`, never both, so this
    // doesn't fight a user-authored constant.
    if (paramType == ANARI_FLOAT32)
      mat->setParameter(paramName.c_str(), baked.constant.x);
    else
      mat->setParameter(paramName.c_str(), ANARI_FLOAT32_VEC3, &baked.constant);
    return;
  case BakedTexture::Kind::Sampler:
    applyAffineToSampler(baked.sampler, baked.tint, baked.offset);
    mat->setParameterObject(paramName.c_str(), *baked.sampler);
    return;
  }
}

// Walk a PBRT texture name through any `scale` wrappers to its leaf
// `imagemap` and accumulate the constant scale factor along the way.
// Used for "texture displacement" where we need direct access to the
// height pixels — the BakedTexture pipeline produces a sampler but
// hides the underlying file path.
static bool resolveImagemapChain(const pbrt::Scene &pbrtScene,
    const std::string &texName,
    const std::string &basePath,
    std::string &outPath,
    float &outScale)
{
  auto texIt = pbrtScene.textures.find(texName);
  if (texIt == pbrtScene.textures.end())
    return false;
  const auto &t = texIt->second;

  if (t.implType == "imagemap") {
    auto filename = t.params.getString("filename");
    if (filename.empty())
      return false;
    try {
      outPath = pbrt::resolveScenePath(basePath, filename);
    } catch (const std::exception &) {
      return false;
    }
    return true;
  }

  if (t.implType == "scale") {
    // PBRT v4 `scale` has two slots, `tex` and `scale`, each either a
    // float/rgb constant or a texture reference. Multiply through the
    // constant factors and recurse into the (only) textured slot.
    float k = 1.f;
    std::string nextTex;
    for (const char *slot : {"tex", "scale"}) {
      auto it = t.params.values.find(slot);
      if (it == t.params.values.end())
        continue;
      if (auto *fv = std::get_if<std::vector<float>>(&it->second);
          fv && !fv->empty())
        k *= (*fv)[0];
      else if (auto *sv = std::get_if<std::vector<std::string>>(&it->second);
          sv && !sv->empty() && nextTex.empty())
        nextTex = (*sv)[0];
    }
    if (nextTex.empty())
      return false;
    float subScale = 1.f;
    if (!resolveImagemapChain(pbrtScene, nextTex, basePath, outPath, subScale))
      return false;
    outScale *= k * subScale;
    return true;
  }

  return false;
}

// Derive a tangent-space normal map from a single-channel height map. PBRT
// applies `texture displacement` as actual geometric displacement; ANARI
// has no equivalent, so the best fallback is a bump-style normal map
// built from the height gradient via central differences.
//
// `heightScale` carries any `scale` factor that PBRT applied to the
// height. The fixed `kBumpStrength` boost exists because PBRT scales
// (e.g. 0.25 in crown.pbrt) are calibrated for geometric displacement;
// a tangent-only fake of the same scale would be visually invisible.
static SamplerRef importHeightAsNormalMap(Scene &scene,
    const std::string &filepath,
    float heightScale,
    TextureCache &texCache)
{
  // Cache under a separate key so we don't collide with any value-domain
  // sampler that may already exist for the same file.
  const std::string cacheKey = filepath + "::normal";
  auto cached = texCache[cacheKey];

  if (!cached.valid()) {
    int w = 0, h = 0, channels = 0;
    stbi_ldr_to_hdr_scale(1.f);
    stbi_ldr_to_hdr_gamma(1.f);
    float *raw = stbi_loadf(filepath.c_str(), &w, &h, &channels, 1);
    if (!raw) {
      logWarning(
          "[import_PBRT] displacement: failed to load '%s'", filepath.c_str());
      return {};
    }

    constexpr float kBumpStrength = 16.f;
    const float k = heightScale * kBumpStrength;

    auto arr = scene.createArray(ANARI_FLOAT32_VEC4, size_t(w), size_t(h));
    auto *out = arr->mapAs<float4>();
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int xp = (x + 1) % w;
        const int xm = (x - 1 + w) % w;
        const int yp = std::min(y + 1, h - 1);
        const int ym = std::max(y - 1, 0);
        const float hx = raw[y * w + xp] - raw[y * w + xm];
        const float hy = raw[yp * w + x] - raw[ym * w + x];
        const float nx = -hx * k;
        const float ny = -hy * k;
        const float nz = 1.f;
        const float invLen = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
        // Pack [-1,1] -> [0,1] (glTF normal-map convention).
        out[size_t(y) * w + x] = float4(nx * invLen * 0.5f + 0.5f,
            ny * invLen * 0.5f + 0.5f,
            nz * invLen * 0.5f + 0.5f,
            1.f);
      }
    }
    arr->unmap();
    stbi_image_free(raw);

    cached = arr;
    texCache[cacheKey] = cached;
  }

  auto sampler = scene.createObject<Sampler>(tokens::sampler::image2D);
  sampler->setParameterObject("image", *cached);
  sampler->setParameter("inAttribute", "attribute0");
  sampler->setParameter("wrapMode1", "repeat");
  sampler->setParameter("wrapMode2", "repeat");
  sampler->setParameter("filter", "linear");
  sampler->setName((fileOf(filepath) + "_bump").c_str());
  return sampler;
}

// Approximate normal-incidence reflectance for common PBRT named metal spectra.
// Values derived from Fresnel equations using published optical constants.
static float3 conductorSpectrumToRgb(const std::string &etaName)
{
  if (etaName.find("Cu") != std::string::npos)
    return float3(0.96f, 0.64f, 0.54f);
  if (etaName.find("Au") != std::string::npos)
    return float3(1.00f, 0.78f, 0.34f);
  if (etaName.find("Ag") != std::string::npos)
    return float3(0.97f, 0.96f, 0.91f);
  if (etaName.find("Fe") != std::string::npos)
    return float3(0.56f, 0.57f, 0.58f);
  if (etaName.find("Cr") != std::string::npos)
    return float3(0.55f, 0.55f, 0.55f);
  if (etaName.find("Ni") != std::string::npos)
    return float3(0.66f, 0.61f, 0.53f);
  if (etaName.find("Ti") != std::string::npos)
    return float3(0.54f, 0.50f, 0.45f);
  if (etaName.find("Pt") != std::string::npos)
    return float3(0.67f, 0.64f, 0.59f);
  // Default: aluminum (also covers "metal-Al-eta" and unknown spectra)
  return float3(0.91f, 0.92f, 0.92f);
}

// Highest IOR we let through to ANARI for a dielectric. Real gems top
// out at diamond (~2.42); above ~2.5 the critical angle for total
// internal reflection drops under 24°, so most rays loop inside the
// medium before exiting and the gem renders as polished chrome instead
// of as a coloured stone. PBRT scenes (including crown.pbrt) routinely
// author 3.0–3.5 for stylistic punch; PBRT cancels the resulting
// over-bright Fresnel with spectral path tracing and aggressive
// volumetric absorption — single-scalar IOR + ANARI volumes can't.
// Capping at 2.5 preserves "diamond-like" reflectivity while letting
// the medium colour read through.
constexpr float kMaxDielectricIor = 2.5f;

// Resolve a PBRT dielectric `eta` to a scalar ANARI IOR. PBRT v4 lets the
// caller write `"float eta" [1.5]`, `"spectrum eta" [λ₁ η₁ λ₂ η₂ …]`,
// `"rgb eta" [r g b]`, or `"spectrum eta" "named-spd"`. The parser strips
// the type qualifier, so `getFloat("eta")` would return λ₁ for the sampled
// form — that's how `small_ruby` and `diamond` ended up with ior=200.
static float resolveDielectricEta(const pbrt::ParamList &params, float def)
{
  auto extract = [&](const std::vector<float> &fv, float fallback) -> float {
    if (fv.empty())
      return fallback;
    if (fv.size() == 1)
      return fv[0];
    if (fv.size() == 3) {
      // `rgb eta` (per-channel constant) — average to a scalar.
      return (fv[0] + fv[1] + fv[2]) / 3.f;
    }
    if (fv.size() % 2 == 0) {
      // Sampled spectrum: (wavelength, value) pairs. Values are at odd
      // indices; their mean is a reasonable scalar approximation.
      float sum = 0.f;
      const size_t pairs = fv.size() / 2;
      for (size_t i = 0; i < pairs; ++i)
        sum += fv[2 * i + 1];
      return sum / float(pairs);
    }
    logWarning(
        "[import_PBRT] dielectric: unrecognized eta layout "
        "(%zu floats), using %.3f",
        fv.size(),
        fallback);
    return fallback;
  };

  auto it = params.values.find("eta");
  if (it == params.values.end())
    return def;

  float eta = def;
  if (auto *fv = std::get_if<std::vector<float>>(&it->second)) {
    eta = extract(*fv, def);
  } else if (auto *sv = std::get_if<std::vector<std::string>>(&it->second);
      sv && !sv->empty()) {
    logWarning(
        "[import_PBRT] dielectric: named spectrum '%s' for eta "
        "not supported, using %.3f",
        (*sv)[0].c_str(),
        def);
  }

  if (eta > kMaxDielectricIor) {
    logWarning(
        "[import_PBRT] dielectric: eta %.3f exceeds the physical "
        "gem range; capping to %.2f so the medium colour reads",
        eta,
        kMaxDielectricIor);
    eta = kMaxDielectricIor;
  }
  return eta;
}

// Read conductor roughness, handling isotropic and anisotropic cases.
static float conductorRoughness(const pbrt::ParamList &params)
{
  if (params.has("roughness"))
    return params.getFloat("roughness", 0.01f);
  if (params.has("uroughness")) {
    float u = params.getFloat("uroughness", 0.01f);
    float v = params.getFloat("vroughness", u);
    return std::sqrt(u * v);
  }
  return 0.01f;
}

// Map a PBRT homogeneous medium's absorption (rgb sigma_a, per scene unit)
// onto ANARI physicallyBased volume parameters. KHR_materials_volume says
// `attenuationColor` is the tint that white light becomes after travelling
// `attenuationDistance` through the medium, i.e.
//   T(d) = pow(attenuationColor, d / attenuationDistance) = exp(-sigma_a * d)
// PBRT dielectrics commonly author a high IOR and rely on total internal
// reflection to bounce a ray through the medium several times, accumulating
// the per-traversal sigma_a on each pass. This importer caps IOR at 2.5
// (see kMaxDielectricIor) and KHR_materials_volume models only a single
// transmission, so neither effect lengthens the in-medium path and the
// authored sigma_a reads too faint. Pre-amplify by setting
// attenuationDistance well below 1 to recover the visual punch.
constexpr float kPbrtMediumAbsorptionBoost = 5.f;

static void applyMediumToMaterial(
    MaterialRef &mat, const pbrt::MediumDef &medium)
{
  if (medium.type != "homogeneous") {
    logWarning("[import_PBRT] medium type '%s' not supported, ignoring",
        medium.type.c_str());
    return;
  }
  auto &sigmaA = medium.params.getFloats("sigma_a");
  if (sigmaA.size() < 3) {
    logWarning("[import_PBRT] medium: sigma_a missing or not rgb, ignoring");
    return;
  }
  auto &sigmaS = medium.params.getFloats("sigma_s");
  const bool hasScattering = sigmaS.size() >= 3
      && (sigmaS[0] > 0.f || sigmaS[1] > 0.f || sigmaS[2] > 0.f);
  if (hasScattering) {
    logWarning(
        "[import_PBRT] medium: sigma_s > 0 (scattering) approximated as "
        "extra absorption");
  }
  const float ax = sigmaA[0] + (hasScattering ? sigmaS[0] : 0.f);
  const float ay = sigmaA[1] + (hasScattering ? sigmaS[1] : 0.f);
  const float az = sigmaA[2] + (hasScattering ? sigmaS[2] : 0.f);
  const float3 attenuationColor(std::exp(-ax), std::exp(-ay), std::exp(-az));
  mat->setParameter("attenuationColor", ANARI_FLOAT32_VEC3, &attenuationColor);
  mat->setParameter("attenuationDistance", 1.f / kPbrtMediumAbsorptionBoost);
  mat->setParameter("thickness", 1.f);
}

using MaterialCacheKey = std::pair<std::string, std::string>;

static MaterialRef convertMaterial(Scene &scene,
    const pbrt::Scene &pbrtScene,
    const std::string &materialName,
    const std::string &interiorMedium,
    const std::string &basePath,
    TextureCache &texCache,
    std::map<MaterialCacheKey, MaterialRef> &matCache)
{
  if (materialName.empty())
    return scene.defaultMaterial();

  const MaterialCacheKey key{materialName, interiorMedium};
  auto it = matCache.find(key);
  if (it != matCache.end())
    return it->second;

  auto matIt = pbrtScene.namedMaterials.find(materialName);
  if (matIt == pbrtScene.namedMaterials.end()) {
    logWarning("[import_PBRT] material '%s' not found", materialName.c_str());
    auto mat = scene.defaultMaterial();
    matCache[key] = mat;
    return mat;
  }

  auto &matDef = matIt->second;
  const auto &type = matDef.type;
  const auto &params = matDef.params;

  MaterialRef mat;

  if (type == "diffuse") {
    mat = scene.createObject<Material>(tokens::material::matte);
    auto color = getRgb(params, "reflectance");
    mat->setParameter("color", ANARI_FLOAT32_VEC3, &color);
    resolveTexture(scene,
        mat,
        "color",
        "reflectance",
        matDef,
        pbrtScene,
        basePath,
        texCache);
  } else if (type == "coateddiffuse") {
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    auto baseColor = getRgb(params, "reflectance");
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("metallic", 0.f);
    mat->setParameter("roughness", 1.f);
    mat->setParameter("specular", 1.f);
    mat->setParameter("clearcoat", 1.f);
    // PBRT roughness is the coating interface roughness, not the diffuse base
    float coatRoughness;
    if (params.has("roughness"))
      coatRoughness = params.getFloat("roughness", 0.f);
    else if (params.has("uroughness"))
      coatRoughness = params.getFloat("uroughness", 0.f);
    else
      coatRoughness = 0.f;
    mat->setParameter("clearcoatRoughness", coatRoughness);
    resolveTexture(scene,
        mat,
        "baseColor",
        "reflectance",
        matDef,
        pbrtScene,
        basePath,
        texCache);
    resolveTexture(scene,
        mat,
        "clearcoatRoughness",
        "roughness",
        matDef,
        pbrtScene,
        basePath,
        texCache,
        ANARI_FLOAT32);
  } else if (type == "conductor") {
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    auto baseColor = getRgb(
        params, "reflectance", conductorSpectrumToRgb(params.getString("eta")));
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("metallic", 1.f);
    mat->setParameter("roughness", conductorRoughness(params));
    resolveTexture(scene,
        mat,
        "roughness",
        "roughness",
        matDef,
        pbrtScene,
        basePath,
        texCache,
        ANARI_FLOAT32);
  } else if (type == "dielectric" || type == "thindielectric") {
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    float3 baseColor(1.f);
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("ior", resolveDielectricEta(params, 1.5f));
    mat->setParameter("roughness", params.getFloat("roughness", 0.f));
    mat->setParameter("metallic", 0.f);
    mat->setParameter("specular", 1.f);
    mat->setParameter("transmission", 1.f);
    resolveTexture(scene,
        mat,
        "roughness",
        "roughness",
        matDef,
        pbrtScene,
        basePath,
        texCache,
        ANARI_FLOAT32);
  } else if (type == "diffusetransmission") {
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    auto baseColor = getRgb(params, "reflectance");
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("metallic", 0.f);
    mat->setParameter("specular", 1.f);
    mat->setParameter("roughness", 1.f);
    mat->setParameter("transmission", 1.f);
    resolveTexture(scene,
        mat,
        "baseColor",
        "reflectance",
        matDef,
        pbrtScene,
        basePath,
        texCache);
  } else if (type == "coatedconductor") {
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    auto baseColor = getRgb(params,
        "reflectance",
        conductorSpectrumToRgb(
            params.getString("eta", params.getString("conductor.eta"))));
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("metallic", 1.f);
    mat->setParameter("roughness",
        params.getFloat("conductor.roughness", conductorRoughness(params)));
    mat->setParameter("clearcoat", 1.f);
    mat->setParameter(
        "clearcoatRoughness", params.getFloat("interface.roughness", 0.f));
  } else if (type == "subsurface") {
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    auto baseColor = getRgb(params, "reflectance", float3(0.8f));
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("metallic", 0.f);
    mat->setParameter("specular", 1.f);
    mat->setParameter("roughness", params.getFloat("roughness", 0.5f));
  } else if (type == "measured") {
    // Measured BSDFs (`bsdfs/*.bsdf`) hold tabulated reflectance data that
    // ANARI's PBR model can't replay. Almost every measured BSDF in the
    // PBRT v4 scene library names a glossy white-ish material (porcelain,
    // satin, paint); approximate that as non-metallic white with a low
    // roughness so the surface reads correctly instead of falling into the
    // matte fallback (which used to render bright red).
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    float3 baseColor(1.f);
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("metallic", 0.f);
    mat->setParameter("specular", 1.f);
    mat->setParameter("roughness", 0.2f);
  } else if (type == "hair") {
    mat = scene.createObject<Material>(tokens::material::matte);
    auto color = getRgb(params, "color", float3(0.3f, 0.2f, 0.1f));
    mat->setParameter("color", ANARI_FLOAT32_VEC3, &color);
  } else if (type == "interface") {
    mat = scene.createObject<Material>(tokens::material::physicallyBased);
    float3 baseColor(1.f);
    mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    mat->setParameter("metallic", 0.f);
    mat->setParameter("specular", 1.f);
    mat->setParameter("transmission", 1.f);
  } else if (type == "mix") {
    auto matNamesIt = params.values.find("materials");
    const auto *names = matNamesIt != params.values.end()
        ? std::get_if<std::vector<std::string>>(&matNamesIt->second)
        : nullptr;
    if (!names || names->size() < 2) {
      logWarning("[import_PBRT] 'mix' material: missing sub-material list");
      auto fallback = scene.defaultMaterial();
      matCache[key] = fallback;
      return fallback;
    }
    {
      // PBRT v4 mix: result = (1-α)·mat0 + α·mat1. α is a float or a texture.
      // Texture-driven amount can be exactly represented for baseColor —
      // bake `c0 + (c1-c0)*mask` into the mask sampler via its outTransform
      // and outOffset. The other scalar parameters (metallic, roughness, …)
      // can't vary per-pixel in a single ANARI material; they keep the
      // amount=0.5 lerp as a uniform compromise.
      float amount = params.getFloat("amount", 0.5f);
      std::string maskTexName;
      auto amountValIt = params.values.find("amount");
      if (amountValIt != params.values.end()) {
        if (auto *sv =
                std::get_if<std::vector<std::string>>(&amountValIt->second);
            sv && !sv->empty()) {
          maskTexName = (*sv)[0];
          amount = 0.5f;
        }
      }
      const bool textureDriven = !maskTexName.empty();
      amount = std::clamp(amount, 0.f, 1.f);

      // Children resolve without the parent's medium: the medium is a
      // property of the shape boundary and is applied once to the mix
      // result below, not redundantly to each sub-material.
      auto mat0 = convertMaterial(
          scene, pbrtScene, (*names)[0], "", basePath, texCache, matCache);
      auto mat1 = convertMaterial(
          scene, pbrtScene, (*names)[1], "", basePath, texCache, matCache);

      if (!mat0 || !mat1) {
        logWarning(
            "[import_PBRT] 'mix' material: cannot resolve sub-materials");
        // Reuse a sub-material rather than constructing a new one; do not
        // rename or re-cache (would mutate a shared/cached material).
        auto fallback = mat0 ? mat0 : (mat1 ? mat1 : scene.defaultMaterial());
        matCache[key] = fallback;
        return fallback;
      }
      {
        // Read baseColor from either matte ("color") or PBR ("baseColor")
        auto readBaseColor = [](const MaterialRef &m) -> float3 {
          if (m->subtype() == tokens::material::matte) {
            auto v = m->parameterValueAs<float3>("color");
            return v.value_or(float3(0.8f));
          }
          auto v = m->parameterValueAs<float3>("baseColor");
          return v.value_or(float3(0.8f));
        };

        auto readFloat =
            [](const MaterialRef &m, const char *name, float def) -> float {
          auto v = m->parameterValueAs<float>(name);
          return v.value_or(def);
        };

        const float w0 = 1.f - amount;
        const float w1 = amount;
        auto lerpF = [w0, w1](float a, float b) { return a * w0 + b * w1; };
        auto lerpC = [w0, w1](float3 a, float3 b) {
          return float3(
              a.x * w0 + b.x * w1, a.y * w0 + b.y * w1, a.z * w0 + b.z * w1);
        };

        // Matte defaults: metallic=0, roughness=1, specular=0, clearcoat=0
        float3 bc = lerpC(readBaseColor(mat0), readBaseColor(mat1));
        float metallic = lerpF(
            readFloat(mat0, "metallic", 0.f), readFloat(mat1, "metallic", 0.f));
        float roughness = lerpF(readFloat(mat0, "roughness", 1.f),
            readFloat(mat1, "roughness", 1.f));
        float specular = lerpF(
            readFloat(mat0, "specular", 0.f), readFloat(mat1, "specular", 0.f));
        float clearcoat = lerpF(readFloat(mat0, "clearcoat", 0.f),
            readFloat(mat1, "clearcoat", 0.f));
        float ccRough = lerpF(readFloat(mat0, "clearcoatRoughness", 0.f),
            readFloat(mat1, "clearcoatRoughness", 0.f));
        float transmission = lerpF(readFloat(mat0, "transmission", 0.f),
            readFloat(mat1, "transmission", 0.f));
        float ior =
            lerpF(readFloat(mat0, "ior", 1.5f), readFloat(mat1, "ior", 1.5f));

        mat = scene.createObject<Material>(tokens::material::physicallyBased);
        mat->setParameter("baseColor", ANARI_FLOAT32_VEC3, &bc);
        mat->setParameter("metallic", metallic);
        mat->setParameter("roughness", roughness);
        mat->setParameter("specular", specular);
        mat->setParameter("clearcoat", clearcoat);
        mat->setParameter("clearcoatRoughness", ccRough);
        mat->setParameter("transmission", transmission);
        mat->setParameter("ior", ior);

        if (textureDriven) {
          // Bake `result = c0 + (c1 - c0) * mask` into the mask sampler.
          // BakedTexture's affine is `tint*S + offset` per channel, so the
          // composed affine is `(c1-c0)*tint*S + ((c1-c0)*offset + c0)`.
          auto baked =
              bakeTexture(scene, pbrtScene, maskTexName, basePath, texCache);
          if (baked.kind == BakedTexture::Kind::Sampler) {
            const float3 bc0 = readBaseColor(mat0);
            const float3 bc1 = readBaseColor(mat1);
            const float3 dc(bc1.x - bc0.x, bc1.y - bc0.y, bc1.z - bc0.z);
            const float3 newTint(
                dc.x * baked.tint.x, dc.y * baked.tint.y, dc.z * baked.tint.z);
            const float3 newOffset(dc.x * baked.offset.x + bc0.x,
                dc.y * baked.offset.y + bc0.y,
                dc.z * baked.offset.z + bc0.z);
            applyAffineToSampler(baked.sampler, newTint, newOffset);
            mat->setParameterObject("baseColor", *baked.sampler);
          } else {
            logWarning(
                "[import_PBRT] 'mix' material: mask texture '%s' did "
                "not resolve to an image — keeping constant baseColor",
                maskTexName.c_str());
          }
        } else {
          // Constant amount: pick the dominant side's texture if any.
          const auto &dominant = (amount < 0.5f) ? mat0 : mat1;
          const char *texParam =
              (dominant->subtype() == tokens::material::matte) ? "color"
                                                               : "baseColor";
          if (auto *obj = dominant->parameterValueAsObject(texParam))
            mat->setParameterObject("baseColor", *obj);
        }
      }
    }
  } else {
    logWarning(
        "[import_PBRT] unsupported material type '%s', using matte fallback",
        type.c_str());
    mat = scene.createObject<Material>(tokens::material::matte);
    // TSD's matte default color is (1,0,0) — bright red. That makes every
    // unsupported material light up like a bug report; overwrite with a
    // neutral grey so the scene reads correctly while flagged in the log.
    float3 fallbackColor(0.7f);
    mat->setParameter("color", ANARI_FLOAT32_VEC3, &fallbackColor);
  }

  // Normal map: PBRT uses "string normalmap" with a direct file path
  auto normalMapPath = params.getString("normalmap");
  if (!normalMapPath.empty()) {
    try {
      auto fullPath = pbrt::resolveScenePath(basePath, normalMapPath);
      if (auto sampler = importTexture(scene, fullPath, texCache, true))
        mat->setParameterObject("normal", *sampler);
    } catch (const std::exception &e) {
      logWarning("[import_PBRT] normalmap: %s", e.what());
    }
  } else {
    // Fallback: convert a PBRT `texture displacement` height map into a
    // bump-style tangent normal. Real PBRT does geometric displacement,
    // which ANARI cannot represent — this is a "looks-the-part" hack.
    auto dispIt = params.values.find("displacement");
    if (dispIt != params.values.end()) {
      if (auto *sv = std::get_if<std::vector<std::string>>(&dispIt->second);
          sv && !sv->empty()) {
        std::string heightPath;
        float heightScale = 1.f;
        if (resolveImagemapChain(
                pbrtScene, (*sv)[0], basePath, heightPath, heightScale)) {
          if (auto sampler = importHeightAsNormalMap(
                  scene, heightPath, heightScale, texCache))
            mat->setParameterObject("normal", *sampler);
        } else {
          logWarning(
              "[import_PBRT] displacement '%s' does not resolve to a "
              "single imagemap chain; ignoring",
              (*sv)[0].c_str());
        }
      }
    }
  }

  // Volume absorption from PBRT's MediumInterface (interior medium).
  // Only meaningful for materials that actually transmit light through the
  // shape volume. Mix is intentionally excluded — its children were resolved
  // without the medium and the surface boundary doesn't have a well-defined
  // interior here.
  if (!interiorMedium.empty()
      && (type == "dielectric" || type == "thindielectric"
          || type == "interface")) {
    auto medIt = pbrtScene.namedMedia.find(interiorMedium);
    if (medIt == pbrtScene.namedMedia.end()) {
      logWarning("[import_PBRT] medium '%s' not found", interiorMedium.c_str());
    } else {
      applyMediumToMaterial(mat, medIt->second);
    }
  }

  // TSD defaults alphaMode to "blend" (ANARI spec default is "opaque"); set
  // it explicitly so an RGBA baseColor sampler doesn't bleed its .w into
  // opacity and silently turn solid surfaces transparent. `applyShapeAlpha`
  // overrides to "mask" on shapes that actually carry an alpha texture.
  mat->setParameter("alphaMode", "opaque");

  mat->setName(materialName.c_str());
  matCache[key] = mat;
  return mat;
}

// Select the right channel of a multi-channel sampler into output.x via
// outTransform. PBRT v4's FloatImageTexture prefers alpha, then luminance,
// then red — leaf textures store the mask in .a of the same RGBA file used
// for color, so this lets the opacity sampler share the loaded data array
// with the base-color sampler instead of allocating a parallel copy.
//
// Per the ANARI spec (samplers): the fetched texel is completed to four
// components only when the source has fewer than 4 channels (the missing
// first three default to 0, the missing fourth defaults to 1); outTransform
// is then applied. A scalar parameter like `opacity` consumes .x of the
// final result. So RGBA sources carry their real alpha in .w (swizzle it
// into .x); RGB sources have .w padded to 1, so .w is useless as a mask
// and we use BT.709 luminance instead.
static void setAlphaChannelSwizzle(
    SamplerRef &sampler, anari::DataType imageElementType)
{
  mat4 m{float4(0.f), float4(0.f), float4(0.f), float4(0.f)};
  switch (imageElementType) {
  case ANARI_FLOAT32_VEC4:
  case ANARI_UFIXED8_VEC4:
  case ANARI_UFIXED8_RGBA_SRGB:
  case ANARI_UFIXED16_VEC4:
    m[3] = float4(1.f, 0.f, 0.f, 0.f); // texel.w -> out.x
    break;
  case ANARI_FLOAT32_VEC3:
  case ANARI_UFIXED8_VEC3:
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED16_VEC3:
    // ITU-R BT.709 luminance for sources without an alpha channel.
    m[0] = float4(0.2126f, 0.f, 0.f, 0.f);
    m[1] = float4(0.7152f, 0.f, 0.f, 0.f);
    m[2] = float4(0.0722f, 0.f, 0.f, 0.f);
    break;
  case ANARI_FLOAT32_VEC2:
  case ANARI_UFIXED8_VEC2:
  case ANARI_UFIXED16_VEC2:
    m[1] = float4(1.f, 0.f, 0.f, 0.f); // gray+alpha -> alpha into .x
    break;
  default:
    m[0] = float4(1.f, 0.f, 0.f, 0.f); // single-channel: pass-through
    break;
  }
  sampler->setParameter("outTransform", m);
}

// Mirror a sampler, forcing its output .w to 1 while preserving .xyz. Used
// when wiring an explicit opacity sampler on a material whose baseColor is
// also a sampler: per the spec
// (chapters/object_types/materials.txt, physicallyBased), when baseColor is
// a sampler and alphaMode is not opaque, the sampler's 4th component is
// multiplied into opacity. Without neutralising it, an RGBA leaf texture's
// alpha would be applied twice (once via baseColor.w, once via the opacity
// sampler we just attached). Zeroing the matrix's 4th row drops the texel's
// .w contribution, and outOffset.w = 1 pins the output to fully opaque.
static SamplerRef cloneSamplerOpaqueAlpha(Scene &scene, const Sampler &src)
{
  auto clone = scene.createObject<Sampler>(src.subtype());
  for (size_t i = 0; i < src.numParameters(); ++i) {
    auto name = src.parameterNameAt(i);
    auto &p = src.parameterAt(i);
    if (auto *obj = src.parameterValueAsObject(name))
      clone->setParameterObject(name, *obj);
    else
      clone->setParameter(name, p.value().type(), p.value().data());
  }
  mat4 M{float4(1.f, 0.f, 0.f, 0.f),
      float4(0.f, 1.f, 0.f, 0.f),
      float4(0.f, 0.f, 1.f, 0.f),
      float4(0.f, 0.f, 0.f, 1.f)};
  if (auto opt = src.parameterValueAs<mat4>("outTransform"))
    M = *opt;
  float4 O(0.f, 0.f, 0.f, 0.f);
  if (auto opt = src.parameterValueAs<float4>("outOffset"))
    O = *opt;
  M[0].w = 0.f;
  M[1].w = 0.f;
  M[2].w = 0.f;
  M[3].w = 0.f;
  O.w = 1.f;
  clone->setParameter("outTransform", M);
  clone->setParameter("outOffset", O);
  clone->setName((std::string(src.name()) + "_rgb1").c_str());
  return clone;
}

// Apply alpha cutout texture from shape params to the material.
// PBRT shapes can have "texture alpha" referencing a named float texture.
// Since the material may be shared, we clone it when adding alpha.
static MaterialRef applyShapeAlpha(Scene &scene,
    MaterialRef mat,
    const pbrt::Shape &shape,
    const pbrt::Scene &pbrtScene,
    const std::string &basePath,
    TextureCache &texCache)
{
  // PBRT v4: shape "alpha" can be a float (uniform cutoff/blend) or a
  // texture reference. The parser stores floats as a vector<float> and
  // texture references as vector<string>; pick the form actually present.
  auto alphaIt = shape.params.values.find("alpha");
  if (alphaIt == shape.params.values.end())
    return mat;

  SamplerRef sampler;
  float floatAlpha = 1.f;
  bool haveFloatAlpha = false;

  if (auto *fv = std::get_if<std::vector<float>>(&alphaIt->second)) {
    if (fv->empty() || (*fv)[0] >= 1.f)
      return mat;
    floatAlpha = std::clamp((*fv)[0], 0.f, 1.f);
    haveFloatAlpha = true;
  } else if (auto *sv = std::get_if<std::vector<std::string>>(&alphaIt->second);
      sv && !sv->empty()) {
    const auto &alphaTexName = (*sv)[0];
    auto texIt = pbrtScene.textures.find(alphaTexName);
    if (texIt == pbrtScene.textures.end()) {
      logWarning(
          "[import_PBRT] alpha texture '%s' not found", alphaTexName.c_str());
      return mat;
    }
    auto filename = texIt->second.params.getString("filename");
    if (filename.empty())
      return mat;
    std::string fullPath;
    try {
      fullPath = pbrt::resolveScenePath(basePath, filename);
    } catch (const std::exception &e) {
      logWarning("[import_PBRT] alpha texture '%s': %s",
          alphaTexName.c_str(),
          e.what());
      return mat;
    }
    sampler = importTexture(scene, fullPath, texCache, true);
    if (!sampler)
      return mat;
    // The standard importer wires all 4 channels straight through. The
    // opacity slot reads .x, but the leaf mask sits in .a — swizzle here so
    // the sampler returns the right channel without re-baking the image.
    // Query the sampler's bound image directly; the texture cache may be
    // keyed differently (e.g. by gamma) than this caller knows.
    if (auto *image = sampler->parameterValueAsObject<Array>("image"))
      setAlphaChannelSwizzle(sampler, image->elementType());
  } else {
    return mat;
  }

  // Clone the material so we don't modify the shared original.
  auto alphaMat = scene.createObject<Material>(mat->subtype());
  for (size_t i = 0; i < mat->numParameters(); ++i) {
    auto name = mat->parameterNameAt(i);
    auto &p = mat->parameterAt(i);
    if (auto *obj = mat->parameterValueAsObject(name))
      alphaMat->setParameterObject(name, *obj);
    else
      alphaMat->setParameter(name, p.value().type(), p.value().data());
  }
  if (sampler)
    alphaMat->setParameterObject("opacity", *sampler);
  else if (haveFloatAlpha)
    alphaMat->setParameter("opacity", floatAlpha);
  alphaMat->setParameter("alphaMode", "mask");
  // Spec: when baseColor is a sampler and alphaMode != opaque, baseColor.w
  // is multiplied into opacity. Our explicit opacity already carries the
  // PBRT alpha — neutralise baseColor.w to avoid applying alpha twice.
  if (auto *bc = alphaMat->parameterValueAsObject<Sampler>("baseColor")) {
    auto bcClone = cloneSamplerOpaqueAlpha(scene, *bc);
    alphaMat->setParameterObject("baseColor", *bcClone);
  }
  alphaMat->setName((std::string(mat->name()) + "_alpha").c_str());
  return alphaMat;
}

// Apply PBRT light scale and film exposure to a light color.
// PBRT v4 "float scale" is a uniform scalar (default 1).
// exposureScale compensates for PBRT film ISO sensitivity.
static float3 applyScale(
    const pbrt::ParamList &params, float3 color, float exposureScale = 1.f)
{
  float s = params.getFloat("scale", 1.f) * exposureScale;
  return float3(color.x * s, color.y * s, color.z * s);
}

// Tanner Helland's blackbody-temperature-to-sRGB approximation. Output is
// max-channel-normalized (the dominant channel is 1.0) — absolute intensity
// is restored separately via the PBRT `scale` parameter. Good enough for
// importer purposes; PBRT itself uses a full Planckian SPD integration.
static float3 blackbodyToRgb(float kelvin)
{
  const float t = std::clamp(kelvin, 1000.f, 40000.f) / 100.f;
  float r, g, b;
  if (t <= 66.f) {
    r = 1.f;
    g = 0.3900815787f * std::log(t) - 0.6318414438f;
    b = (t <= 19.f) ? 0.f : 0.5432067891f * std::log(t - 10.f) - 1.1962540891f;
  } else {
    r = 1.2929361861f * std::pow(t - 60.f, -0.1332047592f);
    g = 1.1298908609f * std::pow(t - 60.f, -0.0755148492f);
    b = 1.f;
  }
  return float3(std::clamp(r, 0.f, 1.f),
      std::clamp(g, 0.f, 1.f),
      std::clamp(b, 0.f, 1.f));
}

// Resolve a PBRT emission/radiance parameter (light L/I, area light L) to
// an RGB triple, honouring the type qualifier preserved by the parser:
//   "rgb"/"color"/"xyz"  -> first 3 floats
//   "blackbody"           -> 1 float = temperature in K
//   "spectrum" (pairs)    -> mean of sample values (grayscale fallback)
//   named spectrum string -> warn + fallback
// Without the qualifier we fall back to the legacy "first 3 floats or
// gray-broadcast first float" behaviour.
static float3 resolveEmissionColor(const pbrt::ParamList &params,
    const std::string &name,
    float3 fallback = float3(1.f))
{
  auto it = params.values.find(name);
  if (it == params.values.end())
    return fallback;
  const std::string type = params.getType(name);

  if (auto *fv = std::get_if<std::vector<float>>(&it->second)) {
    if (fv->empty())
      return fallback;
    if (type == "blackbody")
      return blackbodyToRgb((*fv)[0]);
    if (type == "spectrum" && fv->size() >= 2 && fv->size() % 2 == 0) {
      logWarning(
          "[import_PBRT] sampled spectrum '%s' approximated as "
          "grayscale (mean of sample values)",
          name.c_str());
      float sum = 0.f;
      const size_t pairs = fv->size() / 2;
      for (size_t i = 0; i < pairs; ++i)
        sum += (*fv)[2 * i + 1];
      return float3(sum / float(pairs));
    }
    if (fv->size() >= 3)
      return float3((*fv)[0], (*fv)[1], (*fv)[2]);
    return float3((*fv)[0]);
  }

  if (auto *sv = std::get_if<std::vector<std::string>>(&it->second);
      sv && !sv->empty()) {
    logWarning(
        "[import_PBRT] named spectrum '%s' for '%s' not supported, "
        "using fallback",
        (*sv)[0].c_str(),
        name.c_str());
  }
  return fallback;
}

static MaterialRef makeAreaEmissiveMaterial(Scene &scene,
    const pbrt::ParamList &areaLightParams,
    float exposureScale = 1.f)
{
  auto mat = scene.createObject<Material>(tokens::material::physicallyBased);
  mat->setParameter("baseColor", float3(1.f));
  float3 emissive = resolveEmissionColor(areaLightParams, "L");
  emissive = applyScale(areaLightParams, emissive, exposureScale);
  mat->setParameter("emissive", emissive);
  mat->setName("pbrt_area_emitter");
  return mat;
}

// Load and prepare an "infinite" HDRI as an equirectangular radiance array.
// PBRT v4 ships either equal-area (square) or equirectangular (2:1) HDRIs;
// the equal-area case is converted to equirectangular here so any ANARI device
// can render it.
static ArrayRef loadInfiniteRadiance(Scene &scene, const std::string &fullPath)
{
  HDRImage img;
  if (!img.import(fullPath)) {
    logWarning(
        "[import_PBRT] infinite light: failed to load '%s'", fullPath.c_str());
    return {};
  }

  std::vector<float3> in(size_t(img.width) * img.height);
  std::memcpy(in.data(), img.pixel.data(), sizeof(float3) * in.size());

  if (img.width == img.height) {
    // Equal-area -> equirectangular (2*W x W).
    std::vector<float3> out;
    int outW = 0, outH = 0;
    convertEqualAreaToEquirectangular(in, int(img.width), out, outW, outH);
    auto arr = scene.createArray(ANARI_FLOAT32_VEC3, outW, outH);
    arr->setData(out.data());
    return arr;
  }

  if (img.width != 2 * img.height) {
    logWarning(
        "[import_PBRT] infinite light: unexpected aspect %ux%u, "
        "treating as equirectangular",
        img.width,
        img.height);
  }

  auto arr = scene.createArray(ANARI_FLOAT32_VEC3, img.width, img.height);
  arr->setData(in.data());
  return arr;
}

static void convertLight(Scene &scene,
    const pbrt::LightDef &lightDef,
    LayerNodeRef parent,
    const std::string &basePath,
    TextureCache &texCache,
    float exposureScale = 1.f)
{
  (void)texCache;
  const auto &type = lightDef.type;
  const auto &params = lightDef.params;
  const auto xfm = pbrtTransformToMat4(lightDef.lightToWorld);

  // PBRT places light parameters (from/to) in the light-local frame; ANARI
  // light parameters are in world space. Pre-transform here so we don't depend
  // on the device applying parent transforms to direction/position params.

  if (type == "distant") {
    auto light = scene.createObject<Light>(tokens::light::directional);
    const float3 from =
        transformPoint(xfm, getFloat3(params, "from", float3(0.f, 0.f, 0.f)));
    const float3 to =
        transformPoint(xfm, getFloat3(params, "to", float3(0.f, 0.f, 1.f)));
    light->setParameter(
        "direction", safeNormalize(to - from, float3(0.f, 0.f, 1.f)));
    auto color =
        applyScale(params, resolveEmissionColor(params, "L"), exposureScale);
    light->setParameter("color", ANARI_FLOAT32_VEC3, &color);
    scene.insertChildObjectNode(parent, light);
  } else if (type == "point") {
    auto light = scene.createObject<Light>(tokens::light::point);
    const float3 pos =
        transformPoint(xfm, getFloat3(params, "from", float3(0.f)));
    light->setParameter("position", pos);
    auto color =
        applyScale(params, resolveEmissionColor(params, "I"), exposureScale);
    light->setParameter("color", ANARI_FLOAT32_VEC3, &color);
    scene.insertChildObjectNode(parent, light);
  } else if (type == "spot") {
    auto light = scene.createObject<Light>(tokens::light::spot);
    const float3 from =
        transformPoint(xfm, getFloat3(params, "from", float3(0.f, 0.f, 0.f)));
    const float3 to =
        transformPoint(xfm, getFloat3(params, "to", float3(0.f, 0.f, 1.f)));
    light->setParameter("position", from);
    light->setParameter(
        "direction", safeNormalize(to - from, float3(0.f, 0.f, 1.f)));
    auto color =
        applyScale(params, resolveEmissionColor(params, "I"), exposureScale);
    light->setParameter("color", ANARI_FLOAT32_VEC3, &color);
    const float openingAngle =
        params.getFloat("coneangle", 30.f) * 2.f * DEG_TO_RAD;
    light->setParameter("openingAngle", openingAngle);
    const float falloffAngle =
        params.getFloat("conedeltaangle", 5.f) * DEG_TO_RAD;
    light->setParameter("falloffAngle", falloffAngle);
    scene.insertChildObjectNode(parent, light);
  } else if (type == "infinite") {
    auto light = scene.createObject<Light>(tokens::light::hdri);
    const auto filename = params.getString("filename");
    if (!filename.empty()) {
      try {
        auto fullPath = pbrt::resolveScenePath(basePath, filename);
        if (auto arr = loadInfiniteRadiance(scene, fullPath))
          light->setParameterObject("radiance", *arr);
      } catch (const std::exception &e) {
        logWarning("[import_PBRT] infinite light: %s", e.what());
      }
    }

    auto color =
        applyScale(params, resolveEmissionColor(params, "L"), exposureScale);
    light->setParameter("color", ANARI_FLOAT32_VEC3, &color);
    light->setParameter("layout", "equirectangular");

    // PBRT's image-based infinite light uses +Z up in its local frame
    // (the equal-area mapping's center is +Z); after conversion to
    // equirectangular we keep that convention. lightToWorld's columns 0
    // and 2 give the world-space "direction" and "up" axes.
    const float3 dir = safeNormalize(
        float3(xfm[0][0], xfm[0][1], xfm[0][2]), float3(1.f, 0.f, 0.f));
    const float3 up = safeNormalize(
        float3(xfm[2][0], xfm[2][1], xfm[2][2]), float3(0.f, 0.f, 1.f));
    light->setParameter("direction", dir);
    light->setParameter("up", up);

    scene.insertChildObjectNode(parent, light);
  } else {
    logWarning("[import_PBRT] unsupported light type: %s", type.c_str());
  }
}

static void convertCamera(
    Scene &scene, const std::string &name, const pbrt::Scene &pbrtScene)
{
  const auto &camDef = pbrtScene.camera;
  const auto &type = camDef.type;
  const auto &params = camDef.params;
  auto m = pbrtTransformToMat4(camDef.cameraToWorld);

  float3 forward(m[2][0], m[2][1], m[2][2]);
  float3 up(m[1][0], m[1][1], m[1][2]);
  float3 eye(m[3][0], m[3][1], m[3][2]);

  // Pick the ANARI camera subtype to match PBRT's, with sensible fallbacks
  // for camera kinds ANARI doesn't model (spherical/realistic).
  auto cameraType = tokens::camera::perspective;
  if (type == "orthographic") {
    cameraType = tokens::camera::orthographic;
  } else if (type == "spherical") {
    logWarning("[import_PBRT] 'spherical' camera approximated as perspective");
  } else if (type == "realistic") {
    logWarning("[import_PBRT] 'realistic' camera approximated as perspective");
  } else if (type != "perspective") {
    logWarning("[import_PBRT] unsupported camera type '%s', using perspective",
        type.c_str());
  }

  auto cam = scene.createObject<Camera>(cameraType);
  cam->setName(name.c_str());

  cam->setParameter("position", eye);
  cam->setParameter("direction", forward);
  cam->setParameter("up", up);

  const int W = std::max(1, pbrtScene.film.xResolution);
  const int H = std::max(1, pbrtScene.film.yResolution);
  cam->setParameter("aspect", float(W) / float(H));

  if (cameraType == tokens::camera::perspective) {
    // PBRT v4 `fov` is along the *smaller* image dimension; ANARI `fovy` is
    // the vertical FoV. Convert when the image is portrait (H > W) so e.g.
    // crown.pbrt (1000x1400, fov=47) renders with the right field of view.
    float fovy;
    if (type == "spherical") {
      fovy = 90.f * DEG_TO_RAD;
    } else if (type == "realistic") {
      fovy = 50.f * DEG_TO_RAD;
    } else {
      const float pbrtFov = params.getFloat("fov", 90.f) * DEG_TO_RAD;
      fovy = (H > W)
          ? 2.f * std::atan(std::tan(pbrtFov * 0.5f) * float(H) / float(W))
          : pbrtFov;
    }
    cam->setParameter("fovy", fovy);
  }

  // Depth of field: PBRT's `lensradius` / `focaldistance` correspond directly
  // to ANARI's `apertureRadius` / `focusDistance`. Default lensradius is 0
  // (pinhole) so we only set when explicitly given.
  if (params.has("lensradius"))
    cam->setParameter("apertureRadius", params.getFloat("lensradius", 0.f));
  if (params.has("focaldistance"))
    cam->setParameter("focusDistance", params.getFloat("focaldistance", 1.f));

  // Motion-blur shutter window. Many devices ignore it; harmless when set.
  if (params.has("shutteropen") || params.has("shutterclose")) {
    float2 shutter(params.getFloat("shutteropen", 0.f),
        params.getFloat("shutterclose", 1.f));
    cam->setParameter("shutter", shutter);
  }
}

} // namespace

// Main importer ///////////////////////////////////////////////////////////////

void import_PBRT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filename,
    LayerNodeRef location)
{
  (void)animMgr;
  logStatus("[import_PBRT] Parsing '%s'...", filename);

  pbrt::Scene pbrtScene;
  try {
    pbrtScene = pbrt::Parser::parseFile(filename);
  } catch (const std::exception &e) {
    logError("[import_PBRT] Parse error: %s", e.what());
    return;
  }

  const std::string file = fileOf(filename);
  const std::string basePath = pathOf(filename);

  scene.beginLayerEditBatch();

  auto root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), file.c_str());

  TextureCache texCache;
  std::map<MaterialCacheKey, MaterialRef> matCache;

  // PBRT film ISO controls sensor exposure — scale lights to compensate
  float filmIso = pbrtScene.film.params.getFloat("iso", 100.f);
  float exposureScale = filmIso / 100.f;

  // PBRT v4 syntax is `MediumInterface "interior" "exterior"`. Some scenes
  // (crown's rubies/sapphires, dambreak's water) describe the gem/liquid as
  // the *exterior* medium because their meshes are inside-out (obj2pbrt
  // output, or ReverseOrientation). Picking whichever side is non-empty
  // matches the scenes' intent and keeps disney-cloud's interior-only case
  // unchanged.
  auto effectiveMedium = [](const pbrt::Shape &shape) {
    return shape.interiorMedium.empty() ? shape.exteriorMedium
                                        : shape.interiorMedium;
  };

  auto resolveShapeMaterial = [&](const pbrt::Shape &shape) {
    MaterialRef mat = !shape.areaLightType.empty()
        ? makeAreaEmissiveMaterial(scene, shape.areaLightParams, exposureScale)
        : convertMaterial(scene,
              pbrtScene,
              shape.materialName,
              effectiveMedium(shape),
              basePath,
              texCache,
              matCache);
    return applyShapeAlpha(scene, mat, shape, pbrtScene, basePath, texCache);
  };

  // Shapes
  for (auto &shape : pbrtScene.shapes)
    convertShape(scene, shape, resolveShapeMaterial(shape), root, basePath);

  // ObjectInstance: build each ObjectDef's surface list lazily, then share
  // the resulting Surface refs across every instance. Without sharing,
  // scenes like 'crown' allocate fresh Geometry per instance.
  struct InstanceTemplate
  {
    std::vector<std::pair<SurfaceRef, mat4>> surfaces;
  };
  std::map<std::string, InstanceTemplate> instTemplates;
  auto getInstanceTemplate =
      [&](const pbrt::ObjectDef &obj) -> const InstanceTemplate & {
    auto [it, inserted] = instTemplates.try_emplace(obj.name);
    if (!inserted)
      return it->second;
    auto &tmpl = it->second;
    for (auto &shape : obj.shapes) {
      auto geom = buildShapeGeometry(scene, shape, basePath);
      if (!geom)
        continue;
      auto mat = resolveShapeMaterial(shape);
      auto surface = makeShapeSurface(scene, shape, geom, mat);
      tmpl.surfaces.emplace_back(
          surface, pbrtTransformToMat4(shape.objectToWorld));
    }
    return tmpl;
  };

  for (auto &inst : pbrtScene.instances) {
    auto it = pbrtScene.objects.find(inst.name);
    if (it == pbrtScene.objects.end()) {
      logWarning(
          "[import_PBRT] unknown object instance: %s", inst.name.c_str());
      continue;
    }
    auto xfmNode = scene.insertChildTransformNode(
        root, pbrtTransformToMat4(inst.instanceToWorld), inst.name.c_str());
    const auto &tmpl = getInstanceTemplate(it->second);
    for (auto &[surface, localXfm] : tmpl.surfaces) {
      auto subXfm = scene.insertChildTransformNode(xfmNode, localXfm);
      scene.insertChildObjectNode(subXfm, surface);
    }
    for (auto &light : it->second.lights)
      convertLight(scene, light, xfmNode, basePath, texCache, exposureScale);
  }

  // Lights
  for (auto &light : pbrtScene.lights)
    convertLight(scene, light, root, basePath, texCache, exposureScale);

  // Camera
  convertCamera(scene, file, pbrtScene);

  scene.endLayerEditBatch();

  logStatus("[import_PBRT] Done: %zu shapes, %zu lights, %zu instances",
      pbrtScene.shapes.size(),
      pbrtScene.lights.size(),
      pbrtScene.instances.size());
}

} // namespace tsd::io
