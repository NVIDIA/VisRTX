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

#include "ComputeTangent.h"

#include <cuda_runtime.h>

#include <glm/common.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/geometric.hpp>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace {

constexpr const auto eps = 1e-8f;

__device__ glm::vec3 safeNormalize(
    const glm::vec3 &v, const glm::vec3 &fallback)
{
  const float l2 = glm::dot(v, v);
  return l2 > eps ? v * rsqrtf(l2) : fallback;
}

__device__ void makeTangentFrame(
    const glm::vec3 &normal, glm::vec3 *tangent, glm::vec3 *bitangent)
{
  // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
  const glm::vec3 n = safeNormalize(normal, glm::vec3(0.f, 0.f, 1.f));
  const float sign = n.z >= 0.0f ? 1.0f : -1.0f;
  const float a = -1.0f / (sign + n.z);
  const float b = n.x * n.y * a;
  *tangent = glm::vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
  *bitangent = glm::vec3(b, sign + n.y * n.y * a, -n.y);
}

__device__ glm::vec3 computeGeometricNormal(
    const glm::vec3 &e1, const glm::vec3 &e2)
{
  return safeNormalize(glm::cross(e1, e2), glm::vec3(0.f, 0.f, 1.f));
}

__device__ float cornerAngle(
    const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c)
{
  const glm::vec3 ab = b - a;
  const glm::vec3 ac = c - a;
  const float lab = sqrtf(glm::dot(ab, ab));
  const float lac = sqrtf(glm::dot(ac, ac));
  if (lab < eps || lac < eps)
    return 0.0f;
  const float cosT = glm::clamp(glm::dot(ab, ac) / (lab * lac), -1.0f, 1.0f);
  return acosf(cosT);
}

__device__ bool sameBits(const glm::vec2 &a, const glm::vec2 &b)
{
  return __float_as_uint(a.x) == __float_as_uint(b.x)
      && __float_as_uint(a.y) == __float_as_uint(b.y);
}

__device__ bool sameBits(const glm::vec3 &a, const glm::vec3 &b)
{
  return __float_as_uint(a.x) == __float_as_uint(b.x)
      && __float_as_uint(a.y) == __float_as_uint(b.y)
      && __float_as_uint(a.z) == __float_as_uint(b.z);
}

bool reportCudaError(
    visrtx::Geometry *geometry, cudaError_t error, const char *operation)
{
  if (error == cudaSuccess)
    return false;

  geometry->reportMessage(ANARI_SEVERITY_ERROR,
      "CUDA error while computing tangents for geometry %p during %s: %s",
      geometry,
      operation,
      cudaGetErrorString(error));
  return true;
}

// The mesh as the kernels see it. Corner c is corner c % 3 of triangle c / 3.
template <typename TexCoord>
struct MeshView
{
  const glm::uvec3 *indices; // null = triangle soup
  const glm::vec3 *positions;
  const glm::vec3 *normals; // null = no normals
  bool normalsFV;
  const TexCoord *uvs;
  bool uvsFV;

  __device__ glm::uvec3 triangle(uint32_t t) const
  {
    return indices ? indices[t] : glm::uvec3(3 * t) + glm::uvec3(0, 1, 2);
  }

  __device__ uint32_t vertex(uint32_t c) const
  {
    return triangle(c / 3)[c % 3];
  }

  __device__ glm::vec2 uv(uint32_t c) const
  {
    const TexCoord &st = uvs[uvsFV ? c : vertex(c)];
    return glm::vec2(st.x, st.y);
  }

  __device__ glm::vec3 normal(uint32_t c) const
  {
    return normals[normalsFV ? c : vertex(c)];
  }

  // The corner's angle, which weights its face's contribution to the frame.
  __device__ float weight(uint32_t c) const
  {
    const glm::uvec3 idx = triangle(c / 3);
    const uint32_t k = c % 3;
    return cornerAngle(positions[idx[k]],
        positions[idx[(k + 1) % 3]],
        positions[idx[(k + 2) % 3]]);
  }
};

// Pass 1 (one thread per triangle): the face's +dP/du and +dP/dv, and a count
// of each vertex's corners.
template <typename TexCoord>
__global__ void computeFaceFrames(MeshView<TexCoord> mesh,
    uint32_t numTriangles,
    glm::vec3 *faceT,
    glm::vec3 *faceB,
    uint32_t *cornerCounts)
{
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= numTriangles)
    return;

  const glm::uvec3 idx = mesh.triangle(t);
  for (int k = 0; k < 3; k++)
    atomicAdd(&cornerCounts[idx[k]], 1u);

  const glm::vec3 e1 = mesh.positions[idx.y] - mesh.positions[idx.x];
  const glm::vec3 e2 = mesh.positions[idx.z] - mesh.positions[idx.x];
  const glm::vec2 s = mesh.uv(3 * t + 1) - mesh.uv(3 * t);
  const glm::vec2 r = mesh.uv(3 * t + 2) - mesh.uv(3 * t);
  const float det = s.x * r.y - s.y * r.x;

  glm::vec3 T, B;
  if (glm::dot(e1, e1) < eps || glm::dot(e2, e2) < eps || glm::abs(det) < eps) {
    makeTangentFrame(computeGeometricNormal(e1, e2), &T, &B);
  } else {
    // Bitangent along +dP/dv, matching halcyon (ADR 0010).
    const float invdet = 1.0f / det;
    T = (r.y * e1 - s.y * e2) * invdet;
    B = (s.x * e2 - r.x * e1) * invdet;
  }
  faceT[t] = T;
  faceB[t] = B;
}

// Pass 2 (one thread per corner): bucket corners by vertex.
template <typename TexCoord>
__global__ void bucketCorners(MeshView<TexCoord> mesh,
    uint32_t numCorners,
    uint32_t *cursors,
    uint32_t *corners)
{
  const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= numCorners)
    return;
  corners[atomicAdd(&cursors[mesh.vertex(c)], 1u)] = c;
}

// Pass 3 (one thread per vertex): each corner sums the angle-weighted face
// frames of the vertex's corners that share its normal and texture coordinate,
// then writes vec4(T orthogonalized against its normal, handedness).
template <typename TexCoord>
__global__ void finalizeTangents(MeshView<TexCoord> mesh,
    uint32_t numVertices,
    const uint32_t *offsets,
    uint32_t *corners,
    const glm::vec3 *faceT,
    const glm::vec3 *faceB,
    bool perCorner,
    glm::vec4 *tangents)
{
  const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numVertices)
    return;

  const uint32_t begin = offsets[v];
  const uint32_t end = offsets[v + 1];

  if (begin == end) {
    if (!perCorner)
      tangents[v] = glm::vec4(1.f, 0.f, 0.f, 1.f);
    return;
  }

  // Corners were bucketed in arbitrary order; sort them so the sums below
  // don't depend on scheduling.
  for (uint32_t i = begin + 1; i < end; i++) {
    const uint32_t c = corners[i];
    uint32_t j = i;
    for (; j > begin && corners[j - 1] > c; j--)
      corners[j] = corners[j - 1];
    corners[j] = c;
  }

  for (uint32_t i = begin; i < end; i++) {
    const uint32_t c = corners[i];
    const glm::vec2 uv = mesh.uv(c);
    const glm::vec3 n = mesh.normals ? mesh.normal(c) : glm::vec3(0.f);

    glm::vec3 T(0.f), B(0.f), N(0.f);
    for (uint32_t j = begin; j < end; j++) {
      const uint32_t o = corners[j];
      if (!sameBits(mesh.uv(o), uv)
          || (mesh.normals && !sameBits(mesh.normal(o), n)))
        continue;
      const float w = mesh.weight(o);
      T += faceT[o / 3] * w;
      B += faceB[o / 3] * w;
      if (!mesh.normals) {
        const glm::uvec3 idx = mesh.triangle(o / 3);
        N += computeGeometricNormal(
                 mesh.positions[idx.y] - mesh.positions[idx.x],
                 mesh.positions[idx.z] - mesh.positions[idx.x])
            * w;
      }
    }

    const glm::vec3 normal =
        safeNormalize(mesh.normals ? n : N, glm::vec3(0.f, 0.f, 1.f));
    glm::vec3 fallbackT, fallbackB;
    makeTangentFrame(normal, &fallbackT, &fallbackB);
    const glm::vec3 Torth =
        safeNormalize(T - normal * glm::dot(normal, T), fallbackT);
    const float sign =
        glm::dot(glm::cross(normal, Torth), B) < 0.f ? -1.f : 1.f;

    if (!perCorner) {
      // Without face-varying data all of a vertex's corners match.
      tangents[v] = glm::vec4(Torth, sign);
      break;
    }
    tangents[c] = glm::vec4(Torth, sign);
  }
}

__global__ void padTangentsVec3ToVec4(
    glm::vec4 *dst, const glm::vec3 *src, unsigned int count)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  // No authored handedness in a VEC3 tangent; default the sign to +1.
  dst[i] = glm::vec4(src[i], 1.0f);
}

bool isTexCoordType(ANARIDataType type)
{
  return type == ANARI_FLOAT32_VEC2 || type == ANARI_FLOAT32_VEC3;
}

template <typename TexCoord>
bool runGenerator(visrtx::Geometry *geometry,
    const MeshView<TexCoord> &mesh,
    uint32_t numVertices,
    uint32_t numTriangles,
    bool perCorner,
    glm::vec4 *dst)
{
  const uint32_t numCorners = 3 * numTriangles;

  glm::vec3 *faceT = nullptr;
  glm::vec3 *faceB = nullptr;
  uint32_t *offsets = nullptr; // numVertices + 1
  uint32_t *cursors = nullptr; // numVertices
  uint32_t *corners = nullptr; // numCorners

  auto cleanup = [&] {
    cudaFree(faceT);
    cudaFree(faceB);
    cudaFree(offsets);
    cudaFree(cursors);
    cudaFree(corners);
  };
  auto failed = [&](cudaError_t status, const char *operation) {
    if (!reportCudaError(geometry, status, operation))
      return false;
    cleanup();
    return true;
  };

  if (failed(cudaMalloc(&faceT, sizeof(glm::vec3) * numTriangles),
          "allocating face tangents")
      || failed(cudaMalloc(&faceB, sizeof(glm::vec3) * numTriangles),
          "allocating face bitangents")
      || failed(cudaMalloc(&offsets, sizeof(uint32_t) * (numVertices + 1)),
          "allocating corner offsets")
      || failed(cudaMalloc(&cursors, sizeof(uint32_t) * numVertices),
          "allocating corner cursors")
      || failed(cudaMalloc(&corners, sizeof(uint32_t) * numCorners),
          "allocating corner lists")
      || failed(cudaMemset(offsets, 0, sizeof(uint32_t) * (numVertices + 1)),
          "clearing corner counts"))
    return false;

  computeFaceFrames<<<(numTriangles + 63) / 64, 64>>>(
      mesh, numTriangles, faceT, faceB, offsets);
  if (failed(cudaGetLastError(), "launching face frame kernel"))
    return false;

  // Counts (with a trailing zero) become each vertex's corner range.
  thrust::exclusive_scan(
      thrust::device, offsets, offsets + numVertices + 1, offsets);
  if (failed(cudaMemcpy(cursors,
                 offsets,
                 sizeof(uint32_t) * numVertices,
                 cudaMemcpyDeviceToDevice),
          "copying corner offsets"))
    return false;

  bucketCorners<<<(numCorners + 63) / 64, 64>>>(
      mesh, numCorners, cursors, corners);
  if (failed(cudaGetLastError(), "launching corner bucketing kernel"))
    return false;

  finalizeTangents<<<(numVertices + 63) / 64, 64>>>(
      mesh, numVertices, offsets, corners, faceT, faceB, perCorner, dst);
  if (failed(cudaGetLastError(), "launching finalize kernel")
      || failed(cudaDeviceSynchronize(), "computing tangents"))
    return false;

  cleanup();
  return true;
}

} // namespace

namespace visrtx {

TangentLayout generateTangents(Geometry *geometry,
    const TangentGenerationInput &in,
    DeviceBuffer &perVertex,
    DeviceBuffer &perCornerOut)
{
  perVertex.reset();
  perCornerOut.reset();

  if (!in.positions || in.positions->size() == 0 || in.numTriangles == 0)
    return TangentLayout::NONE;

  const size_t numVertices = in.positions->size();
  const size_t numCorners = 3 * in.numTriangles;

  // Face-varying arrays of the wrong size are reported at finalize; skip them.
  const Array1D *uvsFV =
      in.uvsFV && in.uvsFV->size() >= numCorners ? in.uvsFV : nullptr;
  const Array1D *uvs =
      in.uvs && in.uvs->size() >= numVertices ? in.uvs : nullptr;
  const Array1D *uvArray = uvsFV ? uvsFV : uvs;
  if (!uvArray) {
    geometry->reportMessage(ANARI_SEVERITY_INFO,
        "geometry %p has no usable attribute0, cannot generate tangents",
        geometry);
    return TangentLayout::NONE;
  }
  if (!isTexCoordType(uvArray->elementType())) {
    geometry->reportMessage(ANARI_SEVERITY_INFO,
        "can only generate tangents from attribute0 of type "
        "ANARI_FLOAT32_VEC2 or ANARI_FLOAT32_VEC3, not '%s'",
        anari::toString(uvArray->elementType()));
    return TangentLayout::NONE;
  }

  const Array1D *normalsFV = in.normalsFV && in.normalsFV->size() >= numCorners
      ? in.normalsFV
      : nullptr;
  const Array1D *normals =
      in.normals && in.normals->size() >= numVertices ? in.normals : nullptr;
  const Array1D *normalArray = normalsFV ? normalsFV : normals;

  // Without indices every corner has its own vertex, so per-vertex output is
  // already per corner.
  const bool perCorner = in.indices && (normalsFV || uvsFV);
  DeviceBuffer &dst = perCorner ? perCornerOut : perVertex;
  const size_t count = perCorner ? numCorners : numVertices;
  dst.reserve(count * sizeof(glm::vec4));
  if (!dst)
    return TangentLayout::NONE;

  const auto *positions = in.positions->beginAs<glm::vec3>(AddressSpace::GPU);
  const auto *normalsPtr = normalArray
      ? normalArray->beginAs<glm::vec3>(AddressSpace::GPU)
      : nullptr;

  bool ok = false;
  if (uvArray->elementType() == ANARI_FLOAT32_VEC2) {
    const MeshView<glm::vec2> mesh{in.indices,
        positions,
        normalsPtr,
        normalsFV != nullptr,
        uvArray->beginAs<glm::vec2>(AddressSpace::GPU),
        uvsFV != nullptr};
    ok = runGenerator(geometry,
        mesh,
        uint32_t(numVertices),
        uint32_t(in.numTriangles),
        perCorner,
        dst.ptrAs<glm::vec4>());
  } else {
    const MeshView<glm::vec3> mesh{in.indices,
        positions,
        normalsPtr,
        normalsFV != nullptr,
        uvArray->beginAs<glm::vec3>(AddressSpace::GPU),
        uvsFV != nullptr};
    ok = runGenerator(geometry,
        mesh,
        uint32_t(numVertices),
        uint32_t(in.numTriangles),
        perCorner,
        dst.ptrAs<glm::vec4>());
  }

  if (!ok) {
    dst.reset();
    return TangentLayout::NONE;
  }
  return perCorner ? TangentLayout::PER_CORNER : TangentLayout::PER_VERTEX;
}

bool prepareTangentArray(Geometry *geometry,
    const helium::IntrusivePtr<Array1D> &tangents,
    DeviceBuffer &converted,
    const char *paramName)
{
  // Only the staging buffer is rebuilt here; the array member is left untouched
  // so it keeps faithfully reflecting the committed parameter.
  converted.reset();

  if (!tangents)
    return false;

  const auto type = tangents->elementType();

  // Already in the internal layout: read zero-copy in gpuData().
  if (type == ANARI_FLOAT32_VEC4)
    return true;

  // Spec-allowed VEC3 tangents: pad to vec4 with a default +1 handedness.
  if (type == ANARI_FLOAT32_VEC3) {
    const auto count = tangents->size();
    if (count == 0)
      return false;
    converted.reserve(count * sizeof(glm::vec4));
    // On allocation (empty buffer) or conversion failure, leave 'converted'
    // empty; gpuData() then emits no tangents (resolveTangentPtr zero-copies
    // VEC4 only) rather than reading an uninitialized buffer.
    if (!converted)
      return false;
    const auto n = static_cast<unsigned int>(count);
    padTangentsVec3ToVec4<<<(n + 63) / 64, 64>>>(converted.ptrAs<glm::vec4>(),
        tangents->beginAs<glm::vec3>(AddressSpace::GPU),
        n);
    if (reportCudaError(geometry,
            cudaGetLastError(),
            "launching tangent vec3->vec4 padding")
        || reportCudaError(geometry,
            cudaDeviceSynchronize(),
            "padding vec3 tangents to vec4")) {
      converted.reset();
      return false;
    }
    return true;
  }

  // Anything else (e.g. the FIXED16 variants) is advertised by the query
  // metadata but not yet handled here. Report it; gpuData() emits no tangents
  // rather than throwing on a VEC4 read of a non-VEC4 array.
  geometry->reportMessage(ANARI_SEVERITY_WARNING,
      "'%s' has unsupported element type '%s'; expected ANARI_FLOAT32_VEC3 or "
      "ANARI_FLOAT32_VEC4 -- ignoring tangents",
      paramName,
      anari::toString(type));

  return false;
}

const glm::vec4 *resolveTangentPtr(
    const helium::IntrusivePtr<Array1D> &tangents,
    const DeviceBuffer &converted)
{
  if (converted)
    return converted.ptrAs<glm::vec4>();
  if (tangents && tangents->size() > 0
      && tangents->elementType() == ANARI_FLOAT32_VEC4)
    return tangents->beginAs<glm::vec4>(AddressSpace::GPU);
  return nullptr;
}

} // namespace visrtx
