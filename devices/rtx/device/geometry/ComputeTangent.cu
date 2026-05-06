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
#include "array/Array.h"
#include "geometry/Triangle.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
#include <vector_types.h>

#include <glm/common.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/geometric.hpp>

#include <cstdio>
#include <glm/vector_relational.hpp>

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

// Each face-vertex's contribution to its vertex's accumulated tangent frame
// is weighted by the triangle's interior angle at that corner — same scheme
// MikkTSpace uses to average across incident faces. Angle weighting (over
// uniform or area) keeps thin sliver triangles from dominating shared
// vertices.
__device__ float cornerAngle(
    const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c)
{
  const glm::vec3 ab = b - a;
  const glm::vec3 ac = c - a;
  const float lab = sqrtf(glm::dot(ab, ab));
  const float lac = sqrtf(glm::dot(ac, ac));
  if (lab < eps || lac < eps)
    return 0.0f;
  const float cosT =
      glm::clamp(glm::dot(ab, ac) / (lab * lac), -1.0f, 1.0f);
  return acosf(cosT);
}

__device__ void atomicAddVec3(glm::vec3 &dst, const glm::vec3 &v)
{
  atomicAdd(&dst.x, v.x);
  atomicAdd(&dst.y, v.y);
  atomicAdd(&dst.z, v.z);
}

void cudaFreeMemoryDeleter(const void *, const void *memory)
{
  cudaFree(const_cast<void *>(memory));
}

bool reportCudaError(
    visrtx::Triangle *triangle, cudaError_t error, const char *operation)
{
  if (error == cudaSuccess)
    return false;

  triangle->reportMessage(ANARI_SEVERITY_ERROR,
      "CUDA error while computing tangents for Triangle %p during %s: %s",
      triangle,
      operation,
      cudaGetErrorString(error));
  return true;
}

} // namespace

namespace visrtx {

__device__ void __computeTangentAndBitangent(
    glm::vec3 *tangent, // Output tangent vectors with handedness (w component)
    glm::vec3 *bitangent, // Output bitangent vectors
    glm::vec3 p0, // Input vertex positions
    glm::vec3 p1,
    glm::vec3 p2,
    glm::vec2 uv0, // Input texture coordinates
    glm::vec2 uv1,
    glm::vec2 uv2)
{
  // Compute edges of the triangle
  glm::vec3 e1 = p1 - p0;
  glm::vec3 e2 = p2 - p0;
  const auto normal = computeGeometricNormal(e1, e2);

  if (glm::dot(e1, e1) < eps || glm::dot(e2, e2) < eps) {
    makeTangentFrame(normal, tangent, bitangent);
    return;
  }

  // Compute differences in texture coordinates
  auto s = uv1 - uv0;
  auto t = uv2 - uv0;

  auto det = s.x * t.y - s.y * t.x;

  if (glm::abs(det) < eps) {
    makeTangentFrame(normal, tangent, bitangent);
    return;
  }

  float invdet = 1.0f / det;
  *tangent = (t.y * e1 - s.y * e2) * invdet;
  *bitangent = (s.x * e2 - t.x * e1) * invdet;
}

// Pass 1 (one thread per triangle): compute the per-triangle T/B from the UV
// gradient, then atomicAdd those vectors into per-vertex accumulators —
// weighted by the triangle's interior angle at each corner. Per-vertex
// normals are accumulated the same way so Pass 2 has a coordinate frame to
// orthogonalize against, regardless of whether input normals are vertex,
// face-varying, or absent.
template <bool VerticesIndexed,
    bool NormalsIndexed,
    bool UVsIndexed,
    typename TexCoord>
__global__ void __doAccumulateTangents(
    glm::vec3 *tangentAccum,
    glm::vec3 *bitangentAccum,
    glm::vec3 *normalAccum,
    const glm::uvec3 *indices,
    const glm::vec3 *positions,
    const glm::vec3 *normals,
    const TexCoord *uvs,
    unsigned int numTriangles)
{
  unsigned int tri = blockIdx.x * blockDim.x + threadIdx.x;

  if (tri >= numTriangles)
    return;

  auto perFaceBaseIdx = tri * 3 + glm::uvec3(0, 1, 2);
  auto indexedIdx = (VerticesIndexed || NormalsIndexed || UVsIndexed)
      ? indices[tri]
      : glm::uvec3(0);

  vec3 p0, p1, p2;
  if constexpr (VerticesIndexed) {
    p0 = positions[indexedIdx.x];
    p1 = positions[indexedIdx.y];
    p2 = positions[indexedIdx.z];
  } else {
    p0 = positions[perFaceBaseIdx.x];
    p1 = positions[perFaceBaseIdx.y];
    p2 = positions[perFaceBaseIdx.z];
  }

  vec2 uv0, uv1, uv2;
  if constexpr (UVsIndexed) {
    uv0 = uvs[indexedIdx.x];
    uv1 = uvs[indexedIdx.y];
    uv2 = uvs[indexedIdx.z];
  } else {
    uv0 = uvs[perFaceBaseIdx.x];
    uv1 = uvs[perFaceBaseIdx.y];
    uv2 = uvs[perFaceBaseIdx.z];
  }

  vec3 tangent, bitangent;
  __computeTangentAndBitangent(
      &tangent, &bitangent, p0, p1, p2, uv0, uv1, uv2);

  const vec3 geometricNormal = computeGeometricNormal(p1 - p0, p2 - p0);
  vec3 n0 = geometricNormal;
  vec3 n1 = geometricNormal;
  vec3 n2 = geometricNormal;
  if (normals) {
    if constexpr (NormalsIndexed) {
      n0 = normals[indexedIdx.x];
      n1 = normals[indexedIdx.y];
      n2 = normals[indexedIdx.z];
    } else {
      n0 = normals[perFaceBaseIdx.x];
      n1 = normals[perFaceBaseIdx.y];
      n2 = normals[perFaceBaseIdx.z];
    }
    n0 = safeNormalize(n0, geometricNormal);
    n1 = safeNormalize(n1, geometricNormal);
    n2 = safeNormalize(n2, geometricNormal);
  }

  // For indexed meshes, accumulate at the shared vertex slot so adjacent
  // triangles average their contributions. For triangle-soup each face-vertex
  // already has a unique slot.
  const glm::uvec3 outIdx = VerticesIndexed ? indexedIdx : perFaceBaseIdx;

  const float w0 = cornerAngle(p0, p1, p2);
  const float w1 = cornerAngle(p1, p0, p2);
  const float w2 = cornerAngle(p2, p0, p1);

  atomicAddVec3(tangentAccum[outIdx.x], tangent * w0);
  atomicAddVec3(tangentAccum[outIdx.y], tangent * w1);
  atomicAddVec3(tangentAccum[outIdx.z], tangent * w2);

  atomicAddVec3(bitangentAccum[outIdx.x], bitangent * w0);
  atomicAddVec3(bitangentAccum[outIdx.y], bitangent * w1);
  atomicAddVec3(bitangentAccum[outIdx.z], bitangent * w2);

  atomicAddVec3(normalAccum[outIdx.x], n0 * w0);
  atomicAddVec3(normalAccum[outIdx.y], n1 * w1);
  atomicAddVec3(normalAccum[outIdx.z], n2 * w2);
}

template <bool VerticesIndexed,
    bool NormalsIndexed,
    bool UVsIndexed,
    typename TexCoord>
void __computeTangents(
    glm::vec3 *tangentAccum,
    glm::vec3 *bitangentAccum,
    glm::vec3 *normalAccum,
    const glm::uvec3 *indices,
    const glm::vec3 *positions,
    const glm::vec3 *normals,
    const TexCoord *uvs,
    unsigned int numTriangles)
{
  __doAccumulateTangents<VerticesIndexed,
      NormalsIndexed,
      UVsIndexed,
      TexCoord><<<(numTriangles + 63) / 64, 64>>>(tangentAccum,
      bitangentAccum,
      normalAccum,
      indices,
      positions,
      normals,
      uvs,
      numTriangles);
}

// Pass 2 (one thread per vertex): normalize the accumulated frame and write
// vec4(T_orthog, sign). The accumulated normal is used as the orthogonalization
// basis; for vertex-indexed input it averages back to each vertex's authored
// normal, and for face-varying or missing normals it gives the angle-weighted
// average across incident faces.
__global__ void __doFinalizeTangents(glm::vec4 *tangents,
    const glm::vec3 *tangentAccum,
    const glm::vec3 *bitangentAccum,
    const glm::vec3 *normalAccum,
    unsigned int numVertices)
{
  unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;

  if (v >= numVertices)
    return;

  const vec3 T_in = tangentAccum[v];
  const vec3 B_in = bitangentAccum[v];
  const vec3 N_in = normalAccum[v];

  const vec3 n = safeNormalize(N_in, vec3(0.0f, 0.0f, 1.0f));

  vec3 fallbackT, fallbackB;
  makeTangentFrame(n, &fallbackT, &fallbackB);

  const vec3 T_orth =
      safeNormalize(T_in - n * glm::dot(n, T_in), fallbackT);

  const float bitangentSign = glm::dot(glm::cross(n, T_orth), B_in);
  const float sign = bitangentSign < 0.0f ? -1.0f : 1.0f;

  tangents[v] = glm::vec4(T_orth, sign);
}

void updateGeometryTangent(Triangle *triangle)
{
  auto indices = triangle->getParamObject<Array1D>("primitive.index");
  auto positions = triangle->getParamObject<Array1D>("vertex.position");
  auto normals = triangle->getParamObject<Array1D>("vertex.normal");
  auto uvs = triangle->getParamObject<Array1D>("vertex.attribute0");
  auto normalsFV = triangle->getParamObject<Array1D>("faceVarying.normal");
  auto uvsFV = triangle->getParamObject<Array1D>("faceVarying.attribute0");

  if (!positions) {
    triangle->reportMessage(ANARI_SEVERITY_INFO,
        "Triangle %p has no positions, cannot compute tangents",
        triangle);
    return;
  }

  if (!uvs && !uvsFV) {
    triangle->reportMessage(ANARI_SEVERITY_INFO,
        "Triangle %p has no texture coordinates, cannot compute tangents",
        triangle);
    return;
  }

  if (uvsFV && uvsFV->elementType() != ANARI_FLOAT32_VEC2
      && uvsFV->elementType() != ANARI_FLOAT32_VEC3) {
    triangle->reportMessage(ANARI_SEVERITY_INFO,
        "Can only compute tangents for face varying UVs of type ANARI_FLOAT32_VEC2 or ANARI_FLOAT32_VEC3",
        triangle);
    return;
  }

  if (uvs && uvs->elementType() != ANARI_FLOAT32_VEC2
      && uvs->elementType() != ANARI_FLOAT32_VEC3) {
    triangle->reportMessage(ANARI_SEVERITY_INFO,
        "Can only compute tangents for vertex UVs of type ANARI_FLOAT32_VEC2 or ANARI_FLOAT32_VEC3",
        triangle);
    return;
  }

  // Output is per-vertex (vertex.tangent). For indexed meshes the per-vertex
  // buffer is what lets adjacent triangles share tangent data at common
  // vertices — that sharing is what eliminates the per-triangle facets a
  // face-varying buffer would produce. For triangle-soup input each face-vertex
  // is its own slot, so the same layout works without changes.
  const auto numVertices = static_cast<unsigned int>(positions->size());
  const auto trianglesCount = static_cast<unsigned int>(
      indices ? indices->size() : positions->size() / 3);
  if (trianglesCount == 0 || numVertices == 0) {
    triangle->reportMessage(ANARI_SEVERITY_INFO,
        "Triangle %p has no triangles, cannot compute tangents",
        triangle);
    return;
  }

  glm::vec3 *tangentAccum = nullptr;
  glm::vec3 *bitangentAccum = nullptr;
  glm::vec3 *normalAccum = nullptr;
  glm::vec4 *tangents = nullptr;

  auto cleanup = [&] {
    cudaFree(tangentAccum);
    cudaFree(bitangentAccum);
    cudaFree(normalAccum);
    cudaFree(tangents);
  };

  auto status = cudaMalloc(reinterpret_cast<void **>(&tangentAccum),
      sizeof(glm::vec3) * numVertices);
  if (reportCudaError(triangle, status, "allocating tangent accumulator")) {
    cleanup();
    return;
  }
  status = cudaMalloc(reinterpret_cast<void **>(&bitangentAccum),
      sizeof(glm::vec3) * numVertices);
  if (reportCudaError(triangle, status, "allocating bitangent accumulator")) {
    cleanup();
    return;
  }
  status = cudaMalloc(reinterpret_cast<void **>(&normalAccum),
      sizeof(glm::vec3) * numVertices);
  if (reportCudaError(triangle, status, "allocating normal accumulator")) {
    cleanup();
    return;
  }
  status = cudaMalloc(
      reinterpret_cast<void **>(&tangents), sizeof(glm::vec4) * numVertices);
  if (reportCudaError(triangle, status, "allocating tangent output buffer")) {
    cleanup();
    return;
  }

  status = cudaMemset(tangentAccum, 0, sizeof(glm::vec3) * numVertices);
  if (reportCudaError(triangle, status, "clearing tangent accumulator")) {
    cleanup();
    return;
  }
  status = cudaMemset(bitangentAccum, 0, sizeof(glm::vec3) * numVertices);
  if (reportCudaError(triangle, status, "clearing bitangent accumulator")) {
    cleanup();
    return;
  }
  status = cudaMemset(normalAccum, 0, sizeof(glm::vec3) * numVertices);
  if (reportCudaError(triangle, status, "clearing normal accumulator")) {
    cleanup();
    return;
  }

  auto positionsPtr = positions->dataAs<const glm::vec3>(AddressSpace::GPU);
  if (indices) {
    auto indicesPtr = indices->dataAs<const glm::uvec3>(AddressSpace::GPU);
    if (normalsFV) {
      auto normalsPtr = normalsFV->dataAs<const glm::vec3>(AddressSpace::GPU);
      if (uvsFV) {
        if (uvsFV->elementType() == ANARI_FLOAT32_VEC2) {
          auto uvsPtr = uvsFV->dataAs<const glm::vec2>(AddressSpace::GPU);
          // Vertex indexed, face varying normals and face varyings vec2 UVs.
          __computeTangents<true, false, false>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          auto uvsPtr = uvsFV->dataAs<const glm::vec3>(AddressSpace::GPU);
          // Vertex indexed, face varying normals and face varyings vec3 UVs.
          __computeTangents<true, false, false>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        }
      } else {
        if (uvs->elementType() == ANARI_FLOAT32_VEC2) {
          // Vertex indexed,  face varying normals and indexed vec2 UVs.
          auto uvsPtr = uvs->dataAs<const glm::vec2>(AddressSpace::GPU);
          __computeTangents<true, false, true>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          // Vertex indexed,  face varying normals and indexed vec3 UVs.
          auto uvsPtr = uvs->dataAs<const glm::vec3>(AddressSpace::GPU);
          __computeTangents<true, false, true>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        }
      }
    } else {
      const auto *normalsPtr = normals
          ? normals->dataAs<const glm::vec3>(AddressSpace::GPU)
          : nullptr;
      if (uvsFV) {
        if (uvsFV->elementType() == ANARI_FLOAT32_VEC2) {
          auto uvsPtr = uvsFV->dataAs<const glm::vec2>(AddressSpace::GPU);
          // Vertex indexed, index normals and face varyings vec2 UVs.
          __computeTangents<true, true, false>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          auto uvsPtr = uvsFV->dataAs<const glm::vec3>(AddressSpace::GPU);
          // Vertex indexed, indexed normals and face varyings vec3 UVs.
          __computeTangents<true, true, false>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        }
      } else {
        if (uvs->elementType() == ANARI_FLOAT32_VEC2) {
          // Vertex indexed, indexed normals and indexed vec2 UVs.
          auto uvsPtr = uvs->dataAs<const glm::vec2>(AddressSpace::GPU);
          __computeTangents<true, true, true>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          // Vertex indexed, indexed normals and indexed vec3 UVs.
          auto uvsPtr = uvs->dataAs<const glm::vec3>(AddressSpace::GPU);
          __computeTangents<true, true, true>(tangentAccum,
              bitangentAccum,
              normalAccum,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        }
      }
    }
  } else {
    const glm::uvec3 *indicesPtr = nullptr;
    auto effectiveNormals = normalsFV ? normalsFV : normals;
    auto effectiveUvs = uvsFV ? uvsFV : uvs;

    const auto *normalsPtr = effectiveNormals
        ? effectiveNormals->dataAs<const glm::vec3>(AddressSpace::GPU)
        : nullptr;

    if (effectiveUvs->elementType() == ANARI_FLOAT32_VEC2) {
      // Non indexed vertices, face varying normals and face varyings vec2 UVs.
      auto uvsPtr = effectiveUvs->dataAs<const glm::vec2>(AddressSpace::GPU);
      __computeTangents<false, false, false>(tangentAccum,
          bitangentAccum,
          normalAccum,
          indicesPtr,
          positionsPtr,
          normalsPtr,
          uvsPtr,
          trianglesCount);
    } else {
      // Non indexed vertices, face varying normals and face varyings vec3 UVs.
      auto uvsPtr = effectiveUvs->dataAs<const glm::vec3>(AddressSpace::GPU);
      __computeTangents<false, false, false>(tangentAccum,
          bitangentAccum,
          normalAccum,
          indicesPtr,
          positionsPtr,
          normalsPtr,
          uvsPtr,
          trianglesCount);
    }
  }

  status = cudaGetLastError();
  if (reportCudaError(triangle, status, "launching accumulate kernel")) {
    cleanup();
    return;
  }

  __doFinalizeTangents<<<(numVertices + 63) / 64, 64>>>(
      tangents, tangentAccum, bitangentAccum, normalAccum, numVertices);

  status = cudaGetLastError();
  if (reportCudaError(triangle, status, "launching finalize kernel")) {
    cleanup();
    return;
  }

  status = cudaDeviceSynchronize();
  cudaFree(tangentAccum);
  cudaFree(bitangentAccum);
  cudaFree(normalAccum);
  tangentAccum = nullptr;
  bitangentAccum = nullptr;
  normalAccum = nullptr;
  if (reportCudaError(triangle, status, "computing tangents")) {
    cleanup();
    return;
  }

  auto desc = Array1DMemoryDescriptor{
      {
          tangents,
          cudaFreeMemoryDeleter, // deleter
          {}, // deleterPtr
          ANARI_FLOAT32_VEC4,
      },
      numVertices,
  };
  auto tangentsArray = new Array1D(triangle->deviceState(), desc);
  tangentsArray->commitParameters();
  tangentsArray->finalize();

  triangle->setParam("vertex.tangent", tangentsArray);
  triangle->commitParameters();
  triangle->finalize();

  tangentsArray->refDec(helium::PUBLIC);
}

} // namespace visrtx
