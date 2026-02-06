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
    glm::vec2 uv2
)
{
  // Compute edges of the triangle
  glm::vec3 e1 = p1 - p0;
  glm::vec3 e2 = p2 - p0;

  if (dot(e1, e1) < eps || dot(e2, e2) < eps) {
    // Degenerate triangle, use a default tangent and bitangent
    *tangent = glm::vec3(1.0f, 0.0f, 0.0f);
    *bitangent = glm::vec3(0.0f, 1.0f, 0.0f);
  } else {
    auto normal = normalize(cross(e1, e2));

    // Compute differences in texture coordinates
    auto s = uv1 - uv0;
    auto t = uv2 - uv0;

    auto cross = s.x * t.y - s.y * t.x;

    if (abs(cross) < eps) { // degenerate triangle (null vectors or collinears)
      // Create a default orthonormal basis:
      // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
      float sign = normal.z >= 0.0f ? 1.0f : -1.0f;
      float a = -1.0f / (sign + normal.z);
      float b = normal.x * normal.y * a;
      *tangent = glm::vec3(
          1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
      *bitangent = glm::vec3(b, sign + normal.y * normal.y * a, -normal.y);
    } else {
      // Compute the determinant
      float invdet = 1.0f / cross;
      *tangent = (t.y * e1 - s.y * e2) * invdet;
      *bitangent = (t.x * e1 - s.x * e2) * invdet;
    }
  }
}

template <bool VerticesIndexed,
    bool NormalsIndexed,
    bool UVsIndexed,
    typename TexCoord>
__global__ void __doComputeTangents(
    glm::vec4 *tangents, // Output tangent vectors with handedness (w component)
    glm::vec3 *bitangents, // Output bitangent vectors
    const glm::uvec3 *indices, // Input triangle indices
    const glm::vec3 *positions, // Input vertex positions
    const glm::vec3 *normals, // Input vertex normals
    const TexCoord *uvs, // Input texture coordinates
    unsigned int numTriangles // Number of triangles
)
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
    // Use indexed UVs
    uv0 = uvs[indexedIdx.x];
    uv1 = uvs[indexedIdx.y];
    uv2 = uvs[indexedIdx.z];
  } else {
    // Use per-face UVs
    uv0 = uvs[perFaceBaseIdx.x];
    uv1 = uvs[perFaceBaseIdx.y];
    uv2 = uvs[perFaceBaseIdx.z];
  }

  vec3 tangent, bitangent;

  __computeTangentAndBitangent(
      &tangent, // Output tangent vectors with handedness (w component)
      &bitangent, // Output bitangent vectors
      p0,
      p1,
      p2, // Input vertex positions
      uv0,
      uv1,
      uv2 // Input texture coordinates
  );

  vec3 n0, n1, n2;
  if constexpr (NormalsIndexed) {
    // Use indexed normals
    n0 = normals[indexedIdx.x];
    n1 = normals[indexedIdx.y];
    n2 = normals[indexedIdx.z];
  } else {
    // Use per-face normals
    n0 = normals[perFaceBaseIdx.x];
    n1 = normals[perFaceBaseIdx.y];
    n2 = normals[perFaceBaseIdx.z];
  }

  // Gram-Schmidt orthogonalize and compute handedness
  vec3 t0 = normalize(tangent - n0 * dot(n0, tangent));
  float h0 = copysign(1.0f, dot(cross(n0, t0), bitangent));

  vec3 t1 = normalize(tangent - n1 * dot(n1, tangent));
  float h1 = copysign(1.0f, dot(cross(n1, t1), bitangent));

  vec3 t2 = normalize(tangent - n2 * dot(n2, tangent));
  float h2 = copysign(1.0f, dot(cross(n2, t2), bitangent));

  tangents[perFaceBaseIdx.x] = glm::vec4(t0, h0);
  tangents[perFaceBaseIdx.y] = glm::vec4(t1, h1);
  tangents[perFaceBaseIdx.z] = glm::vec4(t2, h2);
}

template <bool VerticesIndexed,
    bool NormalsIndexed,
    bool UVsIndexed,
    typename TexCoord>
void __computeTangents(
    glm::vec4 *tangents, // Output tangent vectors with handedness (w component)
    glm::vec3 *bitangents, // Output bitangent vectors
    const glm::uvec3 *indices, // Input triangle indices
    const glm::vec3 *positions, // Input vertex positions
    const glm::vec3 *normals, // Input vertex normals
    const TexCoord *uvs, // Input texture coordinates
    unsigned int numTriangles // Number of triangles
)
{
  __doComputeTangents<VerticesIndexed, NormalsIndexed, UVsIndexed, TexCoord>
      <<<(numTriangles + 63) / 64, 64>>>(
          tangents, bitangents, indices, positions, normals, uvs, numTriangles);
}

void updateGeometryTangent(Triangle *triangle)
{
  auto indices = triangle->getParamObject<Array1D>("primitive.index");
  auto positions = triangle->getParamObject<Array1D>("vertex.position");
  auto normals = triangle->getParamObject<Array1D>("vertex.normal");
  auto uvs = triangle->getParamObject<Array1D>("vertex.attribute0");
  auto normalsFV = triangle->getParamObject<Array1D>("faceVarying.normal");
  auto uvsFV = triangle->getParamObject<Array1D>("faceVarying.attribute0");

  if (!positions || (!normals && !normalsFV)) {
    triangle->reportMessage(ANARI_SEVERITY_INFO,
        "Triangle %p has no position or normals, cannot compute tangents",
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

  // Always go with faceVarying tangents. Rational is the following:
  // - Correct UV and normal sharing is achieve through indexing
  // - If faceVarying UVs/normals are used then, it should already imply correct
  // sharing
  //   on common vertices.

  auto tangentsCount = indices ? (indices->size() * 3) : positions->size();
  auto trianglesCount = indices ? indices->size() : positions->size() / 3;
  glm::vec4 *tangents = {};
  cudaMalloc(&tangents, sizeof(glm::vec4) * tangentsCount);
  cudaMemset(tangents, 0, sizeof(glm::vec4) * tangentsCount);
  glm::vec3 *bitangents = {};
  cudaMalloc(&bitangents, sizeof(glm::vec3) * tangentsCount);
  cudaMemset(bitangents, 0, sizeof(glm::vec3) * tangentsCount);

  auto positionsPtr = positions->dataAs<const glm::vec3>(AddressSpace::GPU);
  if (indices) {
    auto indicesPtr = indices->dataAs<const glm::uvec3>(AddressSpace::GPU);
    if (normalsFV) {
      auto normalsPtr = normalsFV->dataAs<const glm::vec3>(AddressSpace::GPU);
      if (uvsFV) {
        if (uvsFV->elementType() == ANARI_FLOAT32_VEC2) {
          auto uvsPtr = uvsFV->dataAs<const glm::vec2>(AddressSpace::GPU);
          // Vertex indexed, face varying normals and face varyings vec2 UVs.
          __computeTangents<true, false, false>(tangents,
              bitangents,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          auto uvsPtr = uvsFV->dataAs<const glm::vec3>(AddressSpace::GPU);
          // Vertex indexed, face varying normals and face varyings vec3 UVs.
          __computeTangents<true, false, false>(tangents,
              bitangents,
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
          __computeTangents<true, false, true>(tangents,
              bitangents,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          // Vertex indexed,  face varying normals and indexed vec3 UVs.
          auto uvsPtr = uvs->dataAs<const glm::vec3>(AddressSpace::GPU);
          __computeTangents<true, false, true>(tangents,
              bitangents,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        }
      }
    } else {
      auto normalsPtr = normals->dataAs<const glm::vec3>(AddressSpace::GPU);
      if (uvsFV) {
        if (uvsFV->elementType() == ANARI_FLOAT32_VEC2) {
          auto uvsPtr = uvsFV->dataAs<const glm::vec2>(AddressSpace::GPU);
          // Vertex indexed, index normals and face varyings vec2 UVs.
          __computeTangents<true, true, false>(tangents,
              bitangents,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          auto uvsPtr = uvsFV->dataAs<const glm::vec3>(AddressSpace::GPU);
          // Vertex indexed, indexed normals and face varyings vec3 UVs.
          __computeTangents<true, true, false>(tangents,
              bitangents,
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
          __computeTangents<true, true, true>(tangents,
              bitangents,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        } else {
          // Vertex indexed, indexed normals and indexed vec3 UVs.
          auto uvsPtr = uvs->dataAs<const glm::vec3>(AddressSpace::GPU);
          __computeTangents<true, true, true>(tangents,
              bitangents,
              indicesPtr,
              positionsPtr,
              normalsPtr,
              uvsPtr,
              trianglesCount);
        }
      }
    }
  } else {
    auto indicesPtr = nullptr;
    normals = normalsFV ? normalsFV : normals;
    uvs = uvsFV ? uvsFV : uvs;

    auto normalsPtr = normals->dataAs<const glm::vec3>(AddressSpace::GPU);

    if (uvs->elementType() == ANARI_FLOAT32_VEC2) {
      // Non indexed vertices, face varying normals and face varyings vec2 UVs.
      auto uvsPtr = uvs->dataAs<const glm::vec2>(AddressSpace::GPU);
      __computeTangents<false, false, false>(tangents,
          bitangents,
          indicesPtr,
          positionsPtr,
          normalsPtr,
          uvsPtr,
          trianglesCount);
    } else {
      // Non indexed vertices, face varying normals and face varyings vec3 UVs.
      auto uvsPtr = uvs->dataAs<const glm::vec3>(AddressSpace::GPU);
      __computeTangents<false, false, false>(tangents,
          bitangents,
          indicesPtr,
          positionsPtr,
          normalsPtr,
          uvsPtr,
          trianglesCount);
    }
  }

  // Release transient bitangent store
  cudaFree(bitangents);

  auto desc = Array1DMemoryDescriptor{
      {
          tangents,
          {}, // deleter
          {}, // deleterPtr
          ANARI_FLOAT32_VEC4,
      },
      tangentsCount,
  };
  auto tangentsArray = new Array1D(triangle->deviceState(), desc);
  tangentsArray->commitParameters();
  tangentsArray->finalize();

  if (indices)
    triangle->setParam("faceVarying.tangent", tangentsArray);
  else
    triangle->setParam("vertex.tangent", tangentsArray);
  triangle->commitParameters();
  triangle->finalize();

  tangentsArray->refDec(helium::PUBLIC);
}

} // namespace visrtx
