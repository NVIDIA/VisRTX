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

#pragma once

#include <helium/utility/IntrusivePtr.h>
#include "Geometry.h"
#include "array/Array1D.h"
#include "utility/DeviceBuffer.h"

namespace visrtx {

struct Triangle : public Geometry
{
  Triangle(DeviceGlobalState *d);
  ~Triangle() override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  void populateBuildInput(OptixBuildInput &) const override;

  int optixGeometryType() const override;

 private:
  GeometryGPUData gpuData() const override;
  void cleanup();

  // Build the GPU-side staging for one authored tangent array into 'converted':
  // VEC4 input is read zero-copy (buffer left empty); VEC3 input is padded to
  // vec4 (sign defaulted to +1); unsupported element types are reported and
  // leave the buffer empty so resolveTangentPtr() emits no tangents. Called per
  // array during finalize().
  // Returns true if a valid tangent is useable, either directly, or converted
  bool prepareTangentArray(const helium::IntrusivePtr<Array1D> &tangents,
      DeviceBuffer &converted,
      const char *paramName);

  void generateVertexTangents(DeviceBuffer &generated);

  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertex;
  helium::IntrusivePtr<Array1D> m_vertexNormal;
  GeometryAttributes m_vertexAttributes;
  GeometryAttributes m_vertexAttributesFV;
  helium::IntrusivePtr<Array1D> m_vertexNormalFV;
  helium::IntrusivePtr<Array1D> m_vertexTangent;
  helium::IntrusivePtr<Array1D> m_vertexTangentFV;

  // Finalized per-vertex tangents (vec4). Empty when vertex.tangent is VEC4
  // (read zero-copy), or when it is absent/unusable and a usable
  // faceVarying.tangent took priority, or when auto-generation was attempted
  // but failed. Non-empty when holding VEC3->vec4 padded tangents, or
  // auto-generated tangents produced because no tangents were authored.
  DeviceBuffer m_vertexTangentFinalized;
  // Finalized faceVarying tangents. Empty if input is already VEC4
  DeviceBuffer m_vertexTangentFVFinalized;

  CUdeviceptr m_vertexBufferPtr{};

  bool m_cullBackfaces{false};
};

} // namespace visrtx
