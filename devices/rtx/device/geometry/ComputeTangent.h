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

#include "array/Array1D.h"
#include "geometry/Geometry.h"
#include "utility/DeviceBuffer.h"
// glm
#include <glm/fwd.hpp>

namespace visrtx {

// Where generated tangents live: one per vertex (`vertex.tangent` layout) or
// one per triangle corner (`faceVarying.tangent` layout, 3 per triangle).
enum class TangentLayout
{
  NONE,
  PER_VERTEX,
  PER_CORNER
};

// A triangle mesh to generate tangents for. Quads pass their two-triangle
// split. Arrays left null are absent; face-varying arrays hold 3 elements per
// triangle.
struct TangentGenerationInput
{
  const Array1D *positions{nullptr};
  // Device triangle indices into `positions`; null means triangle soup.
  const glm::uvec3 *indices{nullptr};
  size_t numTriangles{0};
  const Array1D *normals{nullptr};
  const Array1D *normalsFV{nullptr};
  const Array1D *uvs{nullptr}; // attribute0
  const Array1D *uvsFV{nullptr}; // attribute0
};

// Generate vec4(T, w) tangents that follow attribute0: T along +dP/du, and w
// such that w * cross(N, T) follows +dP/dv (ADR 0010). Corners that share a
// vertex share a tangent only when their normals and texture coordinates both
// match, so frames are smooth within a UV island but never blend across a
// crease or a UV seam. Writes `perCorner` when normals or attribute0 are
// face-varying on an indexed mesh, else `perVertex`, and empties the other.
// Returns NONE (both empty) when there is nothing to follow or generation
// fails.
TangentLayout generateTangents(Geometry *geometry,
    const TangentGenerationInput &input,
    DeviceBuffer &perVertex,
    DeviceBuffer &perCorner);

// Stage an authored tangent array in the internal vec4(T, sign) layout the
// shader reads. VEC4 is read zero-copy (leaves `converted` empty); the
// spec-allowed VEC3 is padded into `converted` with a default +1 handedness.
// Returns false if there are no usable tangents.
bool prepareTangentArray(Geometry *geometry,
    const helium::IntrusivePtr<Array1D> &tangents,
    DeviceBuffer &converted,
    const char *paramName);

// The device pointer the GPU reads tangents from: `converted` if non-empty,
// else a non-empty VEC4 `tangents` array, else null.
const glm::vec4 *resolveTangentPtr(
    const helium::IntrusivePtr<Array1D> &tangents,
    const DeviceBuffer &converted);

} // namespace visrtx
