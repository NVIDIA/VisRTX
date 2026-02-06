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

#include "VisGLSpecializations.h"

#include "math_util.h"
#include <math.h>

#include <cstdlib>
#include <cstring>

namespace visgl {

Object<CameraPerspective>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  commit();
}

void Object<CameraPerspective>::calculateMatrices(float near, float far) {}

void Object<CameraPerspective>::commit()
{
  DefaultObject::commit();

  current.position.get(ANARI_FLOAT32_VEC3, position);
  current.direction.get(ANARI_FLOAT32_VEC3, direction);
  current.up.get(ANARI_FLOAT32_VEC3, up);
  current.imageRegion.get(ANARI_FLOAT32_BOX2, region);
  current.fovy.get(ANARI_FLOAT32, &fovy);
  current.aspect.get(ANARI_FLOAT32, &aspect);

  normalize3(direction);
}

void Object<CameraPerspective>::updateAt(size_t index, float *bounds) const
{
  float range[2];
  projectBoundingBox(range, bounds, position, direction);
  float far = range[1] * 1.0001f;
  float near = fast_maxf(far * 1.0e-3, range[0] * 0.9999f);

  float height = near * tanf(fovy * 0.5f);
  float width = height * aspect;

  std::array<float, 16> projection;
  std::array<float, 16> view;
  std::array<float, 16> projection_view;

  std::array<float, 16> inverse_projection;
  std::array<float, 16> inverse_view;
  std::array<float, 16> inverse_projection_view;

  setFrustum(projection.data(),
      width * (2.0f * region[0] - 1.0f),
      width * (2.0f * region[2] - 1.0f),
      height * (2.0f * region[3] - 1.0f),
      height * (2.0f * region[1] - 1.0f),
      near,
      far);
  setLookDirection(view.data(), position, direction, up);
  mul3(projection_view.data(), projection.data(), view.data());

  setInverseFrustum(inverse_projection.data(),
      width * (2.0f * region[0] - 1.0f),
      width * (2.0f * region[2] - 1.0f),
      height * (2.0f * region[3] - 1.0f),
      height * (2.0f * region[1] - 1.0f),
      near,
      far);
  setInverseLookDirection(inverse_view.data(), position, direction, up);
  mul3(inverse_projection_view.data(),
      inverse_view.data(),
      inverse_projection.data());

  std::array<float, 16> aux{position[0],
      position[1],
      position[2],
      1.0f,
      position[0],
      position[1],
      position[2],
      1.0f};

  thisDevice->transforms.set(index + 0, projection_view);
  thisDevice->transforms.set(index + 1, inverse_projection_view);
  thisDevice->transforms.set(index + 2, projection);
  thisDevice->transforms.set(index + 3, inverse_projection);
  thisDevice->transforms.set(index + 4, view);
  thisDevice->transforms.set(index + 5, inverse_view);
  thisDevice->transforms.set(index + 6, aux);
}

} // namespace visgl
