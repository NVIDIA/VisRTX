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

#include <cfloat>
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

  // clip distances, same rule as VisRTX : unset means no clipping, invalid values are ignored
  nearClip = 0.0f;
  farClip = FLT_MAX;
  float nearValue = nearClip;
  float farValue = farClip;
  const bool hasNear = current.near.get(ANARI_FLOAT32, &nearValue);
  const bool hasFar = current.far.get(ANARI_FLOAT32, &farValue);
  if (hasNear || hasFar) {
    if ((hasNear && !(nearValue > 0.0f)) || !(nearValue < farValue)) {
      anariReportStatus(device,
          handle,
          ANARI_CAMERA,
          ANARI_SEVERITY_WARNING,
          ANARI_STATUS_INVALID_ARGUMENT,
          "camera 'near' (%f) and 'far' (%f) must satisfy 0 < near < far; ignoring both",
          nearValue,
          farValue);
    } else {
      nearClip = nearValue;
      farClip = farValue;
    }
  }
}

void Object<CameraPerspective>::updateAt(size_t index, float *bounds) const
{
  float range[2];
  projectBoundingBox(range, bounds, position, direction);
  // clip range enclosing the world, narrowed to the camera near and far when they are set.
  // Without a near distance, the near plane is pushed to far/1000 to keep the depth precision :
  // it clips the foreground when the world is much larger than the view (a globe seen from low altitude).
  float far = fast_minf(farClip, range[1] * 1.0001f);
  float near = nearClip > 0.0f ? fast_maxf(nearClip, range[0] * 0.9999f)
                               : fast_maxf(far * 1.0e-3f, range[0] * 0.9999f);
  if (!(far > near) && (nearClip > 0.0f || farClip < FLT_MAX)) {
    // the world is outside the camera clip range, nothing is visible : use that range itself
    near = nearClip > 0.0f ? nearClip : farClip * 1.0e-3f;
    far = farClip < FLT_MAX ? farClip : 2.0f * near;
  }

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
