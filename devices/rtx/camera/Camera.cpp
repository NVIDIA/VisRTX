/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Camera.h"
// specific types
#include "Orthographic.h"
#include "Perspective.h"
#include "UnknownCamera.h"
// std
#include <atomic>

namespace visrtx {

Camera::Camera(DeviceGlobalState *s) : Object(ANARI_CAMERA, s)
{
  helium::BaseObject::markParameterChanged();
  s->commitBuffer.addObjectToCommit(this);
}

void Camera::finalize()
{
  upload();
}

Camera *Camera::createInstance(std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "perspective")
    return new Perspective(d);
  else if (subtype == "orthographic")
    return new Orthographic(d);
  else
    return new UnknownCamera(subtype, d);
}

void *Camera::deviceData() const
{
  return DeviceObject<CameraGPUData>::deviceData();
}

void Camera::readBaseParameters(CameraGPUData &hd)
{
  vec4 region = vec4(0.f, 0.f, 1.f, 1.f);
  getParam("imageRegion", ANARI_FLOAT32_BOX2, &region);
  hd.region = region;
  hd.pos = getParam<vec3>("position", vec3(0.f));
  hd.dir = normalize(getParam<vec3>("direction", vec3(0.f, 0.f, 1.f)));
  hd.up = normalize(getParam<vec3>("up", vec3(0.f, 1.f, 0.f)));
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Camera *);
