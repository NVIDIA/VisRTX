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

#include "RayBuffer.h"

namespace visrtx {

RayBuffer::RayBuffer(DeviceGlobalState *s)
    : Camera(s), m_org(this), m_dir(this), m_tmin(this), m_tmax(this)
{}

void RayBuffer::commitParameters()
{
  Camera::commitParameters();
  m_org = getParamObject<Array2D>("ray.org");
  m_dir = getParamObject<Array2D>("ray.dir");
  m_tmin = getParamObject<Array2D>("ray.tmin");
  m_tmax = getParamObject<Array2D>("ray.tmax");
}

void RayBuffer::finalize()
{
  m_gpuData = {};

  bool sizeSet = false;
  bool sizesMatch = true;

  auto validate = [&](helium::ChangeObserverPtr<Array2D> &array,
                      const char *name,
                      ANARIDataType expectedType) {
    if (!array)
      return;
    if (array->elementType() != expectedType) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "'%s' on rayBuffer camera must have element type %s (got %s), "
          "ignoring",
          name,
          anari::toString(expectedType),
          anari::toString(array->elementType()));
      array = nullptr;
      return;
    }
    const auto size = array->size();
    if (!sizeSet) {
      m_gpuData.size = uvec2(size.x, size.y);
      sizeSet = true;
    } else if (m_gpuData.size != uvec2(size.x, size.y)) {
      sizesMatch = false;
    }
  };

  validate(m_org, "ray.org", ANARI_FLOAT32_VEC3);
  validate(m_dir, "ray.dir", ANARI_FLOAT32_VEC3);
  validate(m_tmin, "ray.tmin", ANARI_FLOAT32);
  validate(m_tmax, "ray.tmax", ANARI_FLOAT32);

  if (!sizesMatch) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "ray buffers on rayBuffer camera do not all have the same size");
    m_gpuData.size = uvec2(0u);
  }

  if (m_org)
    m_gpuData.org = m_org->dataAs<vec3>(AddressSpace::GPU);
  if (m_dir)
    m_gpuData.dir = m_dir->dataAs<vec3>(AddressSpace::GPU);
  if (m_tmin)
    m_gpuData.tmin = m_tmin->dataAs<float>(AddressSpace::GPU);
  if (m_tmax)
    m_gpuData.tmax = m_tmax->dataAs<float>(AddressSpace::GPU);

  m_lastWarnedFrameSize = uvec2(0u);
}

void RayBuffer::populateFrameData(CameraGPUData &fd, uvec2 frameSize) const
{
  populateBaseFrameData(fd);
  fd.type = CameraType::RAY_BUFFER;
  fd.rayBuffer = m_gpuData;

  const bool hasBuffers = m_org || m_dir || m_tmin || m_tmax;
  fd.rayBuffer.valid = !hasBuffers || m_gpuData.size == frameSize;

  if (!fd.rayBuffer.valid && m_lastWarnedFrameSize != frameSize) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "ray buffers on rayBuffer camera (%ux%u) do not match the frame size "
        "(%ux%u), all rays will miss",
        m_gpuData.size.x,
        m_gpuData.size.y,
        frameSize.x,
        frameSize.y);
    m_lastWarnedFrameSize = frameSize;
  }
}

} // namespace visrtx
