/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Surface.h"

namespace visrtx {

Surface::Surface(DeviceGlobalState *d)
    : RegisteredObject<SurfaceGPUData>(ANARI_SURFACE, d)
{
  setRegistry(d->registry.surfaces);
}

void Surface::commit()
{
  m_id = getParam<uint32_t>("id", ~0u);
  m_geometry = getParamObject<Geometry>("geometry");
  m_material = getParamObject<Material>("material");

  if (!m_material) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'material' on ANARISurface");
    return;
  }

  if (!m_geometry) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'geometry' on ANARISurface");
    return;
  }

  upload();
}

const Geometry *Surface::geometry() const
{
  return m_geometry.ptr;
}

const Material *Surface::material() const
{
  return m_material.ptr;
}

OptixBuildInput Surface::buildInput() const
{
  OptixBuildInput obi = {};
  if (geometryIsValid())
    m_geometry->populateBuildInput(obi);
  return obi;
}

void Surface::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLASChange = helium::newTimeStamp();
}

bool Surface::isValid() const
{
  return geometryIsValid() && materialIsValid();
}

bool Surface::geometryIsValid() const
{
  return m_geometry && m_geometry->isValid();
}

bool Surface::materialIsValid() const
{
  return m_material && m_material->isValid();
}

SurfaceGPUData Surface::gpuData() const
{
  SurfaceGPUData retval;
  retval.id = m_id;
  retval.geometry = geometry() ? geometry()->index() : -1;
  retval.material = material() ? material()->index() : -1;
  return retval;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Surface *);
