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

#include "Instance.h"

namespace visrtx {

Instance::Instance(DeviceGlobalState *d)
    : Object(ANARI_INSTANCE, d), m_xfmArray(this), m_idArray(this)
{}

void Instance::commitParameters()
{
  m_idArray = getParamObject<Array1D>("id");
  m_id = getParam<uint32_t>("id", ~0u);
  m_xfmArray = getParamObject<Array1D>("transform");
  m_xfm = getParam<mat4x3>("transform", getParam<mat4>("transform", mat4(1)));
  m_group = getParamObject<Group>("group");

  auto getUniformAttribute =
      [&](const std::string &pName) -> std::optional<vec4> {
    vec4 v(0.f, 0.f, 0.f, 1.f);
    if (getParam(pName, ANARI_FLOAT32, &v))
      return v;
    else if (getParam(pName, ANARI_FLOAT32_VEC2, &v))
      return v;
    else if (getParam(pName, ANARI_FLOAT32_VEC3, &v))
      return v;
    else if (getParam(pName, ANARI_FLOAT32_VEC4, &v))
      return v;
    else
      return {};
  };

  m_uniformAttributes.attribute0Array = getParamObject<Array1D>("attribute0");
  m_uniformAttributes.attribute0 = getUniformAttribute("attribute0");

  m_uniformAttributes.attribute1Array = getParamObject<Array1D>("attribute1");
  m_uniformAttributes.attribute1 = getUniformAttribute("attribute1");

  m_uniformAttributes.attribute2 = getUniformAttribute("attribute2");
  m_uniformAttributes.attribute2Array = getParamObject<Array1D>("attribute2");

  m_uniformAttributes.attribute3 = getUniformAttribute("attribute3");
  m_uniformAttributes.attribute3Array = getParamObject<Array1D>("attribute3");

  m_uniformAttributes.color = getUniformAttribute("color");
  m_uniformAttributes.colorArray = getParamObject<Array1D>("color");
}

void Instance::finalize()
{
  if (m_idArray && m_idArray->elementType() != ANARI_UINT32) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'id' array elements are %s, but need to be %s",
        anari::toString(m_idArray->elementType()),
        anari::toString(ANARI_UINT32));
    m_idArray = {};
  }
  if (m_xfmArray && m_xfmArray->elementType() != ANARI_FLOAT32_MAT4) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'transform' array elements are %s, but need to be %s",
        anari::toString(m_idArray->elementType()),
        anari::toString(ANARI_FLOAT32_MAT4));
    m_xfmArray = {};
  }
  if (m_xfmArray) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "using array transforms for ANARIInstance of size %zu",
        m_xfmArray->totalSize());
  }
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");
}

void Instance::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastTLASChange = helium::newTimeStamp();
}

bool Instance::isValid() const
{
  return m_group;
}

uint32_t Instance::userID(size_t i) const
{
  return m_xfmArray && m_idArray ? *m_idArray->valueAt<uint32_t>(i) : m_id;
}

size_t Instance::numTransforms() const
{
  return m_xfmArray ? m_xfmArray->totalSize() : 1;
}

mat4x3 Instance::xfm(size_t i) const
{
  return m_xfmArray ? mat4x3(*m_xfmArray->valueAt<mat4>(i)) : m_xfm;
}

bool Instance::xfmIsIdentity(size_t i) const
{
  return xfm(i) == mat4x3(1);
}

const Group *Instance::group() const
{
  return m_group.ptr;
}

Group *Instance::group()
{
  return m_group.ptr;
}

const UniformAttributes &Instance::uniformAttributes() const
{
  return m_uniformAttributes;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Instance *);
