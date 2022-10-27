/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ParameterizedObject.h"

namespace visrtx {

bool ParameterizedObject::hasParam(const std::string &name)
{
  return findParam(name, false) != nullptr;
}

void ParameterizedObject::setParam(
    const std::string &name, ANARIDataType type, const void *v)
{
  findParam(name, true)->second = AnariAny(type, v);
}

std::string ParameterizedObject::getParamString(
    const std::string &name, const std::string &valIfNotFound)
{
  auto *p = findParam(name);
  return p ? p->second.getString() : valIfNotFound;
}

AnariAny ParameterizedObject::getParamDirect(const std::string &name)
{
  auto *p = findParam(name);
  return p ? p->second : AnariAny();
}

void ParameterizedObject::setParamDirect(
    const std::string &name, const AnariAny &v)
{
  findParam(name, true)->second = v;
}

void ParameterizedObject::removeParam(const std::string &name)
{
  auto foundParam = std::find_if(m_params.begin(),
      m_params.end(),
      [&](const Param &p) { return p.first == name; });

  if (foundParam != m_params.end())
    m_params.erase(foundParam);
}

ParameterizedObject::Param *ParameterizedObject::findParam(
    const std::string &name, bool addIfNotExist)
{
  auto foundParam = std::find_if(m_params.begin(),
      m_params.end(),
      [&](const Param &p) { return p.first == name; });

  if (foundParam != m_params.end())
    return &(*foundParam);
  else if (addIfNotExist) {
    m_params.emplace_back(name, AnariAny());
    return &m_params[m_params.size() - 1];
  } else
    return nullptr;
}

} // namespace visrtx
