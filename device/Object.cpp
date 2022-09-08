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

#include "Object.h"
// std
#include <atomic>
#include <cstdarg>

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

std::string string_printf(const char *fmt, ...)
{
  std::string s;
  va_list args, args2;
  va_start(args, fmt);
  va_copy(args2, args);

  s.resize(vsnprintf(nullptr, 0, fmt, args2) + 1);
  va_end(args2);
  vsprintf(s.data(), fmt, args);
  va_end(args);
  s.pop_back();
  return s;
}

// Object definitions /////////////////////////////////////////////////////////

void Object::commit()
{
  // no-op
}

void Object::upload()
{
  // no-op
}

bool Object::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "valid" && type == ANARI_BOOL) {
    writeToVoidP(ptr, isValid());
    return true;
  }

  return false;
}

void *Object::deviceData() const
{
  return nullptr;
}

bool Object::isValid() const
{
  return true;
}

void Object::setObjectType(ANARIDataType type)
{
  m_type = type;
}

ANARIDataType Object::type() const
{
  return m_type;
}

DeviceGlobalState *Object::deviceState() const
{
  return m_deviceState;
}

void Object::setDeviceState(DeviceGlobalState *d)
{
  if (m_deviceState)
    throw std::runtime_error("cannot re-target the device ptr in an object");

  m_deviceState = d;
}

TimeStamp Object::lastUpdated() const
{
  return m_lastUpdated;
}

void Object::markUpdated()
{
  m_lastUpdated = newTimeStamp();
}

TimeStamp Object::lastCommitted() const
{
  return m_lastCommitted;
}

void Object::markCommitted()
{
  m_lastCommitted = newTimeStamp();
}

int Object::commitPriority() const
{
  return m_commitPriority;
}

void Object::setCommitPriority(int priority)
{
  m_commitPriority = priority;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Object *);
