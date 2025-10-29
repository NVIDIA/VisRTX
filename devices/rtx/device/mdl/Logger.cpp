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

#include "Logger.h"

#include "optix_visrtx.h"

#include <anari/frontend/anari_enums.h>

#include <fmt/format.h>

namespace {

ANARIStatusSeverity miSeverityToAnari(mi::base::Message_severity severity)
{
  switch (severity) {
  case mi::base::MESSAGE_SEVERITY_ERROR:
    return ANARI_SEVERITY_ERROR;
  case mi::base::MESSAGE_SEVERITY_WARNING:
    return ANARI_SEVERITY_WARNING;
  case mi::base::MESSAGE_SEVERITY_INFO:
    return ANARI_SEVERITY_INFO;
  case mi::base::MESSAGE_SEVERITY_VERBOSE:
    return ANARI_SEVERITY_INFO;
  case mi::base::MESSAGE_SEVERITY_DEBUG:
    return ANARI_SEVERITY_DEBUG;
  default:
    return ANARI_SEVERITY_INFO;
  }
}

} // namespace

namespace visrtx::mdl {

void Logger::message(mi::base::Message_severity level,
    const char *moduleCategory,
    const mi::base::Message_details &details,
    const char *message)
{
  this->message(level, moduleCategory, message);
}

void Logger::message(mi::base::Message_severity level,
    const char *moduleCategory,
    const char *message)
{
  m_deviceState->messageFunction(miSeverityToAnari(level),
      fmt::format("[VISRTX:MDL:{}] {}", moduleCategory, message),
      ANARI_UNKNOWN,
      nullptr);
}

} // namespace visrtx::mdl
