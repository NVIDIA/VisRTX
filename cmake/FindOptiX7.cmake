# Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

if (TARGET OptiX7::OptiX7)
  return()
endif()

macro(OptiX7_config_message)
  if (NOT DEFINED OptiX7_FIND_QUIETLY)
    message(${ARGN})
  endif()
endmacro()

find_path(OptiX7_ROOT_DIR NAMES include/optix.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX7
  FOUND_VAR OptiX7_FOUND
  REQUIRED_VARS
    OptiX7_ROOT_DIR
  REASON_FAILURE_MESSAGE
    "OptiX7 installation not found on CMAKE_PREFIX_PATH (include/optix.h)"
)

if (NOT OptiX7_FOUND)
  set(OptiX7_NOT_FOUND_MESSAGE "Unable to find OptiX7, please add your OptiX7 installation to CMAKE_PREFIX_PATH")
  return()
endif()

set(OptiX7_INCLUDE_DIR ${OptiX7_ROOT_DIR}/include)
set(OptiX7_INCLUDE_DIRS ${OptiX7_INCLUDE_DIR})

add_library(OptiX7::OptiX7 INTERFACE IMPORTED)
target_include_directories(OptiX7::OptiX7 INTERFACE ${OptiX7_INCLUDE_DIR})
