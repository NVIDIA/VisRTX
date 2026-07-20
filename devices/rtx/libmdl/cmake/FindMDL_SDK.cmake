# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (TARGET MDL_SDK::MDL_SDK)
  return()
endif()

find_path(MDL_SDK_ROOT NAMES "include/mi/mdl_sdk.h" PATHS ${MDL_SDK_PATH} ENV MDL_SDK_PATH)

# Parallel worker compilation (ADR 0009) relies on the MDL SDK's multiple
# parallel transactions, first shipped in 2023.1.0. Gate the build on it so a
# too-old SDK fails at configure, not with a runtime data race.
set(MDL_SDK_MIN_VERSION "2023.1.0")
if (MDL_SDK_ROOT AND EXISTS "${MDL_SDK_ROOT}/include/mi/neuraylib/version.h")
  file(STRINGS "${MDL_SDK_ROOT}/include/mi/neuraylib/version.h" _mdl_version_line
    REGEX "^#define[ \t]+MI_NEURAYLIB_PRODUCT_VERSION_STRING[ \t]+\"[0-9.]+\"")
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" MDL_SDK_VERSION "${_mdl_version_line}")
  unset(_mdl_version_line)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MDL_SDK
  REQUIRED_VARS MDL_SDK_ROOT
  VERSION_VAR MDL_SDK_VERSION)

if (MDL_SDK_FOUND AND MDL_SDK_VERSION AND MDL_SDK_VERSION VERSION_LESS MDL_SDK_MIN_VERSION)
  message(FATAL_ERROR
    "MDL SDK ${MDL_SDK_VERSION} is too old: parallel MDL compilation (ADR 0009) "
    "requires ${MDL_SDK_MIN_VERSION}+ for parallel database transactions.")
endif()

set(MDL_SDK_INCLUDE_DIR ${MDL_SDK_ROOT}/include)
set(MDL_SDK_INCLUDE_DIRS ${MDL_SDK_ROOT}/include)
mark_as_advanced(MDL_SDK_INCLUDE_DIR MDL_SDK_INCLUDE_DIRS)

add_library(MDL_SDK::MDL_SDK INTERFACE IMPORTED)
target_include_directories(MDL_SDK::MDL_SDK INTERFACE ${MDL_SDK_INCLUDE_DIR})
