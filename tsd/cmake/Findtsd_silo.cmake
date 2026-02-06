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

if (TARGET tsd::silo)
  return()
endif()

find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(SILO QUIET silo siloh5)
endif()

if (NOT SILO_FOUND)
  # Try to find Silo manually
  find_path(SILO_INCLUDE_DIR silo.h)
  find_library(SILO_LIBRARY NAMES siloh5 silo)
  if (SILO_INCLUDE_DIR AND SILO_LIBRARY)
    set(SILO_FOUND TRUE)
    set(SILO_INCLUDE_DIRS ${SILO_INCLUDE_DIR})
    set(SILO_LIBRARIES ${SILO_LIBRARY})
  endif()
endif()

if (SILO_FOUND)
  add_library(tsd::silo INTERFACE IMPORTED)
  target_include_directories(tsd::silo INTERFACE ${SILO_INCLUDE_DIRS})
  target_link_libraries(tsd::silo INTERFACE ${SILO_LIBRARIES})
  message(STATUS "Found Silo: ${SILO_LIBRARIES}")
else()
  if(tsd_silo_FIND_REQUIRED)
    message(FATAL_ERROR "Silo not found")
  else()
    message(WARNING "Silo requested but not found")
  endif()
endif()
