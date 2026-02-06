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

if (TARGET visrtx::glm)
  return()
endif()

# glm
find_package(glm CONFIG QUIET)
if (TARGET glm::glm)
  message(STATUS "Found glm: (external) ${glm_DIR}")
else()
  # Use locally provided version of glm
  set(glm_DIR ${CMAKE_CURRENT_LIST_DIR}/../external/glm/lib/cmake/glm)
  find_package(glm CONFIG REQUIRED)
  message(STATUS "Found glm: (internal) ${glm_DIR}")
endif()
mark_as_advanced(glm_DIR)

add_library(visrtx::glm INTERFACE IMPORTED)
target_link_libraries(visrtx::glm INTERFACE glm::glm)
target_compile_definitions(visrtx::glm INTERFACE GLM_ENABLE_EXPERIMENTAL)
if(WIN32)
  target_compile_definitions(visrtx::glm INTERFACE _USE_MATH_DEFINES NOMINMAX)
endif()
