# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (TARGET visrtx::stb_image)
  return()
endif()

# Check if a different stb_image target exists (e.g., from OWL)
if (TARGET stb_image)
  # Create an alias with VisRTX's expected name
  if (NOT TARGET visrtx_stb_image)
    add_library(visrtx_stb_image STATIC stb_image_write.c stb_image.c)
    target_include_directories(visrtx_stb_image INTERFACE ${CMAKE_CURRENT_LIST_DIR})
    add_library(visrtx::stb_image ALIAS visrtx_stb_image)
    # Also create the expected bare name alias for internal use
    add_library(stb_image_visrtx ALIAS visrtx_stb_image)
  endif()
  return()
endif()

project(stb_image LANGUAGES C)
add_library(stb_image STATIC stb_image_write.c stb_image.c)
target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_LIST_DIR})
add_library(visrtx::stb_image ALIAS stb_image)
