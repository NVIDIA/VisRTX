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

include(${CMAKE_CURRENT_LIST_DIR}/appendSearchPaths.cmake)

function(EmbedPTX)
  set(oneArgs OUTPUT_HEADER_FILE INPUT_TARGET)
  set(multiArgs OUTPUT_TARGETS)
  cmake_parse_arguments(EMBED_PTX "" "${oneArgs}" "${multiArgs}" ${ARGN})

  ## Validate incoming target ##

  get_target_property(INPUT_TARGET_TYPE ${EMBED_PTX_INPUT_TARGET} TYPE)
  if (NOT "${INPUT_TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY")
    message(FATAL_ERROR "EmbedPTX can only take object libraries")
  endif()

  get_target_property(PTX_PROP ${EMBED_PTX_INPUT_TARGET} CUDA_PTX_COMPILATION)
  get_target_property(OPTIX_PROP ${EMBED_PTX_INPUT_TARGET} CUDA_OPTIX_COMPILATION)
  if (NOT PTX_PROP AND NOT OPTIX_PROP)
    message(FATAL_ERROR "'${EMBED_PTX_INPUT_TARGET}' target property 'CUDA_PTX_COMPILATION' must be set to 'ON'")
  endif()

  ## Find EmbedPTXRun CMake script ##

  list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH})
  list(PREPEND CMAKE_FIND_ROOT_PATH ${CMAKE_MODULE_PATH})
  appendSearchPaths(${CMAKE_MODULE_PATH})
  find_file(EMBED_PTX_RUN EmbedPTXRun.cmake)
  mark_as_advanced(EMBED_PTX_RUN)
  if(NOT EMBED_PTX_RUN)
    message(FATAL_ERROR "EmbedPTX.cmake and EmbedPTXRun.cmake must be on CMAKE_MODULE_PATH\n")
  endif()

  ## Create command to run the bin2c via the CMake script ##

  get_filename_component(OUTPUT_FILE_NAME ${EMBED_PTX_OUTPUT_HEADER_FILE} NAME)
  add_custom_command(
    OUTPUT ${EMBED_PTX_OUTPUT_HEADER_FILE}
    COMMAND ${CMAKE_COMMAND}
      "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
      "-DOBJECTS=$<TARGET_OBJECTS:${EMBED_PTX_INPUT_TARGET}>"
      "-DOUTPUT=${EMBED_PTX_OUTPUT_HEADER_FILE}"
      -P ${EMBED_PTX_RUN}
    VERBATIM
    DEPENDS $<TARGET_OBJECTS:${EMBED_PTX_INPUT_TARGET}> ${EMBED_PTX_INPUT_TARGET}
    COMMENT "Generating embedded PTX header file: ${OUTPUT_FILE_NAME}"
  )

  ## Establish dependencies for consuming targets ##

  get_filename_component(OUTPUT_DIR ${EMBED_PTX_OUTPUT_HEADER_FILE} DIRECTORY)
  foreach(OUT_TARGET ${EMBED_PTX_OUTPUT_TARGETS})
    target_include_directories(${OUT_TARGET} PRIVATE ${OUTPUT_DIR})
    target_sources(${OUT_TARGET} PRIVATE ${EMBED_PTX_OUTPUT_HEADER_FILE})
  endforeach()
endfunction()