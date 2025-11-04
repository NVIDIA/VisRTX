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

## Included CMake modules ##

include(ExternalProject)
include(GNUInstallDirs)
include(ProcessorCount)

## Variables ##

ProcessorCount(PROCESSOR_COUNT)
set(NUM_BUILD_JOBS ${PROCESSOR_COUNT} CACHE STRING "Number of build jobs '-j <n>'")
set(DEFAULT_BUILD_COMMAND cmake --build . --config release -j ${NUM_BUILD_JOBS})
get_filename_component(INSTALL_DIR_ABSOLUTE
  ${CMAKE_INSTALL_PREFIX} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

## Functions/macros ##

function(print)
  foreach(arg ${ARGN})
    message("${arg} = ${${arg}}")
  endforeach()
endfunction()

macro(append_cmake_prefix_path)
  list(APPEND CMAKE_PREFIX_PATH ${ARGN})
  string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
endmacro()

macro(setup_subproject_path_vars _NAME)
  set(SUBPROJECT_NAME ${_NAME})

  set(SUBPROJECT_INSTALL_PATH ${INSTALL_DIR_ABSOLUTE})

  set(SUBPROJECT_SOURCE_PATH ${SUBPROJECT_NAME}/source)
  set(SUBPROJECT_STAMP_PATH ${SUBPROJECT_NAME}/stamp)
  set(SUBPROJECT_BUILD_PATH ${SUBPROJECT_NAME}/build)
endmacro()

macro(build_subproject)
  # See cmake_parse_arguments docs to see how args get parsed here:
  #    https://cmake.org/cmake/help/latest/command/cmake_parse_arguments.html
  set(oneValueArgs NAME URL)
  set(multiValueArgs BUILD_ARGS DEPENDS_ON)
  cmake_parse_arguments(BUILD_SUBPROJECT "" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  # Setup SUBPROJECT_* variables (containing paths) for this function
  setup_subproject_path_vars(${BUILD_SUBPROJECT_NAME})

  # Build the actual subproject
  ExternalProject_Add(${SUBPROJECT_NAME}
    PREFIX ${SUBPROJECT_NAME}
    DOWNLOAD_DIR ${SUBPROJECT_NAME}
    STAMP_DIR ${SUBPROJECT_STAMP_PATH}
    SOURCE_DIR ${SUBPROJECT_SOURCE_PATH}
    BINARY_DIR ${SUBPROJECT_BUILD_PATH}
    URL ${BUILD_SUBPROJECT_URL}
    LIST_SEPARATOR | # Use the alternate list separator
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=${SUBPROJECT_INSTALL_PATH}
      -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
      -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
      -DCMAKE_INSTALL_DOCDIR=${CMAKE_INSTALL_DOCDIR}
      -DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
      -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
      ${BUILD_SUBPROJECT_BUILD_ARGS}
    BUILD_COMMAND ${DEFAULT_BUILD_COMMAND}
    BUILD_ALWAYS OFF
  )

  if(BUILD_SUBPROJECT_DEPENDS_ON)
    ExternalProject_Add_StepDependencies(${SUBPROJECT_NAME}
      configure ${BUILD_SUBPROJECT_DEPENDS_ON}
    )
  endif()

  # Place installed component on CMAKE_PREFIX_PATH for downstream consumption
  append_cmake_prefix_path(${SUBPROJECT_INSTALL_PATH})
endmacro()
