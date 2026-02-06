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

cmake_minimum_required(VERSION 3.17)

include(GenerateExportHeader)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

project(anari_library_visgl LANGUAGES C CXX)

find_package(anari 0.15.0 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL OPTIONAL_COMPONENTS EGL GLX)
find_package(X11)

anari_generate_frontend(
  TARGET generate_visgl_frontend
  DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/visgl_device.json
  NAME VisGL
  PREFIX VisGL
  NAMESPACE visgl
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
  CODE_HEADER_FILE ${CMAKE_CURRENT_LIST_DIR}/license.txt
  JSON ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(${PROJECT_NAME} SHARED
  src/VisGLLibrary.cpp
  src/VisGLDevice.cpp
  src/VisGLArrayObjects.cpp
  src/VisGLInstanceObject.cpp
  src/VisGLGroupObject.cpp
  src/VisGLWorldObject.cpp
  src/VisGLSurfaceObject.cpp
  src/VisGLFrameObject.cpp
  src/VisGLDeviceObject.cpp
  src/VisGLGeometryTriangleObject.cpp
  src/VisGLGeometrySphereObject.cpp
  src/VisGLGeometryCylinderObject.cpp
  src/VisGLMaterialMatteObject.cpp
  src/VisGLMaterialPbrObject.cpp
  src/VisGLCameraPerspectiveObject.cpp
  src/VisGLCameraOrthographicObject.cpp
  src/VisGLLightDirectionalObject.cpp
  src/VisGLLightPointObject.cpp
  src/VisGLLightSpotObject.cpp
  src/VisGLSamplerImage1DObject.cpp
  src/VisGLSamplerImage2DObject.cpp
  src/VisGLSamplerCompressedImage2DObject.cpp
  src/VisGLSamplerImage3DObject.cpp
  src/VisGLSamplerPrimitiveObject.cpp
  src/VisGLSamplerTransformObject.cpp
  src/VisGLSpatial_FieldStructuredRegularObject.cpp
  src/VisGLVolumeTransferFunction1DObject.cpp
  src/VisGLRendererDefaultObject.cpp
  src/sphere_sample_directions.cpp
  generated/VisGLObjects.cpp
  generated/VisGLDeviceFactories.cpp
  generated/VisGLString.cpp
  src/glad/src/gl.c
)

anari_generate_queries(
  DEVICE_TARGET ${PROJECT_NAME}
  CPP_NAMESPACE visgl
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visgl_device.json
)

generate_export_header(${PROJECT_NAME}
  EXPORT_MACRO_NAME "VISGL_DEVICE_INTERFACE"
)

target_include_directories(${PROJECT_NAME}
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/glad/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/egl>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/glx>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/wgl>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  anari::anari_headers
  anari::anari_backend
  anari::anari ## required because the device uses the C API internally
  Threads::Threads
)

if(WIN32)
  target_sources(${PROJECT_NAME} PRIVATE
    src/wgl/wgl_context.cpp
    src/glad/src/wgl.c
  )
  target_link_libraries(${PROJECT_NAME}
  PUBLIC
    OpenGL::GL
  )
  target_compile_definitions(${PROJECT_NAME}
  PUBLIC
    VISGL_USE_WGL
  )
else()
  target_link_libraries(${PROJECT_NAME}
  PUBLIC
    OpenGL::OpenGL
  )
endif()

if(OpenGL_EGL_FOUND)
  target_sources(${PROJECT_NAME} PRIVATE
    src/egl/egl_context.cpp
  )
  target_link_libraries(${PROJECT_NAME}
  PUBLIC
    OpenGL::EGL
  )
  target_compile_definitions(${PROJECT_NAME}
  PUBLIC
    VISGL_USE_EGL
  )
endif()

if(OpenGL_GLX_FOUND AND X11_FOUND)
  target_sources(${PROJECT_NAME} PRIVATE
    src/glx/glx_context.cpp
  )
  target_link_libraries(${PROJECT_NAME}
  PUBLIC
    OpenGL::GLX
    X11::X11
    X11::Xutil
  )
  target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${X11_INCLUDE_DIR}
  )

  target_compile_definitions(${PROJECT_NAME}
  PUBLIC
    VISGL_USE_GLX
  )
endif()

target_compile_definitions(${PROJECT_NAME}
PRIVATE
  "anari_library_visgl_EXPORTS"
)

if (MSVC)
  target_compile_options(${PROJECT_NAME}
  PRIVATE
    "/bigobj"
  )
endif()

include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
  EXPORT VisGL_Exports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
