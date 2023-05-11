# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CMAKE_CXX_STANDARD 11)

project(anari_library_visgl LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL OPTIONAL_COMPONENTS EGL GLX)
find_package(X11)

add_subdirectory(generated)

set(CMAKE_INSTALL_RPATH "$ORIGIN")

add_library(${PROJECT_NAME} SHARED
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
  src/VisGLMaterialTransparentMatteObject.cpp
  src/VisGLMaterialPbrObject.cpp
  src/VisGLCameraPerspectiveObject.cpp
  src/VisGLCameraOrthographicObject.cpp
  src/VisGLLightDirectionalObject.cpp
  src/VisGLLightPointObject.cpp
  src/VisGLSamplerImage1DObject.cpp
  src/VisGLSamplerImage2DObject.cpp
  src/VisGLSamplerImage3DObject.cpp
  src/VisGLSamplerPrimitiveObject.cpp
  src/VisGLSamplerTransformObject.cpp
  src/VisGLSpatial_FieldStructuredRegularObject.cpp
  src/VisGLVolumeScivisObject.cpp
  src/VisGLRendererDefaultObject.cpp
  src/sphere_sample_directions.cpp
  generated/VisGLObjects.cpp
  generated/VisGLDeviceFactories.cpp
  generated/VisGLString.cpp
  generated/VisGLQueries.cpp
  generated/VisGLDebugFactory.cpp
  src/glad/src/gl.c
)

target_include_directories(${PROJECT_NAME}
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  generated
  src
  src/glad/include
  src/egl/
  src/glx/
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
  anari::anari
  anari::anari_library_debug
  OpenGL::OpenGL
  Threads::Threads
)

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
    X11_INCLUDE_DIR
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

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)