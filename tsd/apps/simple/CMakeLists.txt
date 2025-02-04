## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

macro(define_example EXAMPLE_NAME)
  project(${EXAMPLE_NAME})
  project_add_executable(${EXAMPLE_NAME}.cpp)
  project_link_libraries(tsd tsd_stb tsd_flip)
endmacro()

define_example(tsdSimple)
define_example(tsdRender)
define_example(tsdMultiRender)
define_example(tsdPipeline)
define_example(tsdLoadObj)
define_example(tsdFlipDiff)
define_example(tsdForestDemo)
define_example(tsdLoadContext)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(tsdFlipDiff OpenMP::OpenMP_CXX)
endif()
target_link_libraries(tsdPipeline render_pipeline)
