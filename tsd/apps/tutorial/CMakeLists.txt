## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

macro(define_tutorial TUTORIAL_NAME)
  project(${TUTORIAL_NAME})
  project_add_executable(${TUTORIAL_NAME}.cpp)
  project_link_libraries(
    tsd_core
    tsd_rendering
    tsd_io
    stb_image
    tsd_ext_flip
  )
endmacro()

define_tutorial(tsdTutorialDataTreeDemo)
define_tutorial(tsdTutorialFlipDiff)
define_tutorial(tsdTutorialForestDemo)
define_tutorial(tsdTutorialLoadContext)
define_tutorial(tsdTutorialLoadObj)
define_tutorial(tsdTutorialLoadUSD)
define_tutorial(tsdTutorialMultiRender)
define_tutorial(tsdTutorialPipeline)
define_tutorial(tsdTutorialRender)
define_tutorial(tsdTutorialSaveContext)
define_tutorial(tsdTutorialSimple)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(tsdTutorialFlipDiff OpenMP::OpenMP_CXX)
endif()
