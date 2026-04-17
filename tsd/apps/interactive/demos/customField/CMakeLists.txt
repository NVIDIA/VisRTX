## Copyright 2025-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

## Custom Field Demo
##
## This demo showcases a custom spatial field type for VisRTX using the
## "weightedPoints" field as an example.  The field implementation is compiled
## directly into the VisRTX shared library (not into the demo executable),
## so it requires the RTX device to be built.

if (NOT TARGET anari_library_visrtx)
  return()
endif()

## ---------------------------------------------------------------------------
## 1. Inject custom field implementation into the VisRTX shared library
## ---------------------------------------------------------------------------

set(CUSTOM_FIELD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/fields/RegisterWeightedPointsField.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fields/WeightedPointsField.cpp
)

target_sources(anari_library_visrtx PRIVATE ${CUSTOM_FIELD_SOURCES})

target_include_directories(anari_library_visrtx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/fields
)

## Configure the PTX compilation target to include our sampler dispatch
if (TARGET CustomFieldSampler_ptx)
  target_include_directories(CustomFieldSampler_ptx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fields
  )
  target_compile_definitions(CustomFieldSampler_ptx PRIVATE
    VISRTX_CUSTOM_FIELD_DATA_HEADER="WeightedPointsFieldData.h"
    VISRTX_CUSTOM_SAMPLERS_HEADER="WeightedPointsFieldDispatch.h"
  )
endif()

## ---------------------------------------------------------------------------
## 2. Build the demo application
## ---------------------------------------------------------------------------

project(tsdDemoCustomField)
project_add_executable(
  WeightedPointsControls.cpp
  WeightedPointsOctree.cpp
  tsdDemoCustomField.cpp
)
project_link_libraries(tsd_ui_imgui)
