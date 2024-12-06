## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(anari_library_tsd LANGUAGES CXX)

project_add_library(SHARED
  Array.cpp
  Device.cpp
  DeviceGlobalState.cpp
  Frame.cpp
  Group.cpp
  Instance.cpp
  Library.cpp
  Object.cpp
  World.cpp
)
project_link_libraries(PRIVATE tsd_app tsd_io anari::helium)
project_compile_definitions(PRIVATE "anari_library_tsd_EXPORTS")
project_include_directories(
PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

## Generated code ##

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_MACRO_NAME "ANARI_TSD_EXPORT")

anari_generate_queries(
  DEVICE_TARGET ${PROJECT_NAME}
  CPP_NAMESPACE tsd_device
  JSON_DEFINITIONS_FILE ${ANARI_CODE_GEN_ROOT}/devices/experimental_device.json
)

## Install ##

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
