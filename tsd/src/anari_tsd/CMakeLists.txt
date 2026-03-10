## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_device LANGUAGES CXX)

project_add_library(OBJECT
  Array.cpp
  Device.cpp
  DeviceGlobalState.cpp
  Frame.cpp
  Group.cpp
  Instance.cpp
  Object.cpp
  World.cpp
)
project_link_libraries(PUBLIC tsd_app tsd_io anari::helium)
project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

anari_generate_queries(
  DEVICE_TARGET ${PROJECT_NAME}
  CPP_NAMESPACE tsd_device
  JSON_DEFINITIONS_FILE ${ANARI_CODE_GEN_ROOT}/devices/experimental_device.json
)

## Installable TSD device library ##

project(anari_library_tsd LANGUAGES CXX)
project_add_library(SHARED Library.cpp)
project_link_libraries(PUBLIC tsd_device)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_MACRO_NAME "ANARI_TSD_EXPORT")

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
