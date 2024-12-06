## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(anari_library_tsd LANGUAGES CXX)

anari_generate_queries(
  NAME tsd_queries
  PREFIX Device
  CPP_NAMESPACE tsd_device
  JSON_DEFINITIONS_FILE ${ANARI_CODE_GEN_ROOT}/devices/experimental_device.json
)

project_add_library(SHARED
  Device.cpp
  DeviceQueries.cpp
  Frame.cpp
  Library.cpp
  Object.cpp
)
project_link_libraries(PRIVATE tsd anari::helium)
project_compile_definitions(PRIVATE "anari_library_tsd_EXPORTS")
project_include_directories(
PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_MACRO_NAME "ANARI_TSD_EXPORT")

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
