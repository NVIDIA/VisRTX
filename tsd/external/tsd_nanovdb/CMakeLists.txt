## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_nanovdb LANGUAGES CXX)
project_add_library(INTERFACE)
project_include_directories(INTERFACE ${CMAKE_CURRENT_LIST_DIR})
if (TSD_USE_TBB)
  find_package(TBB REQUIRED)
  project_link_libraries(INTERFACE TBB::tbb)
  project_compile_definitions(INTERFACE NANOVDB_USE_TBB)
endif()

if (TSD_NANOVDB_USE_ZIP)
  # zlib decompresses ZIP-encoded .nvdb files at read time. ANARI always
  # receives the decompressed in-memory grid buffer — the codec only
  # affects on-disk storage. The tsd FindZLIB.cmake module supplies a
  # fetched static fallback when no system zlib is available.
  project_link_libraries(INTERFACE tsd_ext_zlib)
  project_compile_definitions(INTERFACE NANOVDB_USE_ZIP)
endif()
