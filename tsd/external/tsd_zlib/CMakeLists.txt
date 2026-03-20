## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

# zlib - compression/decompression library
# Static fallback build when no system zlib is available.

project(tsd_ext_zlib LANGUAGES C)

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
  project_add_library(INTERFACE)
  project_link_libraries(INTERFACE ZLIB::ZLIB)
  return()
endif()

message(STATUS
  "ZLIB not found via find_package — fetching upstream zlib 1.3.1.")
anari_sdk_fetch_project(
  NAME ${PROJECT_NAME}
  URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
  MD5 9855b6d802d7fe5b7bd5b196a2271655
)

# Minimal static build — only the bits the compress/uncompress path needs.
# gzip stream I/O (gz*.c) is intentionally excluded.
project_add_library(STATIC
  ${tsd_ext_zlib_LOCATION}/adler32.c
  ${tsd_ext_zlib_LOCATION}/crc32.c
  ${tsd_ext_zlib_LOCATION}/deflate.c
  ${tsd_ext_zlib_LOCATION}/infback.c
  ${tsd_ext_zlib_LOCATION}/inffast.c
  ${tsd_ext_zlib_LOCATION}/inflate.c
  ${tsd_ext_zlib_LOCATION}/inftrees.c
  ${tsd_ext_zlib_LOCATION}/trees.c
  ${tsd_ext_zlib_LOCATION}/zutil.c
  ${tsd_ext_zlib_LOCATION}/compress.c
  ${tsd_ext_zlib_LOCATION}/uncompr.c
)

project_include_directories(
  PUBLIC
    $<BUILD_INTERFACE:${tsd_ext_zlib_LOCATION}>
)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Suppress warnings from zlib code
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  project_compile_options(PRIVATE -w)
elseif(MSVC)
  # zlib uses a handful of CRT functions MSVC flags as "unsafe".
  project_compile_options(PRIVATE /W0)
  project_compile_definitions(PRIVATE
    _CRT_SECURE_NO_DEPRECATE
    _CRT_NONSTDC_NO_DEPRECATE
  )
endif()

