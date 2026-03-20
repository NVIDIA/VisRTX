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
  # zlib is required at read time to decompress ZIP-encoded .nvdb files.
  # ANARI consumes the decompressed in-memory grid buffer, so the codec
  # only affects on-disk storage, not what crosses the ANARI boundary.
  find_package(ZLIB QUIET)
  if (NOT TARGET ZLIB::ZLIB)
    message(STATUS
      "ZLIB not found via find_package — fetching upstream zlib 1.3.1.")
    anari_sdk_fetch_project(
      NAME tsd_ext_zlib
      URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
      MD5 9855b6d802d7fe5b7bd5b196a2271655
    )

    # Build a minimal static zlib — only the bits NanoVDB's compress/uncompress
    # path needs. gzip stream I/O (gz*.c) is intentionally excluded.
    add_library(tsd_ext_zlib STATIC
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
    target_include_directories(tsd_ext_zlib
      PUBLIC $<BUILD_INTERFACE:${tsd_ext_zlib_LOCATION}>
    )
    set_target_properties(tsd_ext_zlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(tsd_ext_zlib PRIVATE -w)
    elseif (MSVC)
      target_compile_options(tsd_ext_zlib PRIVATE /W0
        # zlib uses a handful of CRT functions MSVC flags as "unsafe".
        /D_CRT_SECURE_NO_DEPRECATE /D_CRT_NONSTDC_NO_DEPRECATE
      )
    endif()

    add_library(ZLIB::ZLIB ALIAS tsd_ext_zlib)
  endif()
  project_link_libraries(INTERFACE ZLIB::ZLIB)
  project_compile_definitions(INTERFACE NANOVDB_USE_ZIP)
endif()
