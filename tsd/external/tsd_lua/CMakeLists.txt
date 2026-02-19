## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_lua LANGUAGES C)

anari_sdk_fetch_project(
  NAME ${PROJECT_NAME}
  URL https://www.lua.org/ftp/lua-5.4.8.tar.gz
  MD5 81cf5265b8634967d8a7480d238168ce
)

project_add_library(STATIC
  # Lua source files (core library, no standalone interpreter)
  ${tsd_ext_lua_LOCATION}/src/lapi.c
  ${tsd_ext_lua_LOCATION}/src/lcode.c
  ${tsd_ext_lua_LOCATION}/src/lctype.c
  ${tsd_ext_lua_LOCATION}/src/ldebug.c
  ${tsd_ext_lua_LOCATION}/src/ldo.c
  ${tsd_ext_lua_LOCATION}/src/ldump.c
  ${tsd_ext_lua_LOCATION}/src/lfunc.c
  ${tsd_ext_lua_LOCATION}/src/lgc.c
  ${tsd_ext_lua_LOCATION}/src/llex.c
  ${tsd_ext_lua_LOCATION}/src/lmem.c
  ${tsd_ext_lua_LOCATION}/src/lobject.c
  ${tsd_ext_lua_LOCATION}/src/lopcodes.c
  ${tsd_ext_lua_LOCATION}/src/lparser.c
  ${tsd_ext_lua_LOCATION}/src/lstate.c
  ${tsd_ext_lua_LOCATION}/src/lstring.c
  ${tsd_ext_lua_LOCATION}/src/ltable.c
  ${tsd_ext_lua_LOCATION}/src/ltm.c
  ${tsd_ext_lua_LOCATION}/src/lundump.c
  ${tsd_ext_lua_LOCATION}/src/lvm.c
  ${tsd_ext_lua_LOCATION}/src/lzio.c
  # Lua auxiliary library
  ${tsd_ext_lua_LOCATION}/src/lauxlib.c
  ${tsd_ext_lua_LOCATION}/src/lbaselib.c
  ${tsd_ext_lua_LOCATION}/src/lcorolib.c
  ${tsd_ext_lua_LOCATION}/src/ldblib.c
  ${tsd_ext_lua_LOCATION}/src/liolib.c
  ${tsd_ext_lua_LOCATION}/src/lmathlib.c
  ${tsd_ext_lua_LOCATION}/src/loadlib.c
  ${tsd_ext_lua_LOCATION}/src/loslib.c
  ${tsd_ext_lua_LOCATION}/src/lstrlib.c
  ${tsd_ext_lua_LOCATION}/src/ltablib.c
  ${tsd_ext_lua_LOCATION}/src/lutf8lib.c
  ${tsd_ext_lua_LOCATION}/src/linit.c
)

project_include_directories(PUBLIC ${tsd_ext_lua_LOCATION}/src)

# Platform-specific definitions
if(UNIX AND NOT APPLE)
  project_compile_definitions(PRIVATE LUA_USE_LINUX)
  project_link_libraries(PRIVATE m dl)
elseif(APPLE)
  project_compile_definitions(PRIVATE LUA_USE_MACOSX)
  project_link_libraries(PRIVATE m)
elseif(WIN32)
  project_compile_definitions(PRIVATE LUA_USE_WINDOWS)
endif()

# Suppress warnings from Lua code
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  project_compile_options(PRIVATE -w)
elseif(MSVC)
  project_compile_options(PRIVATE /W0)
endif()
