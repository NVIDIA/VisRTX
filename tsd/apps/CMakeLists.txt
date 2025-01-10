## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (BUILD_INTERACTIVE_APPS)
  add_subdirectory(interactive)
endif()
add_subdirectory(simple)
add_subdirectory(tools)
