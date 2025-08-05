## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(app)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(rendering)
if (BUILD_INTERACTIVE_APPS)
  add_subdirectory(ui)
endif()
