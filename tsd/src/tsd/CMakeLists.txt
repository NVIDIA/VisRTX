## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

# Always present components
add_subdirectory(app)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(rendering)

# Optional components
if (TSD_USE_MPI)
  add_subdirectory(mpi)
endif()
if (TSD_USE_NETWORKING)
  add_subdirectory(network)
endif()
if (TSD_BUILD_UI_LIBRARY)
  add_subdirectory(ui)
endif()
