## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(demos)
add_subdirectory(multiDeviceViewer)
add_subdirectory(viewer)
if (TSD_USE_MPI)
  add_subdirectory(mpiViewer)
endif()
