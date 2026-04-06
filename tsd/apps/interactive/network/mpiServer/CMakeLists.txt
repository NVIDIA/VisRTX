## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdMPIServer)
project_add_executable(DistributedRenderServer.cpp mpiServer.cpp)
project_link_libraries(tsd_app tsd_network tsd_mpi)
