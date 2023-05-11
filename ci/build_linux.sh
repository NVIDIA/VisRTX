#!/bin/bash

mkdir build
cd build

cmake \
  -DVISRTX_PRECOMPILE_SHADERS=OFF \
  -DVISRTX_BUILD_GL_DEVICE=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=$CXX \
  ..

cmake --build . -j `nproc`
