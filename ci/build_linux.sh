#!/bin/bash

mkdir build
cd build

cmake \
  -DVISRTX_PRECOMPILE_SHADERS=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=$CXX \
  ..

cmake --build . -j `nproc`
