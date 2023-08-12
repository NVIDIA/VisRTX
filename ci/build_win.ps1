md build
cd build

cmake -DVISRTX_PRECOMPILE_SHADERS=OFF -DVISRTX_BUILD_GL_DEVICE=OFF ..
cmake --build . -j --config Release --target ALL_BUILD

exit $LASTEXITCODE
