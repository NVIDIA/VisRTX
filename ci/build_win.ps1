md build
cd build

cmake -DVISRTX_PRECOMPILE_SHADERS=OFF ..
cmake --build . -j --config Release --target ALL_BUILD

exit $LASTEXITCODE
