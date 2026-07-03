# Embed PTX as string resources

Device shaders are compiled to PTX at build time and embedded in the library
as C++ string resources, rather than shipped as files on disk or compiled at
runtime. Embedding keeps the device a self-contained shared library with no
install-path or file-lookup fragility. PTX (rather than OptiX-IR) was chosen
because OptiX-IR did not exist at the time; no switch is planned.

## Consequences

- The driver JIT-compiles PTX at first pipeline creation; shader changes
  require rebuilding the library.
- A future OptiX-IR migration could only be partial: MDL target code is PTX
  (the MDL SDK has no OptiX-IR backend), and those modules would stay PTX in
  otherwise-IR pipelines, which OptiX permits.
