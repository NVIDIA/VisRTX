# VisRTX Devices

![VisRTX Teaser](teaser.png)

This repository contains multiple implementations of the [Khronos ANARI
standard](https://www.khronos.org/anari) developed by the HPC Visualization
Developer Technology team at NVIDIA, primarily for scalable, scientific
visualizations.

The following ANARI devices are available:

- [RTX device](devices/rtx/) based on OptiX
- [OpenGL device](devices/gl/) (experimental)

For any new feature requests or bugs found in extensions that are implemented,
do not hesitate to [open an issue](https://github.com/NVIDIA/VisRTX/issues/new)!

### Sample Applications

The testing/demo applications which used to live in this repository (TSD) now
live in the [Vela](https://github.com/NVIDIA/Vela) repository, which provides
an interactive collection of applications for loading and interacting with
ANARI scenes.

## Build + Install

[VisRTX](devices/rtx/) and [VisGL](devices/gl/) are supported on both Linux and
Windows.

Each device can be built stand alone (separately invoking CMake on their
respective subdirectories), or as a combined build (invoking CMake on the
repository's root directory).

Please refer to each device's README for more details.

The devices are installable to `CMAKE_INSTALL_PREFIX`.
