## Copyright 2025-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

import struct
import sys
from pxr import Usd, UsdGeom, Vt, Gf

def main(usd_file, instancer_path, output_file):
    stage = Usd.Stage.Open(usd_file)
    if not stage:
        sys.exit(f"Error: could not open USD file '{usd_file}'")

    prim = stage.GetPrimAtPath(instancer_path)
    if not prim:
        sys.exit(f"Error: no prim found at path '{instancer_path}'")

    positions = prim.GetAttribute("positions")
    time_samples = positions.GetTimeSamples()

    #for time in time_samples:
    #    for i, p in enumerate(positions.Get(time)):
    #        print(f"[t, {time}][{i}]: {p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}")

    # Write to binary file
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<Q', len(time_samples)))     # uint64 little-endian
        f.write(struct.pack('<Q', len(positions.Get(0)))) # uint64 little-endian
        for time in time_samples:
            for p in positions.Get(time):
                f.write(struct.pack('<fff', p[0], p[1], p[2])) # 3 float32 values

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_usd_points_to_axyz.py <usd_file> <instancer_path> <output_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

