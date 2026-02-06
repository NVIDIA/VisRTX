## Copyright 2025-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

import struct
import sys
from pye57 import E57
import numpy as np

def extract_points(e57_path, output_path):
    # Load the E57 file
    e57 = E57(e57_path)

    all_points = []
    all_colors = []

    print(f"scan_count: {e57.scan_count}")

    for i in range(e57.scan_count):
        header = e57.get_header(i)
        print(f"Scan: {i}/{e57.scan_count}, point in this scan: {header.point_count}")
        scan = e57.read_scan(i, colors=True, ignore_missing_fields=True)

        assert isinstance(scan["cartesianX"], np.ndarray)
        assert isinstance(scan["cartesianY"], np.ndarray)
        assert isinstance(scan["cartesianZ"], np.ndarray)
        assert isinstance(scan["colorRed"], np.ndarray)
        assert isinstance(scan["colorGreen"], np.ndarray)
        assert isinstance(scan["colorBlue"], np.ndarray)

        xs, ys, zs = scan['cartesianX'], scan['cartesianY'], scan['cartesianZ']
        rs, gs, bs = scan['colorRed'], scan['colorGreen'], scan['colorBlue']
        for x, y, z in zip(xs, ys, zs):
            if x is not None and y is not None and z is not None:
                all_points.append((x, y, z))
        for r, g, b in zip(rs, gs, bs):
            if r is None:
                r = 1.0
            if g is None:
                g = 0.0
            if b is None:
                b = 0.0
            all_colors.append((r, g, b))

    print(f"Total points: {len(all_points)}")

    # Write to binary file
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', len(all_points)))  # uint64 little-endian
        for x, y, z in all_points:
            f.write(struct.pack('<fff', x, y, z)) # 3 float32 values
        for r, g, b in all_colors:
            f.write(struct.pack('<fff', r, g, b)) # 3 float32 values

    print(f"Binary file written to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python e57_to_binary.py <input.e57> <output.bin>")
        sys.exit(1)

    extract_points(sys.argv[1], sys.argv[2])
