#!/usr/bin/env python3
"""Offline pixel/depth vectors from Joe-Kuo parameters and uint32 arithmetic.

No production headers are read. FastOwen follows the pinned pbrt excerpt in
this directory; see README.md / LICENSE.pbrt and LICENSE.joe-kuo.
"""
from pathlib import Path

here = Path(__file__).parent
mask = (1 << 32) - 1
columns = [[1 << (31 - i) for i in range(32)]]
for line in (here / "joe-kuo.txt").read_text().splitlines()[4:]:
    d, degree, a, *m = map(int, line.split())
    assert d == len(columns) + 1
    row = [n << (31 - i) for i, n in enumerate(m)]
    for i in range(degree, 32):
        value = row[i-degree] ^ (row[i-degree] >> degree)
        for k in range(1, degree):
            if a & (1 << (degree-1-k)):
                value ^= row[i-k]
        row.append(value)
    columns.append(row)


def finalizer(value):
    value ^= value >> 16
    value = value * 0x7feb352d & mask
    value ^= value >> 15
    value = value * 0x846ca68b & mask
    return value ^ (value >> 16)


def reverse(value):
    return int(f"{value:032b}"[::-1], 2)


def scramble(value, seed):
    value = reverse(value)
    value ^= value * 0x3d20adea & mask
    value = (value + seed) & mask
    value = value * ((seed >> 16) | 1) & mask
    value ^= value * 0x05526c56 & mask
    value ^= value * 0x53a22864 & mask
    return reverse(value)


text = "# Offline Joe-Kuo / pinned pbrt FastOwen vectors; see README.md.\n"
text += "# x y scrambleSeed bounce transparency index dimension pixelKey vertexKey grid24\n"
x, y = 1, 2
for seed in [0, 1, 0x12345678, mask]:
    pixel = finalizer((x + y * 0x9e3779b9 + 0xb5297a4d) & mask) ^ finalizer(seed)
    for bounce, transparency in [(0, 0), (1, 0), (0, 1), (3, 7)]:
        vertex = pixel ^ (bounce * 0x9e3779b9 & mask) ^ (transparency * 0x85ebca6b & mask)
        for index in [0, 16, mask]:
            for d, row in enumerate(columns):
                raw = 0
                for bit, column in enumerate(row):
                    if index & (1 << bit):
                        raw ^= column
                grid = scramble(raw, vertex ^ ((d + 1) * 0x9e3779b9 & mask)) >> 8
                text += " ".join(f"{v:08x}" for v in
                                 [x, y, seed, bounce, transparency, index, d, pixel, vertex, grid]) + "\n"
(here / "seed-vectors.txt").write_text(text)
