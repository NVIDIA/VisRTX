#!/usr/bin/env python3
"""Reproduce pinned fixtures; validate everything before publishing outputs."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
import tempfile

from table_header import table_header

HERE = Path(__file__).resolve().parent


def check_source(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{path}: does not match the pinned source bytes")


def publish_outputs(outputs):
    # Stage every output before replacing any destination. Each replacement is
    # atomic, but filesystem failure during replacement is not a multi-file
    # transaction. Validation/read/staging failures leave destinations untouched.
    staged = []
    try:
        for path, data in outputs.items():
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
            fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
            temporary = Path(name)
            staged.append((temporary, path))
            with os.fdopen(fd, "wb") as output:
                output.write(data)
            temporary.chmod(mode)
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("pbrt", type=Path)
    p.add_argument("joe_kuo", type=Path)
    p.add_argument("--table", type=Path, help="write production table after oracle agreement")
    p.add_argument("--pbrt-license", type=Path)
    p.add_argument("--joe-license", type=Path)
    p.add_argument("--fast-owen-source", type=Path)
    args = p.parse_args(argv)
    for path, digest in [
        (args.pbrt, "98be104d68f0a90b4043e457b4288d7fb5b3650aae4ec60d72edc07a8435fef6"),
        (args.joe_kuo, "68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441"),
        (args.fast_owen_source, "9b4084174a078713fc756ee14fd303d55a7950cc25543c0194144079e140f16d"),
    ]:
        if path:
            try:
                check_source(path, digest)
            except ValueError as error:
                p.error(str(error))
    here = HERE
    outputs = {}
    source = args.pbrt.read_text()
    outputs[here / "NOTICE"] = (
        source.split("#include", 1)[0]
        + "\nVisRTX modification: extract rows 0..15 and columns 0..31 from stride 52;\n"
        + "adapt FastOwenScrambler naming/annotations, retaining integer operations.\n").encode("utf-8")
    body = source.split("SobolMatrices32[", 1)[1].split("{", 1)[1].split("}", 1)[0]
    words = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", body)]
    columns = [words[d * 52:d * 52 + 32] for d in range(16)]
    # Source parameters, not production columns. Retain original header/spacing.
    fixture = ("# S. Joe and F. Y. Kuo, new-joe-kuo-6.21201\n"
               "# joe-kuo/sobol_data 835a7d7b1ee3bc83e575e302a985c66ec4b65249\n"
               "# Extract: header and dimensions 2..16. See LICENSE.joe-kuo.\n").encode()
    fixture += b"".join(args.joe_kuo.read_bytes().splitlines(keepends=True)[:16])
    assert b"\r" not in fixture
    outputs[here / "joe-kuo.txt"] = fixture
    oracle = [[1 << (31 - i) for i in range(32)]]
    for line in fixture.decode().splitlines()[4:]:
        d, s, a, *m = map(int, line.split())
        assert d == len(oracle) + 1 and len(m) == s
        v = [m[i] << (31 - i) for i in range(s)]
        for i in range(s, 32):
            x = v[i-s] ^ (v[i-s] >> s)
            for k in range(1, s):
                if a & (1 << (s-1-k)):
                    x ^= v[i-k]
            v.append(x)
        oracle.append(v)
    mismatches = [(d, i, columns[d][i], oracle[d][i])
                  for d in range(16) for i in range(32)
                  if columns[d][i] != oracle[d][i]]
    for d, i, actual, expected in mismatches:
        print(f"dimension {d}, column {i}: pbrt={actual:#010x}, Joe-Kuo={expected:#010x}")
    if mismatches:
        raise SystemExit("STOP: pinned references disagree; investigate before changing references")
    print("512 independently regenerated columns agree")
    print("table SHA-256:", hashlib.sha256(b"".join(struct.pack("<I", x) for row in columns for x in row)).hexdigest())
    print("fixture SHA-256:", hashlib.sha256(fixture).hexdigest())
    for path, name in [(args.pbrt_license, "LICENSE.pbrt"),
                       (args.joe_license, "LICENSE.joe-kuo")]:
        if path:
            outputs[here / name] = path.read_bytes()
    if args.fast_owen_source:
        text = args.fast_owen_source.read_text()
        start = text.index("// FastOwenScrambler Definition")
        end = text.index("// OwenScrambler Definition")
        excerpt = text[:text.index("#ifndef")] if "#ifndef" in text else text.split("#include", 1)[0]
        outputs[here / "FastOwenScrambler.reference.txt"] = (excerpt + text[start:end]).encode("utf-8")
        # Independent arbitrary-precision Python rendition of the pinned operations;
        # mask every arithmetic operation to uint32, reverse by string, not bit tricks.
        mask = (1 << 32) - 1
        def reverse(v):
            return int(f"{v:032b}"[::-1], 2)
        def scramble(v, seed):
            v = reverse(v)
            v ^= (v * 0x3d20adea) & mask
            v = (v + seed) & mask
            v = (v * ((seed >> 16) | 1)) & mask
            v ^= (v * 0x05526c56) & mask
            v ^= (v * 0x53a22864) & mask
            return reverse(v)
        indices = [0, 1, 2, 3, 15, 16, 31, 32, 0x80000000, 0xffffffff]
        vectors = "# pbrt/Joe-Kuo binary-index vectors. See README and LICENSE.pbrt.\n"
        vectors += "# index dimension pixelSeed raw reversed scrambled grid24\n"
        for index in indices:
            for d, row in enumerate(columns):
                raw = 0
                for bit in range(32):
                    if index & (1 << bit):
                        raw ^= row[bit]
                for seed in [0, 0x12345678, 0xffffffff]:
                    dim_seed = seed ^ ((0x9e3779b9 * (d + 1)) & mask)
                    value = scramble(raw, dim_seed)
                    vectors += " ".join(f"{x:08x}" for x in
                                        [index, d, seed, raw, reverse(raw), value, value >> 8]) + "\n"
        outputs[here / "vectors.txt"] = vectors.encode("utf-8")
    if args.table:
        notice = source.split("#include", 1)[0]
        outputs[args.table] = table_header(notice, columns).encode("utf-8")

    publish_outputs(outputs)


if __name__ == "__main__":
    main()
