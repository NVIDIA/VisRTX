# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Offline byte-for-byte regeneration gate for the production table header."""

from pathlib import Path
import unittest

from table_header import table_header


class TableHeaderTest(unittest.TestCase):
    def test_complete_header_matches(self):
        here = Path(__file__).resolve().parent
        # Retain the upstream notice independently of the production header.
        notice, separator, _ = (here / "NOTICE").read_text(encoding="utf-8").partition(
            "\nVisRTX modification:")
        self.assertTrue(separator)
        # Regenerate columns from the independent parameter fixture, not from
        # the header being compared. Dimension zero is the standard special case.
        columns = [[1 << (31 - i) for i in range(32)]]
        for line in (here / "joe-kuo.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or not line.lstrip()[0].isdigit():
                continue
            dimension, degree, coefficients, *initial = map(int, line.split())
            self.assertEqual(dimension, len(columns) + 1)
            self.assertEqual(len(initial), degree)
            row = [initial[i] << (31 - i) for i in range(degree)]
            for i in range(degree, 32):
                value = row[i - degree] ^ (row[i - degree] >> degree)
                for k in range(1, degree):
                    if coefficients & (1 << (degree - 1 - k)):
                        value ^= row[i - k]
                row.append(value)
            columns.append(row)
        self.assertEqual(len(columns), 16)
        expected = here.parents[3] / "device/gpu/sobolDirections.h"
        self.assertEqual(table_header(notice, columns).encode("utf-8"),
                         expected.read_bytes())


if __name__ == "__main__":
    unittest.main()
