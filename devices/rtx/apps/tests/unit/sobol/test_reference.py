# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Offline failure-preservation tests for reference fixture publication."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import reference


class ReferencePublicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.outputs = self.root / "output"
        self.outputs.mkdir()
        for name in ("NOTICE", "joe-kuo.txt", "vectors.txt", "sobolDirections.h"):
            (self.outputs / name).write_bytes(b"unchanged " + name.encode())
        self.original = {p.name: p.read_bytes() for p in self.outputs.iterdir()}
        here = Path(__file__).resolve().parent
        # Synthetic inputs exercise publication, not numeric correctness (which
        # has independent tests). Only the source trust gate is stubbed in tests
        # explicitly using accepted_inputs; production validation stays intact.
        header = (here.parents[3] / "device/gpu/sobolDirections.h").read_text()
        words = re.findall(r"0x[0-9a-f]+u", header)
        self.assertEqual(len(words), 512)
        padded = []
        for i in range(16):
            padded += words[i * 32:(i + 1) * 32] + ["0x00000000u"] * 20
        self.source = self.root / "pbrt.cpp"
        self.source.write_text("// synthetic notice\n#include <ignored>\n"
                               "uint32_t SobolMatrices32[] = {" + ",".join(padded) + "};\n")
        self.parameters = self.root / "joe-kuo.txt"
        self.parameters.write_bytes(b"".join(
            (here / "joe-kuo.txt").read_bytes().splitlines(keepends=True)[3:]))
        self.args = [str(self.source), str(self.parameters), "--table",
                     str(self.outputs / "sobolDirections.h")]

    def unchanged(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.outputs.iterdir()},
                         self.original)

    def invoke(self, extra=()):
        with patch.object(reference, "HERE", self.outputs), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            reference.main(self.args + list(extra))

    def test_untrusted_input_leaves_outputs_unchanged(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        self.unchanged()

    def test_oracle_disagreement_leaves_outputs_unchanged(self):
        self.source.write_text(self.source.read_text().replace("0x80000000u", "0x80000001u", 1))
        with patch.object(reference, "check_source"), self.assertRaisesRegex(SystemExit, "STOP"):
            self.invoke()
        self.unchanged()

    def test_late_input_read_failure_leaves_outputs_unchanged(self):
        with patch.object(reference, "check_source"), self.assertRaises(FileNotFoundError):
            self.invoke(["--pbrt-license", str(self.root / "missing-license")])
        self.unchanged()

    def test_staging_failure_removes_temporary_files(self):
        create = tempfile.mkstemp
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated staging failure")
            return create(*args, **kwargs)

        with patch.object(reference, "check_source"), \
                patch.object(reference.tempfile, "mkstemp", side_effect=fail_second), \
                self.assertRaisesRegex(OSError, "staging failure"):
            self.invoke()
        self.unchanged()

    def test_success_publishes_complete_header(self):
        with patch.object(reference, "check_source"):
            self.invoke()
        self.assertEqual(set(p.name for p in self.outputs.iterdir()), set(self.original))
        self.assertIn(b"#pragma once", (self.outputs / "sobolDirections.h").read_bytes())
        self.assertIn(b"synthetic notice", (self.outputs / "NOTICE").read_bytes())
        self.assertNotEqual((self.outputs / "joe-kuo.txt").read_bytes(), self.original["joe-kuo.txt"])
        # Vectors were not requested: leave them byte-for-byte alone.
        self.assertEqual((self.outputs / "vectors.txt").read_bytes(), self.original["vectors.txt"])


if __name__ == "__main__":
    unittest.main()
