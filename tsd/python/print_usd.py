## Copyright 2025-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

from pxr import Usd
import sys

def print_attributes(prim, indent="  "):
    for attr in prim.GetAttributes():
        # Get the attribute's value (at default time code)
        value = attr.Get()
        print(f"{indent}- {attr.GetName()} ({attr.GetTypeName()}): {value}")

def traverse(stage, prim=None, indent=""):
    if prim is None:
        prim = stage.GetPseudoRoot()

    for child in prim.GetChildren():
        print(f"{indent}Prim: {child.GetPath()} (type: {child.GetTypeName()})")
        print_attributes(child, indent + "  ")
        traverse(stage, child, indent + "  ")

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_usd.py <file.usd>")
        sys.exit(1)

    usd_file = sys.argv[1]
    stage = Usd.Stage.Open(usd_file)

    if not stage:
        print(f"Failed to open USD file: {usd_file}")
        sys.exit(1)

    traverse(stage)

if __name__ == "__main__":
    main()
