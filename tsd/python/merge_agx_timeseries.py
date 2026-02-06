## Copyright 2025-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

r"""
Merge AGX files organized as {prefix}_{timestep}.{proc}.agx
into a single AGX file with all processors merged per timestep.

Usage:
  python merge_agx_timeseries.py \
    --input "path/to/iso_lambda2_*.*.agx" \
    --output merged.agx \
    --pattern "iso_lambda2_(\d+)\.(\d+)\.agx"
"""

import argparse
import glob
import re
import sys
from pathlib import Path
from collections import defaultdict
import struct

# AGX file format reader/writer
class AGXReader:
    """Simple AGX file reader for merging purposes."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.header = {}
        self.subtype = ""
        self.constants = {}
        self.timesteps = []

    def __enter__(self):
        try:
            self.file = open(self.filepath, 'rb')
            self._read_header()
        except Exception as e:
            if self.file:
                self.file.close()
                self.file = None
            # Re-raise so caller can handle
            raise
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()

    def _read_header(self):
        """Read AGX header."""
        magic = self.file.read(4)
        if len(magic) < 4:
            raise ValueError(f"File too short or empty: {self.filepath}")
        if magic != b'AGXB':
            raise ValueError(f"Invalid magic number (not an AGX file): {self.filepath}")

        # Read header with native endianness first
        version, endian_marker, obj_type, time_steps, const_count = \
            struct.unpack('=IIIII', self.file.read(20))

        # Check if we need to swap bytes
        if endian_marker == 0x01020304:
            self.endian = '='  # Native (correct)
        elif endian_marker == 0x04030201:
            self.endian = '>' if sys.byteorder == 'little' else '<'
            # Swap the values we just read
            version = struct.unpack(self.endian + 'I', struct.pack('=I', version))[0]
            obj_type = struct.unpack(self.endian + 'I', struct.pack('=I', obj_type))[0]
            time_steps = struct.unpack(self.endian + 'I', struct.pack('=I', time_steps))[0]
            const_count = struct.unpack(self.endian + 'I', struct.pack('=I', const_count))[0]
        else:
            raise ValueError(f"Invalid endian marker: {hex(endian_marker)}")

        self.header = {
            'version': version,
            'endian_marker': endian_marker,
            'object_type': obj_type,
            'time_steps': time_steps,
            'const_count': const_count
        }

        # Read subtype
        subtype_len = struct.unpack(self.endian + 'I', self.file.read(4))[0]
        if subtype_len > 0:
            self.subtype = self.file.read(subtype_len).decode('utf-8')

    def read_all(self):
        """Read all constants and timesteps."""
        # Read constants
        for _ in range(self.header['const_count']):
            name, param = self._read_param()
            self.constants[name] = param

        # Read timesteps
        for _ in range(self.header['time_steps']):
            step_idx, param_count = struct.unpack(self.endian + 'II', self.file.read(8))
            timestep_params = {}
            for _ in range(param_count):
                name, param = self._read_param()
                timestep_params[name] = param
            self.timesteps.append((step_idx, timestep_params))

        return self.constants, self.timesteps

    def _read_param(self):
        """Read a parameter record."""
        name_len = struct.unpack(self.endian + 'I', self.file.read(4))[0]
        name = self.file.read(name_len).decode('utf-8')
        is_array = struct.unpack('B', self.file.read(1))[0]

        if is_array == 0:
            # Single value
            dtype, value_bytes = struct.unpack(self.endian + 'II', self.file.read(8))
            data = self.file.read(value_bytes)
            return name, {'is_array': False, 'type': dtype, 'data': data}
        else:
            # Array - need to read 4 + 8 + 8 = 20 bytes
            elem_type = struct.unpack(self.endian + 'I', self.file.read(4))[0]
            elem_count = struct.unpack(self.endian + 'Q', self.file.read(8))[0]
            data_bytes = struct.unpack(self.endian + 'Q', self.file.read(8))[0]
            data = self.file.read(data_bytes)
            return name, {
                'is_array': True,
                'element_type': elem_type,
                'element_count': elem_count,
                'data': data
            }


class AGXWriter:
    """Simple AGX file writer."""

    def __init__(self, filepath, subtype, object_type=5):  # 5 = ANARI_GEOMETRY
        self.filepath = filepath
        self.subtype = subtype
        self.object_type = object_type
        self.constants = {}
        self.timesteps = []

    def add_constant(self, name, param):
        """Add a constant parameter."""
        self.constants[name] = param

    def add_timestep(self, step_idx, params):
        """Add a timestep with its parameters."""
        self.timesteps.append((step_idx, params))

    def write(self):
        """Write the AGX file."""
        with open(self.filepath, 'wb') as f:
            # Header
            f.write(b'AGXB')
            f.write(struct.pack('I', 1))  # version
            f.write(struct.pack('I', 0x01020304))  # endian marker
            f.write(struct.pack('I', self.object_type))
            f.write(struct.pack('I', len(self.timesteps)))
            f.write(struct.pack('I', len(self.constants)))

            # Subtype
            subtype_bytes = self.subtype.encode('utf-8')
            f.write(struct.pack('I', len(subtype_bytes)))
            f.write(subtype_bytes)

            # Constants
            for name, param in self.constants.items():
                self._write_param(f, name, param)

            # Timesteps
            for step_idx, params in self.timesteps:
                f.write(struct.pack('I', step_idx))
                f.write(struct.pack('I', len(params)))
                for name, param in params.items():
                    self._write_param(f, name, param)

    def _write_param(self, f, name, param):
        """Write a parameter record."""
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('I', len(name_bytes)))
        f.write(name_bytes)

        if not param['is_array']:
            f.write(struct.pack('B', 0))
            f.write(struct.pack('I', param['type']))
            f.write(struct.pack('I', len(param['data'])))
            f.write(param['data'])
        else:
            f.write(struct.pack('B', 1))
            f.write(struct.pack('I', param['element_type']))
            f.write(struct.pack('Q', param['element_count']))
            f.write(struct.pack('Q', len(param['data'])))
            f.write(param['data'])


def parse_filename(filepath, pattern):
    """Extract timestep and processor from filename."""
    match = re.search(pattern, Path(filepath).name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def merge_geometry_arrays(arrays, array_type, vertex_counts=None, debug=False):
    """
    Merge multiple geometry arrays into one.
    For positions: concatenate
    For indices: concatenate with vertex offset
    """
    if not arrays:
        return None

    if array_type == 'vertex.position':
        # Simple concatenation for positions
        merged_data = b''.join(arr['data'] for arr in arrays)
        total_count = sum(arr['element_count'] for arr in arrays)
        return {
            'is_array': True,
            'element_type': arrays[0]['element_type'],
            'element_count': total_count,
            'data': merged_data
        }

    elif array_type == 'primitive.index':
        # Need to offset indices for each processor
        import numpy as np

        if vertex_counts is None or len(vertex_counts) != len(arrays):
            # Fallback: just concatenate without offsetting (will be wrong but won't crash)
            print(f"Warning: Cannot offset indices - vertex count mismatch (counts={len(vertex_counts) if vertex_counts else 0}, arrays={len(arrays)})")
            merged_data = b''.join(arr['data'] for arr in arrays)
            total_count = sum(arr['element_count'] for arr in arrays)
            return {
                'is_array': True,
                'element_type': arrays[0]['element_type'],
                'element_count': total_count,
                'data': merged_data
            }

        elem_type = arrays[0]['element_type']
        # Assuming ANARI_UINT32_VEC3 (most common for triangle indices)
        dtype = np.uint32
        components = 3  # triangle

        merged_indices = []
        vertex_offset = 0

        if debug:
            print(f"  Offsetting indices for {len(arrays)} processors:")

        for i, arr in enumerate(arrays):
            # Get vertex count from corresponding positions
            vertex_count = vertex_counts[i]

            # Parse indices
            indices = np.frombuffer(arr['data'], dtype=dtype)
            if len(indices) == 0:
                # Empty geometry for this processor
                if debug and i < 10:
                    print(f"    Proc {i}: 0 tris (empty), vertices={vertex_count}, skipping")
                vertex_offset += vertex_count
                continue

            indices = indices.reshape(-1, components)

            orig_min = indices.min()
            orig_max = indices.max()

            # Check if indices are local (0-based) or global
            # If orig_max >= vertex_count, they're likely global indices
            if debug and i < 10:
                is_local = orig_max < vertex_count
                print(f"    Proc {i}: {len(indices)} tris, vertices={vertex_count}, orig_range=[{orig_min},{orig_max}], is_local={is_local}")

            # Offset indices
            indices = indices + vertex_offset

            if debug and i < 10:
                print(f"            offset={vertex_offset}, new_range=[{indices.min()},{indices.max()}]")

            merged_indices.append(indices)
            vertex_offset += vertex_count

        if not merged_indices:
            if debug:
                print(f"  Warning: No geometry data to merge (all processors empty)")
            return None

        merged = np.vstack(merged_indices)
        if debug:
            print(f"  Total merged: {len(merged)} triangles, index_range=[{merged.min()},{merged.max()}]")
        return {
            'is_array': True,
            'element_type': elem_type,
            'element_count': len(merged),
            'data': merged.tobytes()
        }
    else:
        # For other arrays (like vertex attributes), just concatenate
        # These should be per-vertex and in the same order as positions
        merged_data = b''.join(arr['data'] for arr in arrays)
        total_count = sum(arr['element_count'] for arr in arrays)

        if debug and array_type.startswith('vertex.'):
            print(f"  Concatenating {array_type}: {len(arrays)} arrays -> {total_count} elements")

        return {
            'is_array': True,
            'element_type': arrays[0]['element_type'],
            'element_count': total_count,
            'data': merged_data
        }


def merge_agx_files(input_files, output_file, pattern):
    """Merge AGX files into a single animated file."""

    print(f"Found {len(input_files)} AGX files to merge")

    # Track corrupted files
    corrupted_files = []

    # Parse and group files by timestep
    timestep_files = defaultdict(dict)  # {timestep: {proc: filepath}}

    for filepath in input_files:
        timestep, proc = parse_filename(filepath, pattern)
        if timestep is None or proc is None:
            print(f"Warning: Skipping file with unexpected name: {filepath}")
            continue
        timestep_files[timestep][proc] = filepath

    if not timestep_files:
        print("Error: No valid files found matching pattern")
        return False

    timesteps = sorted(timestep_files.keys())
    print(f"Found {len(timesteps)} timesteps: {min(timesteps)} to {max(timesteps)}")
    print(f"Timestep values: {timesteps[:10]}{'...' if len(timesteps) > 10 else ''}")

    # Read first file to get subtype and structure
    first_file = timestep_files[timesteps[0]][min(timestep_files[timesteps[0]].keys())]
    print(f"Reading template from: {first_file}")

    with AGXReader(first_file) as reader:
        constants, _ = reader.read_all()
        subtype = reader.subtype
        object_type = reader.header['object_type']

    print(f"Geometry subtype: {subtype}")
    print(f"Object type: {object_type}")

    # Create writer
    writer = AGXWriter(output_file, subtype, object_type)

    # Merge constants (assumes all processors have same constant topology)
    # We'll merge them from all processors in first timestep
    print("Merging constant parameters...")

    first_timestep = timesteps[0]
    procs = sorted(timestep_files[first_timestep].keys())

    # Read all files for first timestep to get geometry structure
    # Note: For single-snapshot AGX files, geometry is in timestep data, not constants
    constant_arrays = defaultdict(list)
    vertex_counts = []

    for proc in procs:
        filepath = timestep_files[first_timestep][proc]
        try:
            with AGXReader(filepath) as reader:
                consts, ts_data = reader.read_all()

                # Collect vertex count for this processor
                # Try constants first, then timestep data
                proc_vertex_count = 0
                if 'vertex.position' in consts and consts['vertex.position']['is_array']:
                    proc_vertex_count = consts['vertex.position']['element_count']
                elif ts_data and len(ts_data) > 0:
                    # Check first timestep
                    _, ts_params = ts_data[0]
                    if 'vertex.position' in ts_params and ts_params['vertex.position']['is_array']:
                        proc_vertex_count = ts_params['vertex.position']['element_count']

                vertex_counts.append(proc_vertex_count)

                # Collect constant arrays (like topology if it exists)
                for name, param in consts.items():
                    if param['is_array']:
                        constant_arrays[name].append(param)
        except Exception as e:
            print(f"  WARNING: Failed to read {filepath}: {e}")
            print(f"           Treating processor {proc} as empty")
            corrupted_files.append(filepath)
            # Add empty data for this processor
            vertex_counts.append(0)
            # Add empty arrays to maintain array size consistency
            for name in constant_arrays.keys():
                constant_arrays[name].append({
                    'is_array': True,
                    'element_type': constant_arrays[name][0]['element_type'] if constant_arrays[name] else 0,
                    'element_count': 0,
                    'data': b''
                })

    print(f"Vertex counts per processor: {vertex_counts[:10]}{'...' if len(vertex_counts) > 10 else ''}")
    print(f"Total processors: {len(vertex_counts)}, Total vertices: {sum(vertex_counts)}")

    # Check if topology (indices) are in constants or timesteps
    has_indices_in_constants = 'primitive.index' in constant_arrays

    if has_indices_in_constants:
        print("WARNING: Indices are in constants but geometry varies per timestep!")
        print("         This will cause incorrect merging. Indices should be in timestep data.")
        print("         Skipping constant indices - will use per-timestep indices instead.")
        # Don't merge indices from constants - we'll get them per-timestep
        constant_arrays.pop('primitive.index', None)

    # Only merge constant arrays if they exist (static topology)
    if constant_arrays:
        print("Found constant data:")
        for name, arrays in constant_arrays.items():
            merged = merge_geometry_arrays(arrays, name, vertex_counts, debug=True)
            if merged:
                writer.add_constant(name, merged)
                print(f"  - {name}: {merged['element_count']} elements")
    else:
        print("No constant data found (all geometry is in timesteps)")

    # Process each timestep
    print(f"\nMerging {len(timesteps)} timesteps...")
    for ts_idx, timestep in enumerate(timesteps):
        print(f"  Output timestep {ts_idx} (source timestep {timestep})")
        procs = sorted(timestep_files[timestep].keys())

        # Read time-varying parameters from all processors
        timestep_arrays = defaultdict(list)
        failed_procs = []

        for proc in procs:
            filepath = timestep_files[timestep][proc]
            try:
                with AGXReader(filepath) as reader:
                    consts, ts_data = reader.read_all()

                    # For single-snapshot files, we need BOTH constants (indices) and timestep data (positions)
                    # Merge them together for this timestep
                    all_params = {}

                    # Get constants (includes indices)
                    for name, param in consts.items():
                        if param['is_array']:
                            all_params[name] = param

                    # Get timestep data (includes positions, may override constants)
                    if ts_data and len(ts_data) > 0:
                        _, params = ts_data[0]
                        for name, param in params.items():
                            if param['is_array']:
                                all_params[name] = param

                    # Add to timestep arrays
                    for name, param in all_params.items():
                        timestep_arrays[name].append(param)
            except Exception as e:
                if ts_idx == 0:  # Only warn on first timestep to avoid spam
                    print(f"    WARNING: Failed to read {filepath}: {e}")
                    print(f"             Treating processor {proc} as empty for this timestep")
                corrupted_files.append(filepath)
                failed_procs.append(proc)

                # Add empty data for this processor to maintain consistency
                # Use the first successful processor's keys as template
                if timestep_arrays:
                    for name in timestep_arrays.keys():
                        # Add empty array with matching type
                        if timestep_arrays[name]:
                            timestep_arrays[name].append({
                                'is_array': True,
                                'element_type': timestep_arrays[name][0]['element_type'],
                                'element_count': 0,
                                'data': b''
                            })

        if failed_procs and ts_idx == 0:
            print(f"    Skipped {len(failed_procs)} corrupted/empty processors")

        # Build vertex counts for THIS specific timestep
        # IMPORTANT: Counts can vary per timestep (processors may be empty at different times)
        ts_vertex_counts = []
        if 'vertex.position' in timestep_arrays:
            for arr in timestep_arrays['vertex.position']:
                ts_vertex_counts.append(arr['element_count'])
        else:
            # Fallback: use counts from first timestep
            ts_vertex_counts = vertex_counts

        # Debug output for first few timesteps
        if ts_idx < 3:
            print(f"    Vertex counts for this timestep: {ts_vertex_counts[:10]}{'...' if len(ts_vertex_counts) > 10 else ''}")
            print(f"    Total vertices in this timestep: {sum(ts_vertex_counts)}")

        # Merge timestep arrays
        merged_params = {}
        for name, arrays in timestep_arrays.items():
            # Enable debug output for first few timesteps
            merged = merge_geometry_arrays(arrays, name, ts_vertex_counts, debug=(ts_idx < 2))
            if merged:
                merged_params[name] = merged
                if ts_idx < 3:  # Show details for first few timesteps
                    print(f"    Merged {name}: {merged['element_count']} elements")

        # Validate and show attribute values
        if ts_idx < 3 and 'vertex.position' in merged_params:
            pos_count = merged_params['vertex.position']['element_count']

            # Check all vertex.attribute* parameters
            for attr_name in [k for k in merged_params.keys() if k.startswith('vertex.attribute')]:
                attr_count = merged_params[attr_name]['element_count']
                if pos_count == attr_count:
                    print(f"    ✓ {attr_name}: {attr_count} values match vertex count")

                    # Show min/max values to verify they're changing
                    import numpy as np
                    dtype = np.float32  # Assuming ANARI_FLOAT32
                    values = np.frombuffer(merged_params[attr_name]['data'], dtype=dtype)
                    if len(values) > 0:
                        print(f"      Value range: [{values.min():.6f}, {values.max():.6f}]")
                else:
                    print(f"    ⚠ WARNING: {attr_name} count ({attr_count}) != vertex count ({pos_count})!")

        # IMPORTANT: Use ts_idx (0, 1, 2...) not timestep (4320, 4321...) for output
        # AGX animations expect sequential timestep indices starting from 0
        writer.add_timestep(ts_idx, merged_params)

    # Write output
    print(f"\nWriting merged file: {output_file}")
    print(f"Summary:")
    print(f"  - Timesteps: {len(writer.timesteps)}")
    print(f"  - Constants: {list(writer.constants.keys())}")
    if writer.timesteps:
        first_ts_params = list(writer.timesteps[0][1].keys())
        print(f"  - Per-timestep parameters: {first_ts_params}")

        # Check if attributes are in timestep data
        attr_params = [p for p in first_ts_params if 'attribute' in p]
        if attr_params:
            print(f"  - ✓ Attributes in timestep data: {attr_params}")
        else:
            print(f"  - ⚠ WARNING: No attributes found in timestep data!")

    writer.write()

    # Report corrupted files
    if corrupted_files:
        unique_corrupted = list(set(corrupted_files))
        print(f"\n⚠ WARNING: Encountered {len(unique_corrupted)} corrupted/unreadable files:")
        for f in unique_corrupted[:10]:
            print(f"  - {f}")
        if len(unique_corrupted) > 10:
            print(f"  ... and {len(unique_corrupted) - 10} more")
        print(f"\nThese files were treated as empty and skipped.")

    print("Done!")

    # Verify the written file by reading it back
    print("\nVerifying written file...")
    try:
        with AGXReader(output_file) as verify_reader:
            consts, ts_data = verify_reader.read_all()
            print(f"Read back: {len(ts_data)} timesteps")

            # Check first timestep
            if ts_data:
                ts_idx, params = ts_data[0]
                print(f"\nFirst timestep (index {ts_idx}):")

                if 'vertex.attribute0' in params:
                    import numpy as np
                    attr_data = params['vertex.attribute0']
                    values = np.frombuffer(attr_data['data'], dtype=np.float32)

                    print(f"  vertex.attribute0: {len(values)} values")
                    print(f"  Range: [{values.min():.6f}, {values.max():.6f}]")
                    print(f"  First 10 values: {values[:10]}")
                    print(f"  Last 10 values: {values[-10:]}")
                    print(f"  Mean: {values.mean():.6f}, Std: {values.std():.6f}")

                    # Check if all values are the same (indicating a problem)
                    unique_count = len(np.unique(values))
                    if unique_count == 1:
                        print(f"  ⚠ WARNING: All values are identical ({values[0]})!")
                    elif unique_count < 10:
                        print(f"  ⚠ WARNING: Only {unique_count} unique values!")
                    else:
                        print(f"  ✓ {unique_count} unique values (good variation)")
                else:
                    print("  ⚠ WARNING: vertex.attribute0 not found in written file!")
    except Exception as e:
        print(f"Error verifying file: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Merge AGX timeseries files')
    parser.add_argument('--input', required=True,
                       help='Input file pattern (e.g., "path/to/iso_*.*.agx")')
    parser.add_argument('--output', required=True,
                       help='Output merged AGX file')
    parser.add_argument('--pattern', default=r'(\d+)\.(\d+)\.agx',
                       help='Regex pattern to extract (timestep, proc) from filename')

    args = parser.parse_args()

    # Expand glob pattern
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"Error: No files found matching pattern: {args.input}")
        return 1

    success = merge_agx_files(input_files, args.output, args.pattern)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

