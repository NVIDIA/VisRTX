# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Runs TestMdlPtxFingerprint under 1 and 8 MDL compile threads and asserts the
# printed PTX fingerprints match: parallel compilation must not alter generated
# code (ADR 0009). Driven by CMake so it is toolchain-agnostic.

function(run_fingerprint threads out_var)
  set(ENV{VISRTX_MDL_COMPILE_THREADS} "${threads}")
  execute_process(
    COMMAND "${TEST_BIN}"
    OUTPUT_VARIABLE output
    RESULT_VARIABLE code)
  if (NOT code EQUAL 0)
    message(FATAL_ERROR "fingerprint run (threads=${threads}) failed: ${code}\n${output}")
  endif()
  if (NOT output MATCHES "mdlPtxFingerprint=([0-9a-f]+)")
    message(FATAL_ERROR "no fingerprint in output (threads=${threads}):\n${output}")
  endif()
  set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

run_fingerprint(1 fp_serial)
run_fingerprint(8 fp_parallel)

message(STATUS "serial=${fp_serial} parallel=${fp_parallel}")
if (NOT fp_serial STREQUAL fp_parallel)
  message(FATAL_ERROR
    "PTX fingerprint differs serial vs parallel: ${fp_serial} != ${fp_parallel} "
    "-- parallel compilation produced different code (silent miscompile).")
endif()
