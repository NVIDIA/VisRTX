// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace visrtx::libmdl {

// 128-bit FNV-1a over the bytes of `data`, rendered as 32 lowercase hex chars.
// Implemented with two 64-bit lanes rather than `__int128` so it compiles on
// MSVC as well as GCC/Clang.
inline std::string fnv1a128Hex(std::string_view data)
{
  struct U128
  {
    std::uint64_t hi, lo;
  };

  // 128-bit multiply modulo 2^128. The low lanes need a full 64x64 -> 128
  // product (computed via 32-bit halves, no compiler intrinsics); the cross
  // lanes only reach the high 64 bits and the 2^128 term vanishes.
  auto mul128 = [](U128 a, U128 b) -> U128 {
    std::uint64_t al = a.lo & 0xffffffffu, ah = a.lo >> 32;
    std::uint64_t bl = b.lo & 0xffffffffu, bh = b.lo >> 32;
    std::uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    std::uint64_t mid = (ll >> 32) + (lh & 0xffffffffu) + (hl & 0xffffffffu);
    std::uint64_t lo = (ll & 0xffffffffu) | (mid << 32);
    std::uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    hi += a.lo * b.hi + a.hi * b.lo;
    return U128{hi, lo};
  };

  // FNV-1a 128-bit parameters.
  const U128 prime{0x0000000001000000ull, 0x000000000000013bull};
  U128 hash{0x6c62272e07bb0142ull, 0x62b821756295c58dull};
  for (unsigned char c : data) {
    hash.lo ^= c;
    hash = mul128(hash, prime);
  }

  auto laneToHex = [](std::uint64_t v) {
    static const char *digits = "0123456789abcdef";
    std::string s(16, '0');
    for (int i = 15; i >= 0; --i) {
      s[i] = digits[v & 0xfu];
      v >>= 4;
    }
    return s;
  };

  return laneToHex(hash.hi) + laneToHex(hash.lo);
}

inline bool endsWith(std::string_view s, std::string_view suffix)
{
  return s.size() >= suffix.size()
      && s.substr(s.size() - suffix.size()) == suffix;
}

// True for things the MDL loader should resolve as a file rather than a
// fully-qualified module name.
inline bool looksLikeModulePath(std::string_view s)
{
  return s.find('/') != std::string_view::npos || endsWith(s, ".mdl")
      || endsWith(s, ".mdle");
}

// Ensure a fully-qualified MDL module name: prepend "::" for a bare relative
// name, but leave absolute names ("::foo") and file paths untouched.
inline std::string normalizeModuleName(std::string_view name)
{
  if (name.size() >= 2 && name[0] == ':' && name[1] == ':')
    return std::string(name);
  if (looksLikeModulePath(name))
    return std::string(name);
  return "::" + std::string(name);
}

// Synthetic, content-addressed module name for an inline source string.
inline std::string makeInlineModuleName(std::string_view source)
{
  return "::__visrtx_inline_" + fnv1a128Hex(source);
}

} // namespace visrtx::libmdl
