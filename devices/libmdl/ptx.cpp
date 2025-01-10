// Copyright 2024 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "ptx.h"

#include <array>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <string_view>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace visrtx::libmdl {

std::vector<char> stitchPTXs(
    nonstd::span<const nonstd::span<const char>> ptxBlobs)
{
  std::vector<char> result;
  for (const auto &s : ptxBlobs) {
    result.insert(end(result), std::cbegin(s), std::cend(s));
  }

  // As we are blending to separate compilation unit PTXs, make sure headers are
  // not conflicting.
  std::string version;
  std::string target;
  std::string addressSize = ".address_size 64";

  {
    // Handle version, target and addressSize metadata
    static constexpr const auto versionKw = "\n.version "sv;
    static constexpr const auto targetKw = "\n.target "sv;
    static constexpr const auto addressSizeKw = "\n.address_size "sv;

    for (auto dotkword : {versionKw, targetKw, addressSizeKw}) {
      for (auto it = std::search(
               begin(result), end(result), cbegin(dotkword), cend(dotkword));
          it != end(result);) {
        auto eolIt = std::find(it + 1, end(result), '\n');
        std::string sub(it + 1, eolIt);
        result.erase(it, eolIt);
        if (dotkword == versionKw) { // .version
          if (sub > version)
            version = sub;
        } else if (dotkword == targetKw) { //.target
          if (sub > target)
            target = sub;
        }
        it = std::search(it, end(result), cbegin(dotkword), cend(dotkword));
      }
    }
  }

  {
    // Same cleanup with forward decls possibly conflicting with actual
    // declarations.
    static constexpr const std::string_view externPrefixes[] = {
        "\n.extern .func "sv,
    };
    static constexpr const std::string_view declsToKeep[] = {"vprintf"sv};
    // For decls we keep, only keep the first occurrence.
    std::set<std::string> previouslySeen;

    for (const auto &externPrefix : externPrefixes) {
      for (auto it = std::search(begin(result),
               end(result),
               cbegin(externPrefix),
               cend(externPrefix));
          it != end(result);) {
        auto semiColonIt = std::find(it, end(result), ';');

        if (semiColonIt != end(result)) {
          for (const auto &decl : declsToKeep) {
            if (std::search(it, semiColonIt, cbegin(decl), cend(decl))
                != semiColonIt) {
              // This is a decl that might need to be kept. Check if we
              // already did so.
              std::string decl(it, semiColonIt);
              if (previouslySeen.find(decl) == cend(previouslySeen)) {
                // First time. Skip it and move on.
                it = semiColonIt;
              } // If not the first occurrence, then it will be erased.
              break;
            }
          }
          if (it != semiColonIt)
            it = result.erase(it, ++semiColonIt);
        }
        it = std::search(
            it, end(result), cbegin(externPrefix), cend(externPrefix));
      }
    }
  }

  {
    // And ditto with colliding symbol names (focusing on constant strings for
    // now).
    int uniquifyIdx = 0;

    std::string prefix = "uniquify_" + std::to_string(uniquifyIdx++);
    static const std::string strPrefix = "$str";
    for (auto it = std::search(
             begin(result), end(result), cbegin(strPrefix), cend(strPrefix));
        it != end(result);) {
      // Insertion might invalidate the iterator. Make sure to get the after
      // insertion value.
      it = result.insert(++it, cbegin(prefix), cend(prefix));
      // And search for next occurrence from there. Note that current $str has
      // been broken to $_something_str and is not a match anymore for the
      // search, so we can resume at current position.
      it = std::search(it, end(result), cbegin(strPrefix), cend(strPrefix));
    }
  }

  // Remove .visible qualifier from of all functions but OptiX semantic entry
  // points.
  {
    static constexpr std::string_view dotVisible = "\n.visible ";
    static constexpr std::string_view directCallableSemantic =
        "__direct_callable__";

    for (auto it = std::search(
             begin(result), end(result), cbegin(dotVisible), cend(dotVisible));
        it != end(result);) {
      it = next(it); // Skip the new line.
      auto eolIt = std::find(it, end(result), '\n');
      if (std::search(it,
              eolIt,
              cbegin(directCallableSemantic),
              cend(directCallableSemantic))
          == eolIt) {
        it = result.erase(it, it + dotVisible.size() - 1);
      }
      // And search for next occurrence from there. Note that current $str has
      // been broken to $_something_str and is not a match anymore for the
      // search, so we can resume at current position.
      it = std::search(it, end(result), cbegin(dotVisible), cend(dotVisible));
    }
  }

  // Last pass: remove duplicates weak symbols
  {
    std::set<std::string> previouslySeen;
    static constexpr const std::string_view weakGlobalPrefix[] = {
        ".weak .global"sv,
        ".weak .const"sv,
    };
    for (const auto &prefix : weakGlobalPrefix) {
      for (auto it = std::search(
               begin(result), end(result), cbegin(prefix), cend(prefix));
          it != end(result);) {
        auto equalSignIt = std::find(it, end(result), '=');
        if (equalSignIt == end(result))
          break;

        std::string decl(it, equalSignIt);
        if (!previouslySeen.insert(decl).second) {
          auto eraseTillIt = std::find(it, end(result), '\n');
          if (eraseTillIt != end(result))
            ++eraseTillIt;
          it = result.erase(it, eraseTillIt);
        } else {
          ++it;
        }

        it = std::search(it, end(result), cbegin(prefix), cend(prefix));
      }
    }
  }

  std::string header =
      "// Generated\n" + version + "\n" + target + "\n" + addressSize + "\n\n";
  result.insert(begin(result), cbegin(header), cend(header));

  {
    std::ofstream ofs("/tmp/shader-new.ptx");
    std::copy(cbegin(result), cend(result), std::ostream_iterator<char>(ofs));
  }

  return result;
}
} // namespace visrtx::libmdl
