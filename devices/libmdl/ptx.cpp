// Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "ptx.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <set>
#include <string>
#include <string_view>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace visrtx::libmdl {

std::vector<char> stitchPTXs(
    nonstd::span<const nonstd::span<const char>> ptxBlobs)
{
  std::vector<std::vector<char>> fixedBlobs;
  fixedBlobs.reserve(size(ptxBlobs));
  std::size_t totalSize = 0;

  // As we are blending multiple separate compilation unit PTXs, make sure
  // headers are not conflicting.
  std::string version;
  std::string target;
  std::string addressSize = ".address_size 64";

  // For disambiguating symbol names
  int uniquifyIdx = 0;
  // For removing latter instances of weak and extern symbols
  std::set<std::string> previouslySeenWeak;
  std::set<std::string> previouslySeenExtern;

  // For disambiguating info strings
  std::uint32_t infoStringBaseIndex = 0;
  // For disambiguating file ids
  std::uint32_t fileBaseIndex = 0;

  for (const auto &s : ptxBlobs) {
    std::vector<char> blob(std::cbegin(s), std::cend(s));

    // Handle version, target and addressSize metadata
    {
      static constexpr const auto versionKw = "\n.version "sv;
      static constexpr const auto targetKw = "\n.target "sv;
      static constexpr const auto addressSizeKw = "\n.address_size "sv;

      for (auto dotkword : {versionKw, targetKw, addressSizeKw}) {
        auto it = std::search(
            begin(blob), end(blob), cbegin(dotkword), cend(dotkword));

        auto eolIt = std::find(++it, end(blob), '\n');
        std::string sub(it, eolIt);

        blob.erase(it, eolIt);

        if (dotkword == versionKw) { // .version
          if (sub > version)
            version = sub;
        } else if (dotkword == targetKw) { //.target
          if (sub > target)
            target = sub;
        }
      }
    }

    {
      // Same cleanup with forward decls possibly conflicting with actual
      // declarations.
      static constexpr const auto externPrefix = "\n.extern .func "sv;
      static constexpr const std::string_view declsToKeep[] = {"vprintf"sv};

      for (auto it = std::search(begin(blob),
               end(blob),
               cbegin(externPrefix),
               cend(externPrefix));
          it != end(blob);) {
        auto semiColonIt = std::find(it, end(blob), ';');

        if (semiColonIt != end(blob)) {
          for (const auto &decl : declsToKeep) {
            if (std::search(it, semiColonIt, cbegin(decl), cend(decl))
                != semiColonIt) {
              // This is a decl that might need to be kept. Check if we
              // already did so.
              std::string decl(it, semiColonIt);

              if (previouslySeenExtern.insert(decl).second) {
                // First time. Skip it and move on.
                it = semiColonIt;
              } // If not the first occurrence, then it will be erased.
              break;
            }
          }

          it = blob.erase(it, ++semiColonIt);
        }
        it = std::search(
            it, end(blob), cbegin(externPrefix), cend(externPrefix));
      }
    }

    // And ditto with colliding symbol names (focusing on constant strings for
    // now).
    {
      std::string prefix = "uniquify_" + std::to_string(uniquifyIdx++);
      static const std::string strPrefix = "$str";
      for (auto it = std::search(
               begin(blob), end(blob), cbegin(strPrefix), cend(strPrefix));
          it != end(blob);) {
        // Insertion might invalidate the iterator. Make sure to get the after
        // insertion value.
        it = blob.insert(++it, cbegin(prefix), cend(prefix));
        // And search for next occurrence from there. Note that current $str has
        // been broken to $_something_str and is not a match anymore for the
        // search, so we can resume at current position.
        it = std::search(it, end(blob), cbegin(strPrefix), cend(strPrefix));
      }
    }

    // Remove .visible qualifier from of all functions but OptiX semantic entry
    // points.
    {
      static constexpr std::string_view dotVisible = "\n.visible ";
      static constexpr std::string_view directCallableSemantic =
          "__direct_callable__";

      for (auto it = std::search(
               begin(blob), end(blob), cbegin(dotVisible), cend(dotVisible));
          it != end(blob);) {
        it = next(it); // Skip the new line.
        auto eolIt = std::find(it, end(blob), '\n');
        if (std::search(it,
                eolIt,
                cbegin(directCallableSemantic),
                cend(directCallableSemantic))
            == eolIt) {
          it = blob.erase(it, it + size(dotVisible) - 1);
        }
        // And search for next occurrence from there. Note that current $str has
        // been broken to $_something_str and is not a match anymore for the
        // search, so we can resume at current position.
        it = std::search(it, end(blob), cbegin(dotVisible), cend(dotVisible));
      }
    }

    // Next remove duplicates weak symbols
    {
      static constexpr const std::string_view weakGlobalPrefix[] = {
          ".weak .global"sv,
          ".weak .const"sv,
      };
      for (const auto &prefix : weakGlobalPrefix) {
        for (auto it = std::search(
                 begin(blob), end(blob), cbegin(prefix), cend(prefix));
            it != end(blob);) {
          auto equalSignIt = std::find(it, end(blob), '=');
          if (equalSignIt == end(blob))
            break;

          std::string decl(it, equalSignIt);
          if (previouslySeenWeak.insert(decl).second) {
            it = equalSignIt;
          } else {
            // Remove duplicate.
            auto eraseTillIt = std::find(it, end(blob), '\n');
            if (eraseTillIt != end(blob))
              ++eraseTillIt;
            it = blob.erase(it, eraseTillIt);
          }

          it = std::search(it, end(blob), cbegin(prefix), cend(prefix));
        }
      }
    }

    // Only needed when we build the cuda code with -lineinfo or -G
    // We don't know that here, so we proceed anyway.
    // Ensure no duplicate info string for debug symbols
    {
      static constexpr const auto prefix = "$L__info_string"sv;
      for (auto it = std::search(
               begin(blob), end(blob), cbegin(prefix), cend(prefix));
          it != end(blob);
          it = std::search(it, end(blob), cbegin(prefix), cend(prefix))) {
        it += size(prefix);
        auto indexEndIt = std::find_if(
            it, end(blob), [](char c) { return !std::isdigit(c); });
        if (indexEndIt == end(blob))
          break;

        auto indexStr = std::string(it, indexEndIt);
        auto index = std::stoi(indexStr);

        int newIndex;
        if (*indexEndIt == ':') {
          newIndex = infoStringBaseIndex++;
        } else {
          newIndex = index + infoStringBaseIndex;
        }

        if (newIndex == index)
          continue;

        // Replace the index with the new one.
        auto newIndexStr = std::to_string(newIndex);
        if (size(newIndexStr) == size(indexStr)) {
          // Same content, just copy the values
          // Returned it points after the last copied element.
          it = std::copy(cbegin(newIndexStr), cend(newIndexStr), it);
        } else {
          // Returned it points after the last removed element.
          // Try and avoid calling erase + insert to move the data only once.
          assert(size(newIndexStr) > size(indexStr));
          it = std::copy_n(cbegin(newIndexStr), size(indexStr), it);
          it = blob.insert(
              it, cbegin(newIndexStr) + size(indexStr), cend(newIndexStr));
        }
      }
    }

    // Ensure no duplicate file ids for debug symbols
    {
      static constexpr const auto dotFilePrefix = ".file"sv;
      static constexpr const auto dotLocPrefix = ".loc"sv;
      static constexpr const auto inlinedAtPrefix = " inlined_at"sv;

      // That's the slow code path of the stitching... Let's try and minimize
      // the amount of traversal we need to do. Let's go with 3 linear
      // traversal, one for each of the prefixes. WARNING: dotFilePrefix needs
      // to be last, as it is the one that will compute the new file base index.
      for (auto dotkword : {inlinedAtPrefix, dotLocPrefix, dotFilePrefix}) {
        for (auto it = std::search(
                 begin(blob), end(blob), cbegin(dotkword), cend(dotkword));
            it != end(blob);
            it = std::search(it, end(blob), cbegin(dotkword), cend(dotkword))) {
          // Eat spaces prior to the actual index
          it += size(dotkword);
          if (!std::isspace(*it)) {
            continue;
          }
          while (std::isspace(*it))
            ++it;

          // And go till the end of the current number
          auto indexEndIt = it + 1;
          while (std::isdigit(*indexEndIt))
            ++indexEndIt;

          if (indexEndIt == end(blob))
            break;

          auto indexStr = std::string(it, indexEndIt);
          auto index = std::stoi(indexStr);

          int newIndex;
          if (dotkword == dotFilePrefix) {
            newIndex = ++fileBaseIndex; // File indices starts at one, hence the
                                        // pre-increment.
          } else {
            newIndex = index + fileBaseIndex;
          }

          if (newIndex == index) {
            continue;
          }

          // Replace the index with the new one.
          auto newIndexStr = std::to_string(newIndex);
          if (size(newIndexStr) == size(indexStr)) {
            // Same content, just copy the values
            // Returned it points after the last copied element.
            it = std::copy(cbegin(newIndexStr), cend(newIndexStr), it);
          } else {
            // Returned it points after the last removed element.
            // Try and avoid calling erase + insert to move the data only once.
            assert(size(newIndexStr) > size(indexStr));
            it = std::copy_n(cbegin(newIndexStr), size(indexStr), it);
            it = blob.insert(
                it, cbegin(newIndexStr) + size(indexStr), cend(newIndexStr));
          }
        }
      }
    }

    fixedBlobs.push_back(std::move(blob));
    totalSize += size(s);
  }

  // Finally join all the fixed blobs
  const std::string header =
      "// Generated\n" + version + "\n" + target + "\n" + addressSize + "\n\n";
  std::vector<char> result;
  result.reserve(size(header) + totalSize);
  result.insert(begin(result), cbegin(header), cend(header));
  for (const auto &blob : fixedBlobs) {
    result.insert(end(result), std::cbegin(blob), std::cend(blob));
  }

  return result;
}
} // namespace visrtx::libmdl
