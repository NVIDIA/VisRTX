// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/Any.hpp"
#include "tsd/core/DataTree.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// std
#include <array>
#include <string>
#include <string_view>
#include <vector>

// Internal helpers shared between the object-file (export_Object) and
// layer-subtree (export_LayerSubtree) serialization paths. These build a
// self-consistent closure of the objects reachable from a set of seed objects,
// rewrite object references to dense local indices for storage, and recreate
// those objects (with references remapped to fresh Scene indices) on import.

namespace tsd::io::detail {

using tsd::core::Any;
using namespace tsd::scene;

// Closure / file bookkeeping /////////////////////////////////////////////////

struct ObjectKey
{
  anari::DataType type{ANARI_UNKNOWN};
  size_t index{tsd::core::INVALID_INDEX};
};

// An object selected for export, paired with the dense local index it will be
// stored at within its pool.
struct ClosureEntry
{
  ObjectKey source;
  anari::DataType objectType{ANARI_UNKNOWN};
  size_t localIndex{tsd::core::INVALID_INDEX};
  Object *object{nullptr};
};

// An object node found in a loaded file's objectDB, keyed by its file-local
// (type, index).
struct FileObjectEntry
{
  ObjectKey file;
  anari::DataType objectType{ANARI_UNKNOWN};
  core::DataNode *node{nullptr};
};

// Maps a file-local object key to the Scene object created for it on import.
struct TargetObjectEntry
{
  ObjectKey file;
  Any target;
};

// Per-pool running counters used to assign dense local indices on export.
struct PoolCounters
{
  size_t arrays{0};
  size_t samplers{0};
  size_t materials{0};
  size_t geometries{0};
  size_t surfaces{0};
  size_t fields{0};
  size_t volumes{0};
  size_t lights{0};
};

// Describes which object types may appear in a closure and whether exactly one
// root object (Surface or Volume) is permitted (object-file semantics).
struct ClosurePolicy
{
  std::array<anari::DataType, 8> allowed{};
  size_t numAllowed{0};
  bool singleRoot{false};
  anari::DataType singleRootType{ANARI_UNKNOWN};

  bool contains(anari::DataType canonicalType) const;
};

// Single Surface/Volume root with its supporting sub-objects (export_Object).
ClosurePolicy objectFilePolicy(anari::DataType rootType);
// Broad multi-root set including lights (export_LayerSubtree).
ClosurePolicy layerSubtreePolicy();
// Lights and the arrays they reference only (light-rig subtree export).
ClosurePolicy lightRigPolicy();

// Object pool layout shared by every serialized objectDB.
extern const std::array<const char *, 8> OBJECT_POOL_NAMES;

anari::DataType canonicalObjectType(anari::DataType type);
ObjectKey makeKey(anari::DataType type, size_t index);
ObjectKey makeKey(const Any &value);
bool sameKey(const ObjectKey &a, const ObjectKey &b);
anari::DataType nonArrayTypeForPoolName(std::string_view name);
bool isKnownObjectPoolName(std::string_view name);
bool typeAllowed(const ClosurePolicy &policy, anari::DataType type);
bool poolAllowed(const ClosurePolicy &policy, std::string_view poolName);

ClosureEntry *findClosureEntry(
    std::vector<ClosureEntry> &entries, const ObjectKey &key);
const ClosureEntry *entryForLocalIndex(
    const std::vector<ClosureEntry> &entries,
    anari::DataType type,
    size_t localIndex);
FileObjectEntry *findFileEntry(
    std::vector<FileObjectEntry> &entries, const ObjectKey &key);
const TargetObjectEntry *findTargetEntry(
    const std::vector<TargetObjectEntry> &entries, const ObjectKey &key);

bool hasObjectArrayNode(core::DataNode &node, std::string *message);

// Export side ////////////////////////////////////////////////////////////////

// Admit a single seed object (and recursively the objects it references through
// parameters/metadata) into the closure, honoring the given policy.
bool admitObject(const Scene &scene,
    const ClosurePolicy &policy,
    const ObjectKey &rootKey,
    Object *object,
    std::vector<ClosureEntry> &entries,
    PoolCounters &counters,
    std::string &errorMessage);

// Build the full closure of seed objects, following param/metadata references.
// rootKey is only consulted for single-root policies.
bool buildClosure(const Scene &scene,
    const std::vector<Object *> &seeds,
    const ClosurePolicy &policy,
    const ObjectKey &rootKey,
    std::vector<ClosureEntry> &entries,
    std::string &errorMessage);

// Rewrite object references within a serialized node from Scene indices to the
// dense local indices assigned by the closure.
bool rewriteRefsToLocal(core::DataNode &root,
    const std::vector<ClosureEntry> &entries,
    std::string &errorMessage);

// Emit each closure entry into its objectDB pool, rewriting references.
bool writeObjectDB(core::DataNode &objectDB,
    const std::vector<ClosureEntry> &entries,
    std::string &errorMessage);

// Import side ////////////////////////////////////////////////////////////////

// Validate the __tsd_metadata envelope against an expected fileType and the set
// of accepted/known schemas.
PayloadValidationResult validateEnvelope(core::DataNode &root,
    std::string_view expectedFileType,
    const std::vector<std::string_view> &acceptedSchemas,
    const std::vector<std::string_view> &knownSchemas);

// Collect (and structurally validate) the objectDB pools of a loaded file.
bool collectFileObjects(core::DataNode &objectDB,
    std::vector<FileObjectEntry> &entries,
    PayloadValidationResult &result);

// Verify every reference in the file's objectDB resolves to a collected object,
// that all collected types are permitted, and (for single-root policies) that
// exactly one root exists. seedKeys are the keys reachability starts from.
bool checkGraphConsistency(std::vector<FileObjectEntry> &entries,
    const std::vector<ObjectKey> &seedKeys,
    const ClosurePolicy &policy,
    bool requireAllReachable,
    PayloadValidationResult &result);

Object *createTargetObject(Scene &scene, core::DataNode &node);
void clearObjectPayload(Object &object);
void rollbackCreatedObjects(Scene &scene, const std::vector<Any> &created);

// Rewrite object references within a serialized node from file-local indices to
// the freshly created Scene object indices.
bool rewriteRefsToTarget(core::DataNode &root,
    const std::vector<TargetObjectEntry> &entries,
    std::string &errorMessage);

// Create every object in fileEntries within the Scene, populate it from its
// node (references remapped scene-relative), and record the file->target map.
// Rolls back any created objects on failure.
bool instantiateObjectDB(Scene &scene,
    std::vector<FileObjectEntry> &fileEntries,
    std::vector<TargetObjectEntry> &targetEntries,
    std::vector<Any> &createdRefs,
    std::string &errorMessage);

} // namespace tsd::io::detail
