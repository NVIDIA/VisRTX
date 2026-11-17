// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "mdl/SceneDataInstance.h"

#include <cstdio>
#include <future>
#include <initializer_list>
#include <stdexcept>
#include <utility>

using visrtx::MaterialAttribute;
using visrtx::libmdl::SceneDataDescriptor;
using visrtx::mdl::SceneDataInstance;
using NamedBinding = SceneDataInstance::NamedBinding;

namespace {

void require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

bool update(
    SceneDataInstance &instance, std::initializer_list<NamedBinding> bindings)
{
  return instance.update({bindings.begin(), bindings.size()});
}

SceneDataDescriptor descriptor()
{
  SceneDataDescriptor result;
  result.m_namesById = {
      "", "temperature", "", "tint", "temperature.uniform", "Temperature"};
  return result;
}

void testSnapshots()
{
  visrtx::SceneDataNames names;
  SceneDataInstance instance(descriptor(), names);
  require(instance.data().size() == 6, "string IDs were compacted");
  for (const auto &binding : instance.data())
    require(binding.attribute == MaterialAttribute::UNKNOWN,
        "initial binding was not empty");
  require(!update(instance, {}), "empty snapshot changed an unbound table");
  require(!update(instance,
              {{"", MaterialAttribute::COLOR},
                  {"missing", MaterialAttribute::COLOR}}),
      "empty or unknown name changed the table");

  require(update(instance,
              {{"temperature", MaterialAttribute::ATTRIB_0},
                  {"tint", MaterialAttribute::COLOR},
                  {"temperature.uniform", MaterialAttribute::ATTRIB_2},
                  {"Temperature", MaterialAttribute::ATTRIB_3}}),
      "initial bindings were not applied");
  require(instance.data()[1].attribute == MaterialAttribute::ATTRIB_0,
      "temperature resolved to the wrong ID");
  require(instance.data()[2].attribute == MaterialAttribute::UNKNOWN,
      "reserved slot acquired a binding");
  require(instance.data()[3].attribute == MaterialAttribute::COLOR,
      "color binding was lost");
  require(instance.data()[4].attribute == MaterialAttribute::ATTRIB_2,
      "dotted name was treated as metadata");
  require(instance.data()[5].attribute == MaterialAttribute::ATTRIB_3,
      "case-sensitive names were merged");
  require(!update(instance,
              {{"Temperature", MaterialAttribute::ATTRIB_3},
                  {"temperature.uniform", MaterialAttribute::ATTRIB_2},
                  {"tint", MaterialAttribute::COLOR},
                  {"temperature", MaterialAttribute::ATTRIB_0},
                  {"unused", MaterialAttribute::COLOR}}),
      "equivalent snapshot reported a change");

  require(update(instance, {{"temperature", MaterialAttribute::ATTRIB_1}}),
      "rebind/unset did not change the table");
  require(instance.data()[1].attribute == MaterialAttribute::ATTRIB_1,
      "rebinding failed");
  require(instance.data()[3].attribute == MaterialAttribute::UNKNOWN,
      "omitted binding was retained");
  require(update(instance, {}), "unsetting all bindings reported no change");
  require(!update(instance, {}), "repeated empty snapshot reported a change");
}

void testAttributeWhitelist()
{
  visrtx::SceneDataNames names;
  SceneDataInstance instance(descriptor(), names);
  for (auto attribute : {MaterialAttribute::ATTRIB_0,
           MaterialAttribute::ATTRIB_1,
           MaterialAttribute::ATTRIB_2,
           MaterialAttribute::ATTRIB_3,
           MaterialAttribute::COLOR}) {
    update(instance, {{"temperature", attribute}, {"tint", attribute}});
    require(instance.data()[1].attribute == attribute
            && instance.data()[3].attribute == attribute,
        "supported attribute or alias was rejected");
  }
  for (auto attribute : {MaterialAttribute::OBJECT_POSITION,
           MaterialAttribute::OBJECT_NORMAL,
           MaterialAttribute::WORLD_POSITION,
           MaterialAttribute::WORLD_NORMAL,
           MaterialAttribute::UNKNOWN,
           static_cast<MaterialAttribute>(255)}) {
    update(instance, {{"temperature", MaterialAttribute::ATTRIB_0}});
    require(update(instance, {{"temperature", attribute}}),
        "invalid attribute did not clear an existing binding");
    require(instance.data()[1].attribute == MaterialAttribute::UNKNOWN,
        "unsupported attribute was accepted");
  }
  update(instance,
      {{"temperature", MaterialAttribute::ATTRIB_0},
          {"temperature", MaterialAttribute::COLOR}});
  require(instance.data()[1].attribute == MaterialAttribute::COLOR,
      "last duplicate binding did not win");
  update(instance,
      {{"temperature", MaterialAttribute::COLOR},
          {"temperature", MaterialAttribute::WORLD_NORMAL}});
  require(instance.data()[1].attribute == MaterialAttribute::UNKNOWN,
      "invalid duplicate retained the earlier binding");
}

void testOwnershipAndSourceChanges()
{
  visrtx::SceneDataNames names;
  auto source = descriptor();
  SceneDataInstance first(source, names);
  source.m_namesById[1] = "replacement";
  require(update(first, {{"temperature", MaterialAttribute::ATTRIB_0}}),
      "instance borrowed the source descriptor");

  auto copy = first;
  update(first, {});
  require(copy.data()[1].attribute == MaterialAttribute::ATTRIB_0,
      "instance copies share mutable bindings");

  // Equal table sizes do not imply equal names or IDs.
  std::swap(source.m_namesById[1], source.m_namesById[3]);
  SceneDataInstance second(source, names);
  update(second,
      {{"temperature", MaterialAttribute::ATTRIB_0},
          {"tint", MaterialAttribute::COLOR}});
  require(second.data()[1].attribute == MaterialAttribute::COLOR,
      "replacement descriptor did not resolve its own IDs");
  require(second.data()[3].attribute == MaterialAttribute::UNKNOWN,
      "old source binding leaked into new descriptor");

  SceneDataDescriptor large;
  large.m_namesById.resize(65);
  large.m_namesById[64] = "last";
  SceneDataInstance highIds(large, names);
  require(update(highIds, {{"last", MaterialAttribute::COLOR}})
          && highIds.data()[64].attribute == MaterialAttribute::COLOR,
      "high string ID was truncated");

  SceneDataInstance empty(SceneDataDescriptor{}, names);
  require(empty.data().empty(), "empty descriptor produced binding entries");
  require(!update(empty, {{"unused", MaterialAttribute::COLOR}}),
      "unknown name changed an empty table");
}

void testGeometryTokens()
{
  visrtx::SceneDataNames names;
  SceneDataInstance first(descriptor(), names);
  auto other = descriptor();
  std::swap(other.m_namesById[1], other.m_namesById[3]);
  SceneDataInstance second(other, names);
  require(first.data()[1].nameToken != 0
          && first.data()[1].nameToken == second.data()[3].nameToken,
      "the same name acquired different tokens across target codes");
  require(first.data()[0].nameToken == 0 && first.data()[2].nameToken == 0,
      "empty string acquired a valid token");
  require(first.data()[1].nameToken != first.data()[5].nameToken,
      "case-sensitive names acquired the same token");
  const auto token = first.data()[1].nameToken;
  const visrtx::MDLSceneDataBinding aliases[] = {
      {MaterialAttribute::COLOR, names.intern("unrelated")},
      {MaterialAttribute::ATTRIB_2, token}};
  update(first, {{"temperature", MaterialAttribute::ATTRIB_0}});
  require(visrtx::resolveSceneDataAttribute(first.data()[1], aliases, 2)
          == MaterialAttribute::ATTRIB_2,
      "geometry alias did not override the material binding");
  require(visrtx::resolveSceneDataAttribute(second.data()[3], aliases, 2)
          == MaterialAttribute::ATTRIB_2,
      "geometry alias required an explicit material binding");
  require(visrtx::resolveSceneDataAttribute(first.data()[1], nullptr, 0)
          == MaterialAttribute::ATTRIB_0,
      "empty geometry aliases lost material fallback");
  update(first, {});
  require(first.data()[1].nameToken == token
          && visrtx::resolveSceneDataAttribute(first.data()[1], aliases, 2)
              == MaterialAttribute::ATTRIB_2,
      "unsetting material bindings lost geometry name identity");
  visrtx::SceneDataNames independent;
  independent.intern("extra");
  require(independent.intern("temperature") != token
          && names.intern("temperature") == token,
      "different devices share name-token state");

  std::vector<std::future<uint32_t>> jobs;
  for (int worker = 0; worker < 8; ++worker) {
    jobs.push_back(std::async(std::launch::async, [&names, worker] {
      for (int i = 0; i < 1000; ++i)
        names.intern(std::to_string(worker) + "." + std::to_string(i));
      return names.intern("temperature");
    }));
  }
  for (auto &job : jobs)
    require(job.get() == token, "concurrent interning changed string identity");
}

} // namespace

int main()
{
  try {
    testSnapshots();
    testAttributeWhitelist();
    testOwnershipAndSourceChanges();
    testGeometryTokens();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  std::puts("MDL scene-data instance checks passed");
  return 0;
}
