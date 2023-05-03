// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <array>
#include <unordered_map>
#include <tuple>
#include "shader_compile_segmented.h"

namespace visgl{

constexpr int SHADER_SEGMENTS = 63;

struct AppendableShader {
  virtual void append(const char *segment) = 0;
  virtual const char* const* source()const = 0;
};

template<int N>
struct StaticAppendableShader : public AppendableShader {
  std::array<const char*, N+1> segments = {};
  int length = 0;
  void append(const char *segment) override {
    if(length<N) {
      segments[length] = segment;
    }
    length += 1;
  }
  const char* const* source() const override {
    return segments.data();
  }
};

template<int N>
bool operator==(const StaticAppendableShader<N> &a, const StaticAppendableShader<N> &b) {
  for(int i=0;i<N+1;++i) {
    if(a.segments[i] != b.segments[i]) {
      return false;
    } else if(a.segments[i] == 0) { // implies a.segments[i] == 0
      return true;
    }
  }
  return true;
}

}

template<int N>
struct std::hash<visgl::StaticAppendableShader<N>> {
  static const size_t C = 486187739;
  size_t operator()(visgl::StaticAppendableShader<N> const& s) const noexcept
  {
    size_t h = 0;
    for(int i=0;s.segments[i] != 0;++i) {
      size_t x = reinterpret_cast<uintptr_t>(s.segments[i]);
      h ^= x;
      h *= C;
    }
    return h;
  }
};

template<int N>
struct std::hash<std::tuple<visgl::StaticAppendableShader<N>,visgl::StaticAppendableShader<N>>> {
  static const size_t C = 486187739;
  size_t operator()(std::tuple<visgl::StaticAppendableShader<N>,visgl::StaticAppendableShader<N>> const& s) const noexcept
  {
    std::hash<visgl::StaticAppendableShader<N>> h;
    return (C*h(std::get<0>(s)))^h(std::get<1>(s));
  }
};

template<int N>
struct std::hash<std::tuple<visgl::StaticAppendableShader<N>,visgl::StaticAppendableShader<N>,visgl::StaticAppendableShader<N>>> {
  static const size_t C = 486187739;
  size_t operator()(std::tuple<visgl::StaticAppendableShader<N>,visgl::StaticAppendableShader<N>,visgl::StaticAppendableShader<N>> const& s) const noexcept
  {
    std::hash<visgl::StaticAppendableShader<N>> h;
    return C*(C*h(std::get<0>(s)))^h(std::get<1>(s))^h(std::get<2>(s));
  }
};

namespace visgl{

template<int N, typename G>
struct ShaderCache {
  G *gl;
  using Key = std::tuple<
    StaticAppendableShader<N>,
    StaticAppendableShader<N>,
    StaticAppendableShader<N>
  >;

  std::unordered_map<Key, GLuint> map;
  void init(G *gl) {
    this->gl = gl;
  }
  GLuint get(const StaticAppendableShader<N> &vs) {
    StaticAppendableShader<N> gs;
    StaticAppendableShader<N> fs;
    Key k(vs, gs, fs);
    auto iter = map.find(k);
    if(iter != map.end()) {
      return iter->second;
    } else {
      GLuint shader = shader_build_graphics_segmented(*gl,
        vs.source(),
        nullptr, nullptr, nullptr, nullptr);

      map.emplace(k, shader);
      return shader;
    }
  }
  GLuint get(const StaticAppendableShader<N> &vs, const StaticAppendableShader<N> &fs) {
    StaticAppendableShader<N> gs;
    Key k(vs, gs, fs);
    auto iter = map.find(k);
    if(iter != map.end()) {
      return iter->second;
    } else {
      GLuint shader = shader_build_graphics_segmented(*gl,
        vs.source(),
        nullptr, nullptr, nullptr,
        fs.source());

      map.emplace(k, shader);
      return shader;
    }
  }
  GLuint get(const StaticAppendableShader<N> &vs, const StaticAppendableShader<N> &gs, const StaticAppendableShader<N> &fs) {
    Key k(vs, gs, fs);
    auto iter = map.find(k);
    if(iter != map.end()) {
      return iter->second;
    } else {
      GLuint shader = shader_build_graphics_segmented(*gl,
        vs.source(),
        nullptr, nullptr,
        gs.source(),
        fs.source());

      map.emplace(k, shader);
      return shader;
    }
  }
  void release() {
    for(auto &x : map) {
      gl->DeleteProgram(x.second);
    }
  }
};

}
