// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <anari/anari_cpp.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace visrtx::fogtest {

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
using UVec2 = std::array<unsigned, 2>;

/* Every mapped channel of one rendered Frame. */
struct Image
{
  std::vector<Vec4> color;
  std::vector<float> depth;
  std::vector<Vec3> normal, albedo;
  std::vector<unsigned> primitive, object, instance;
};

/* Owns one public ANARI reference; the Device must outlive this scope. */
class ObjectOwner
{
 public:
  ObjectOwner(anari::Device device, anari::Object object) noexcept;
  ~ObjectOwner();
  ObjectOwner(const ObjectOwner &) = delete;
  ObjectOwner &operator=(const ObjectOwner &) = delete;
  ObjectOwner(ObjectOwner &&) = delete;
  ObjectOwner &operator=(ObjectOwner &&) = delete;

  // Transfer the reference to a caller or Scene only after setup succeeds.
  void release() noexcept;

 private:
  anari::Device m_device;
  anari::Object m_object;
};

/* Keeps a frame mapping alive through validation and allocating a copy. */
template <typename T>
class FrameMapping
{
 public:
  FrameMapping(anari::Device device, anari::Frame frame, const char *channel);
  ~FrameMapping();
  FrameMapping(const FrameMapping &) = delete;
  FrameMapping &operator=(const FrameMapping &) = delete;
  FrameMapping(FrameMapping &&) = delete;
  FrameMapping &operator=(FrameMapping &&) = delete;

  const T *checkedData(
      unsigned width, unsigned height, ANARIDataType type) const;

 private:
  anari::Device m_device;
  anari::Frame m_frame;
  const char *m_channel;
  anari::MappedFrameData<T> m_mapping;
};

/* Array edits must unmap even when fixture setup throws. */
template <typename T>
class ArrayMapping
{
 public:
  ArrayMapping(anari::Device device, anari::Array array);
  ~ArrayMapping();
  ArrayMapping(const ArrayMapping &) = delete;
  ArrayMapping &operator=(const ArrayMapping &) = delete;
  ArrayMapping(ArrayMapping &&) = delete;
  ArrayMapping &operator=(ArrayMapping &&) = delete;

  T *data() const;

 private:
  anari::Device m_device;
  anari::Array m_array;
  T *m_data;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline ObjectOwner::ObjectOwner(
    anari::Device device, anari::Object object) noexcept
    : m_device(device), m_object(object)
{}

inline ObjectOwner::~ObjectOwner()
{
  if (m_object)
    anari::release(m_device, m_object);
}

inline void ObjectOwner::release() noexcept
{
  m_object = nullptr;
}

template <typename T>
FrameMapping<T>::FrameMapping(
    anari::Device device, anari::Frame frame, const char *channel)
    : m_device(device),
      m_frame(frame),
      m_channel(channel),
      m_mapping(anari::map<T>(device, frame, channel))
{}

template <typename T>
FrameMapping<T>::~FrameMapping()
{
  anari::unmap(m_device, m_frame, m_channel);
}

template <typename T>
const T *FrameMapping<T>::checkedData(
    unsigned width, unsigned height, ANARIDataType type) const
{
  if (!m_mapping.data || m_mapping.width != width || m_mapping.height != height
      || m_mapping.pixelType != type)
    throw std::runtime_error(
        std::string("Invalid mapped channel ") + m_channel);
  return m_mapping.data;
}

template <typename T>
ArrayMapping<T>::ArrayMapping(anari::Device device, anari::Array array)
    : m_device(device), m_array(array), m_data(anari::map<T>(device, array))
{}

template <typename T>
ArrayMapping<T>::~ArrayMapping()
{
  anari::unmap(m_device, m_array);
}

template <typename T>
T *ArrayMapping<T>::data() const
{
  if (!m_data)
    throw std::runtime_error("Invalid mapped array");
  return m_data;
}

template <typename T>
std::vector<T> readChannel(anari::Device device,
    anari::Frame frame,
    const char *name,
    unsigned width,
    unsigned height,
    ANARIDataType type = anari::ANARITypeFor<T>::value)
{
  const FrameMapping<T> mapping(device, frame, name);
  const auto *data = mapping.checkedData(width, height, type);
  return {data, data + std::size_t(width) * height};
}

// Status callbacks: an ANARI error is a test failure, never a skipped case.
inline void exitOnErrorStatus(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "ANARI error: %s\n", message);
    std::exit(1);
  }
  if (severity == ANARI_SEVERITY_WARNING)
    fprintf(stderr, "ANARI warning: %s\n", message);
}

// `user` is a std::vector<std::string>* collecting warnings for assertions.
inline void collectWarningsStatus(const void *user,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "ANARI error: %s\n", message);
    std::exit(1);
  }
  if (severity == ANARI_SEVERITY_WARNING)
    static_cast<std::vector<std::string> *>(const_cast<void *>(user))
        ->emplace_back(message);
}

template <typename T>
std::vector<T> channel(
    anari::Device device, anari::Frame frame, const char *name, UVec2 size)
{
  return readChannel<T>(device, frame, name, size[0], size[1]);
}

inline Image readImage(anari::Device device, anari::Frame frame, UVec2 size)
{
  return {channel<Vec4>(device, frame, "channel.color", size),
      channel<float>(device, frame, "channel.depth", size),
      channel<Vec3>(device, frame, "channel.normal", size),
      channel<Vec3>(device, frame, "channel.albedo", size),
      channel<unsigned>(device, frame, "channel.primitiveId", size),
      channel<unsigned>(device, frame, "channel.objectId", size),
      channel<unsigned>(device, frame, "channel.instanceId", size)};
}

// Fresh accumulation after every Commit pairs identical camera/random samples,
// so alpha and the auxiliary channels must compare exactly, not statistically.
inline bool sameAuxiliary(const Image &a, const Image &b, const char *label)
{
  bool same = a.color.size() == b.color.size() && a.depth == b.depth
      && a.normal == b.normal && a.albedo == b.albedo
      && a.primitive == b.primitive && a.object == b.object
      && a.instance == b.instance;
  for (std::size_t p = 0; same && p < a.color.size(); ++p)
    same = a.color[p][3] == b.color[p][3];
  if (!same)
    fprintf(stderr, "%s changed alpha or auxiliary channels\n", label);
  return same;
}

// Independent specification reference in world units (no production helpers).
inline double referenceVisibility(
    const char *mode, double distance, double start, double end, double density)
{
  const std::string curve(mode);
  if (curve == "none")
    return 1.;
  if (curve == "linear")
    return 1.
        - std::fmin(1., std::fmax(0., (distance - start) / (end - start)));
  const long double optical = density * distance;
  return double(std::exp(curve == "exp" ? -optical : -optical * optical));
}

inline double referenceFraction(
    const char *mode, double distance, double start, double end, double density)
{
  return 1. - referenceVisibility(mode, distance, start, end, density);
}

// A missing advertised subtype is a release failure, never a skipped render.
inline int requireRendererFogSupport(anari::Device device)
{
  int failures = 0;
  for (const char *subtype : {"fast", "interactive", "default", "quality"}) {
    const auto *extensions =
        static_cast<const char *const *>(anariGetObjectInfo(
            device, ANARI_RENDERER, subtype, "extension", ANARI_STRING_LIST));
    bool found = false;
    if (extensions)
      for (auto e = extensions; *e; ++e)
        found |= std::strcmp(*e, "ANARI_VISRTX_RENDERER_FOG") == 0;
    if (!found) {
      fprintf(stderr, "required fog capability missing: %s\n", subtype);
      ++failures;
    }
  }
  return failures;
}

} // namespace visrtx::fogtest
