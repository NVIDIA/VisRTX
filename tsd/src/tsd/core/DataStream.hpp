// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tsd::core {

///////////////////////////////////////////////////////////////////////////////
// Data writers ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct DataWriter
{
  virtual ~DataWriter() = default;

  // Write data to the writer (like std::fwrite)
  virtual size_t write(const void *ptr, size_t size, size_t count) = 0;
};

struct BufferWriter : public DataWriter
{
  explicit BufferWriter(size_t initial_size = 0);

  size_t write(const void *ptr, size_t size, size_t count) override;

  const std::vector<std::byte> &buffer() const;
  std::vector<std::byte> take();
  void clear();

 private:
  std::vector<std::byte> m_buffer;
};

struct FileWriter : public DataWriter
{
  explicit FileWriter(const char *filename, const char *mode = "wb");
  ~FileWriter();

  size_t write(const void *ptr, size_t size, size_t count) override;

  bool valid() const;
  operator bool() const;

 private:
  std::FILE *m_file{nullptr};
};

///////////////////////////////////////////////////////////////////////////////
// Data readers ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct DataReader
{
  virtual ~DataReader() = default;

  // Read data from the reader (like std::fread)
  virtual size_t read(void *ptr, size_t size, size_t count) = 0;
};

struct BufferReader : public DataReader
{
  explicit BufferReader(
      const std::vector<std::byte> &buffer, size_t offset = 0);

  size_t read(void *ptr, size_t size, size_t count) override;

  size_t position() const;
  void reset(size_t offset = 0);

 private:
  const std::vector<std::byte> &m_buffer;
  size_t m_offset{0};
};

struct FileReader : public DataReader
{
  explicit FileReader(const char *filename, const char *mode = "rb");
  ~FileReader();

  size_t read(void *ptr, size_t size, size_t count) override;

  bool valid() const;
  operator bool() const;

 private:
  std::FILE *m_file{nullptr};
};

} // namespace tsd::core
