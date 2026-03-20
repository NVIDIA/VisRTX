// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace pbrt {

enum class TokenType
{
  Identifier, // Bare word: Shape, Material, WorldBegin, etc.
  String, // Quoted: "trianglemesh", "float fov"
  Number, // Numeric literal: 42, -3.14, 1e-5
  LBracket, // [
  RBracket, // ]
  Eof
};

struct Token
{
  TokenType type = TokenType::Eof;
  std::string text;
  size_t line = 0;
};

// Stream-based lexer. Constructed from file contents (string_view).
// Call next() repeatedly to consume tokens until Eof.
class Lexer
{
 public:
  explicit Lexer(std::string_view source, const std::string &filename = "");

  Token next();
  Token peek();

  const std::string &currentFile() const;
  size_t currentLine() const;

 private:
  void skipWhitespaceAndComments();
  Token readString();
  Token readNumber();
  Token readIdentifier();

  std::string_view m_source;
  size_t m_pos = 0;
  size_t m_line = 1;
  std::string m_filename;

  bool m_hasPeeked = false;
  Token m_peeked;
};

} // namespace pbrt
