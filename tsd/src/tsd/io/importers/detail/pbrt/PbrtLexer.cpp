// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "PbrtLexer.hpp"
#include <cctype>
#include <stdexcept>

namespace pbrt {

Lexer::Lexer(std::string_view source, const std::string &filename)
    : m_source(source), m_filename(filename)
{}

Token Lexer::next()
{
  if (m_hasPeeked) {
    m_hasPeeked = false;
    return std::move(m_peeked);
  }

  skipWhitespaceAndComments();

  if (m_pos >= m_source.size())
    return {TokenType::Eof, "", m_line};

  char c = m_source[m_pos];

  if (c == '[') {
    ++m_pos;
    return {TokenType::LBracket, "[", m_line};
  }

  if (c == ']') {
    ++m_pos;
    return {TokenType::RBracket, "]", m_line};
  }

  if (c == '"')
    return readString();

  // Sign followed by digit or dot → number; bare digit or dot → number
  if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
    return readNumber();
  }

  if (c == '-' || c == '+') {
    if (m_pos + 1 < m_source.size()) {
      char next = m_source[m_pos + 1];
      if (std::isdigit(static_cast<unsigned char>(next)) || next == '.')
        return readNumber();
    }
  }

  if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
    return readIdentifier();

  throw std::runtime_error(m_filename + ":" + std::to_string(m_line)
      + ": unexpected character '" + c + "'");
}

Token Lexer::peek()
{
  if (!m_hasPeeked) {
    m_peeked = next();
    m_hasPeeked = true;
  }
  return m_peeked;
}

const std::string &Lexer::currentFile() const
{
  return m_filename;
}

size_t Lexer::currentLine() const
{
  return m_line;
}

void Lexer::skipWhitespaceAndComments()
{
  while (m_pos < m_source.size()) {
    char c = m_source[m_pos];

    if (c == '\n') {
      ++m_line;
      ++m_pos;
      continue;
    }

    if (std::isspace(static_cast<unsigned char>(c))) {
      ++m_pos;
      continue;
    }

    if (c == '#') {
      while (m_pos < m_source.size() && m_source[m_pos] != '\n')
        ++m_pos;
      continue;
    }

    break;
  }
}

Token Lexer::readString()
{
  size_t line = m_line;
  ++m_pos; // skip opening quote

  std::string result;
  while (m_pos < m_source.size()) {
    char c = m_source[m_pos];

    if (c == '"') {
      ++m_pos;
      return {TokenType::String, result, line};
    }

    if (c == '\\' && m_pos + 1 < m_source.size()) {
      ++m_pos;
      char escaped = m_source[m_pos];
      switch (escaped) {
      case '\\':
        result += '\\';
        break;
      case '"':
        result += '"';
        break;
      case 'n':
        result += '\n';
        break;
      case 't':
        result += '\t';
        break;
      default:
        result += escaped;
        break;
      }
      ++m_pos;
      continue;
    }

    if (c == '\n')
      ++m_line;

    result += c;
    ++m_pos;
  }

  throw std::runtime_error(
      m_filename + ":" + std::to_string(line) + ": unterminated string");
}

Token Lexer::readNumber()
{
  size_t line = m_line;
  size_t start = m_pos;

  // Optional sign
  if (m_source[m_pos] == '-' || m_source[m_pos] == '+')
    ++m_pos;

  // Integer part
  while (m_pos < m_source.size()
      && std::isdigit(static_cast<unsigned char>(m_source[m_pos])))
    ++m_pos;

  // Fractional part
  if (m_pos < m_source.size() && m_source[m_pos] == '.') {
    ++m_pos;
    while (m_pos < m_source.size()
        && std::isdigit(static_cast<unsigned char>(m_source[m_pos])))
      ++m_pos;
  }

  // Exponent
  if (m_pos < m_source.size()
      && (m_source[m_pos] == 'e' || m_source[m_pos] == 'E')) {
    ++m_pos;
    if (m_pos < m_source.size()
        && (m_source[m_pos] == '-' || m_source[m_pos] == '+'))
      ++m_pos;
    while (m_pos < m_source.size()
        && std::isdigit(static_cast<unsigned char>(m_source[m_pos])))
      ++m_pos;
  }

  std::string text(m_source.substr(start, m_pos - start));
  return {TokenType::Number, text, line};
}

Token Lexer::readIdentifier()
{
  size_t line = m_line;
  size_t start = m_pos;

  while (m_pos < m_source.size()) {
    char c = m_source[m_pos];
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      ++m_pos;
    else
      break;
  }

  std::string text(m_source.substr(start, m_pos - start));
  return {TokenType::Identifier, text, line};
}

} // namespace pbrt
