/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "catch.hpp"
// visrtx
#include "utility/AnariAny.h"

using namespace visrtx;

template <typename T>
inline void verify_value(const AnariAny &v, const T &correctValue)
{
  REQUIRE(v.valid());
  REQUIRE(v.is<T>());
  REQUIRE(v.get<T>() == correctValue);
}

template <typename T>
inline void test_interface(T testValue, T testValue2)
{
  AnariAny v;
  REQUIRE(!v.valid());

  SECTION("Can make valid by C++ construction")
  {
    AnariAny v2(testValue);
    verify_value<T>(v2, testValue);
  }

  SECTION("Can make valid by C construction")
  {
    AnariAny v2(anari::ANARITypeFor<T>::value, &testValue);
    verify_value<T>(v2, testValue);
  }

  SECTION("Can make valid by calling operator=()")
  {
    v = testValue;
    verify_value<T>(v, testValue);
  }

  SECTION("Can make valid by copy construction")
  {
    v = testValue;
    AnariAny v2(v);
    verify_value<T>(v2, testValue);
  }

  SECTION("Two objects with same value are equal if constructed the same")
  {
    v = testValue;
    AnariAny v2 = testValue;
    REQUIRE(v.type() == v2.type());
    REQUIRE(v == v2);
  }

  SECTION("Two objects with same value are equal if assigned from another")
  {
    v = testValue;
    AnariAny v2 = testValue2;
    v = v2;
    REQUIRE(v == v2);
  }

  SECTION("Two objects with different values are not equal")
  {
    v = testValue;
    AnariAny v2 = testValue2;
    REQUIRE(v != v2);
  }
}

// Value Tests ////////////////////////////////////////////////////////////////

TEST_CASE("AnariAny 'int' type behavior", "[AnariAny]")
{
  test_interface<int>(5, 7);
}

TEST_CASE("AnariAny 'float' type behavior", "[AnariAny]")
{
  test_interface<float>(1.f, 2.f);
}

TEST_CASE("AnariAny 'bool' type behavior", "[AnariAny]")
{
  test_interface<bool>(true, false);
}

// Object Tests ///////////////////////////////////////////////////////////////

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(anari::RefCounted *, ANARI_OBJECT);
ANARI_TYPEFOR_DEFINITION(anari::RefCounted *);
} // namespace anari

TEST_CASE("AnariAny object behavior", "[AnariAny]")
{
  auto *obj = new anari::RefCounted();

  SECTION("Object use count starts at 1")
  {
    REQUIRE(obj->useCount() == 1);
  }

  SECTION("Placing the object in AnariAny increments the ref count")
  {
    AnariAny v = obj;
    REQUIRE(obj->useCount() == 2);
  }

  REQUIRE(obj->useCount() == 1);

  obj->refDec();
}
