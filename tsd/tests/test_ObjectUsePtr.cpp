// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"

using tsd::core::Material;
using tsd::core::MaterialRef;
using tsd::core::Object;
using tsd::core::ObjectUsePtr;
using tsd::core::Scene;
namespace material = tsd::core::tokens::material;

SCENARIO("tsd::core::ObjectUsePtr interface", "[ObjectUsePtr]")
{
  Scene scene;

  GIVEN("A default constructed ObjectUsePtr")
  {
    ObjectUsePtr<Material> ptr;

    THEN("The pointer is null")
    {
      REQUIRE(ptr.get() == nullptr);
      REQUIRE_FALSE(ptr);
    }

    THEN("The ref is invalid")
    {
      REQUIRE_FALSE(ptr.ref());
    }
  }

  GIVEN("An object created through a Scene")
  {
    auto matRef = scene.createObject<Material>(material::matte);
    auto *mat = matRef.data();

    THEN("The object starts with zero use counts")
    {
      REQUIRE(mat->useCount(Object::UseKind::APP) == 0);
      REQUIRE(mat->useCount(Object::UseKind::INTERNAL) == 0);
      REQUIRE(mat->totalUseCount() == 0);
    }

    WHEN("An ObjectUsePtr is constructed from an ObjectPoolRef")
    {
      ObjectUsePtr<Material> ptr(matRef);

      THEN("The pointer is valid")
      {
        REQUIRE(ptr.get() == mat);
        REQUIRE(ptr);
      }

      THEN("The APP use count is incremented to 1")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
        REQUIRE(mat->totalUseCount() == 1);
      }

      THEN("The ref matches the original")
      {
        REQUIRE(ptr.ref() == matRef);
      }
    }

    WHEN("An ObjectUsePtr is constructed from a raw pointer")
    {
      ObjectUsePtr<Material> ptr(mat);

      THEN("The APP use count is incremented to 1")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
        REQUIRE(mat->totalUseCount() == 1);
      }

      THEN("The pointer dereferences to the correct object")
      {
        REQUIRE(ptr.get() == mat);
        REQUIRE(&(*ptr) == mat);
        REQUIRE(ptr->type() == ANARI_MATERIAL);
      }
    }

    WHEN("An INTERNAL ObjectUsePtr is constructed from an ObjectPoolRef")
    {
      ObjectUsePtr<Material, Object::UseKind::INTERNAL> ptr(matRef);

      THEN("The INTERNAL use count is incremented to 1")
      {
        REQUIRE(mat->useCount(Object::UseKind::INTERNAL) == 1);
        REQUIRE(mat->useCount(Object::UseKind::APP) == 0);
        REQUIRE(mat->totalUseCount() == 1);
      }
    }

    WHEN("Multiple ObjectUsePtrs reference the same object")
    {
      ObjectUsePtr<Material> ptr1(matRef);
      ObjectUsePtr<Material> ptr2(matRef);
      ObjectUsePtr<Material> ptr3(matRef);

      THEN("The APP use count reflects all references")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 3);
        REQUIRE(mat->totalUseCount() == 3);
      }
    }
  }

  GIVEN("An ObjectUsePtr managing an object")
  {
    auto mat = scene.createObject<Material>(material::matte);
    ObjectUsePtr<Material> ptr(mat);

    REQUIRE(mat->useCount(Object::UseKind::APP) == 1);

    WHEN("The ObjectUsePtr is copy constructed")
    {
      ObjectUsePtr<Material> copy(ptr);

      THEN("Both pointers reference the same object")
      {
        REQUIRE(copy.get() == ptr.get());
        REQUIRE(copy.get() == mat.data());
      }

      THEN("The use count is incremented to 2")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 2);
        REQUIRE(mat->totalUseCount() == 2);
      }
    }

    WHEN("The ObjectUsePtr is copy assigned")
    {
      ObjectUsePtr<Material> copy;
      copy = ptr;

      THEN("Both pointers reference the same object")
      {
        REQUIRE(copy.get() == mat.data());
      }

      THEN("The use count is incremented to 2")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 2);
      }
    }

    WHEN("The ObjectUsePtr is copy assigned over an existing reference")
    {
      auto mat2 = scene.createObject<Material>(material::matte);
      ObjectUsePtr<Material> other(mat2);

      REQUIRE(mat2->useCount(Object::UseKind::APP) == 1);

      other = ptr;

      THEN("The old object's use count is decremented")
      {
        REQUIRE(mat2->useCount(Object::UseKind::APP) == 0);
      }

      THEN("The new object's use count is incremented")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 2);
      }

      THEN("The pointer now references the new object")
      {
        REQUIRE(other.get() == mat.data());
      }
    }

    WHEN("The ObjectUsePtr is self-assigned")
    {
      ptr = ptr;

      THEN("The use count remains 1")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
      }

      THEN("The pointer is still valid")
      {
        REQUIRE(ptr.get() == mat.data());
      }
    }
  }

  GIVEN("An ObjectUsePtr managing an object for move operations")
  {
    auto mat = scene.createObject<Material>(material::matte);
    ObjectUsePtr<Material> ptr(mat);

    REQUIRE(mat->useCount(Object::UseKind::APP) == 1);

    WHEN("The ObjectUsePtr is move constructed")
    {
      ObjectUsePtr<Material> moved(std::move(ptr));

      THEN("The moved-to pointer references the object")
      {
        REQUIRE(moved.get() == mat.data());
        REQUIRE(moved);
      }

      THEN("The moved-from pointer is null")
      {
        REQUIRE(ptr.get() == nullptr);
        REQUIRE_FALSE(ptr);
      }

      THEN("The use count remains 1 (no increment or decrement)")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
        REQUIRE(mat->totalUseCount() == 1);
      }
    }

    WHEN("The ObjectUsePtr is move assigned")
    {
      ObjectUsePtr<Material> moved;
      moved = std::move(ptr);

      THEN("The moved-to pointer references the object")
      {
        REQUIRE(moved.get() == mat.data());
      }

      THEN("The moved-from pointer is null")
      {
        REQUIRE(ptr.get() == nullptr);
      }

      THEN("The use count remains 1")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
      }
    }

    WHEN("The ObjectUsePtr is move assigned over an existing reference")
    {
      auto mat2 = scene.createObject<Material>(material::matte);
      ObjectUsePtr<Material> other(mat2);

      REQUIRE(mat2->useCount(Object::UseKind::APP) == 1);

      other = std::move(ptr);

      THEN("The old object's use count is decremented")
      {
        REQUIRE(mat2->useCount(Object::UseKind::APP) == 0);
      }

      THEN("The new object's use count remains 1")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
      }

      THEN("The moved-from pointer is null")
      {
        REQUIRE(ptr.get() == nullptr);
      }

      THEN("The moved-to pointer references the correct object")
      {
        REQUIRE(other.get() == mat.data());
      }
    }
  }

  GIVEN("An ObjectUsePtr for testing reset and destruction")
  {
    auto mat = scene.createObject<Material>(material::matte);

    WHEN("An ObjectUsePtr is reset")
    {
      ObjectUsePtr<Material> ptr(mat);
      REQUIRE(mat->useCount(Object::UseKind::APP) == 1);

      ptr.reset();

      THEN("The pointer becomes null")
      {
        REQUIRE(ptr.get() == nullptr);
        REQUIRE_FALSE(ptr);
      }

      THEN("The use count is decremented to 0")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 0);
        REQUIRE(mat->totalUseCount() == 0);
      }
    }

    WHEN("An ObjectUsePtr goes out of scope (destructor)")
    {
      {
        ObjectUsePtr<Material> ptr(mat);
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
      }

      THEN("The use count is decremented back to 0")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 0);
      }
    }

    WHEN("Multiple ObjectUsePtrs go out of scope sequentially")
    {
      {
        ObjectUsePtr<Material> ptr1(mat);
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
        {
          ObjectUsePtr<Material> ptr2(mat);
          REQUIRE(mat->useCount(Object::UseKind::APP) == 2);
        }
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
      }

      THEN("All use counts return to 0")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 0);
      }
    }

    WHEN("A default ObjectUsePtr is reset (no-op)")
    {
      ObjectUsePtr<Material> ptr;
      ptr.reset();

      THEN("The pointer is still null")
      {
        REQUIRE(ptr.get() == nullptr);
      }
    }
  }

  GIVEN("ObjectUsePtrs for testing assignment from raw pointer and ref")
  {
    auto mat = scene.createObject<Material>(material::matte);

    WHEN("An ObjectUsePtr is assigned from a raw pointer")
    {
      ObjectUsePtr<Material> ptr;
      ptr = mat.data();

      THEN("The pointer is valid and use count is 1")
      {
        REQUIRE(ptr.get() == mat.data());
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
      }
    }

    WHEN("An ObjectUsePtr is assigned from an ObjectPoolRef")
    {
      ObjectUsePtr<Material> ptr;
      ptr = mat;

      THEN("The pointer is valid and use count is 1")
      {
        REQUIRE(ptr.get() == mat.data());
        REQUIRE(mat->useCount(Object::UseKind::APP) == 1);
      }
    }

    WHEN("A non-empty ObjectUsePtr is assigned a new raw pointer")
    {
      auto mat2 = scene.createObject<Material>(material::matte);

      ObjectUsePtr<Material> ptr(mat);
      REQUIRE(mat->useCount(Object::UseKind::APP) == 1);

      ptr = mat2.data();

      THEN("The old object's use count is decremented")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 0);
      }

      THEN("The new object's use count is incremented")
      {
        REQUIRE(mat2->useCount(Object::UseKind::APP) == 1);
      }
    }

    WHEN("An ObjectUsePtr is assigned nullptr via raw pointer")
    {
      ObjectUsePtr<Material> ptr(mat);
      REQUIRE(mat->useCount(Object::UseKind::APP) == 1);

      ptr = static_cast<Material *>(nullptr);

      THEN("The use count is decremented and pointer is null")
      {
        REQUIRE(mat->useCount(Object::UseKind::APP) == 0);
        REQUIRE(ptr.get() == nullptr);
      }
    }
  }

  GIVEN("Two ObjectUsePtrs for testing equality operators")
  {
    auto matRef = scene.createObject<Material>(material::matte);

    ObjectUsePtr<Material> ptr1(matRef);
    ObjectUsePtr<Material> ptr2(matRef);

    THEN("Pointers to the same object compare equal")
    {
      REQUIRE(ptr1 == ptr2);
      REQUIRE_FALSE(ptr1 != ptr2);
    }

    WHEN("A second object is created")
    {
      auto mat2Ref = scene.createObject<Material>(material::matte);
      ObjectUsePtr<Material> ptr3(mat2Ref);

      THEN("Pointers to different objects compare unequal")
      {
        REQUIRE(ptr1 != ptr3);
        REQUIRE_FALSE(ptr1 == ptr3);
      }
    }
  }
}
