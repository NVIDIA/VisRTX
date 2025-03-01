R"(

mdl 1.8;

import ::df::*;
import ::math::*;
import ::state::*;

export material diffuseWhite() = let
{
  material mat = material(surface : material_surface(
      scattering : df::diffuse_reflection_bsdf(tint : color(0.8f))));
}
in mat;

export material debugCheckerBoard(float3 scale = float3(3.0f),
    color color1 = color(1.0f, 0.0f, 0.0f),
    color color2 = color(0.0f, 1.0f, 0.0f)) = let
{
  float3 objectPosition =
      math::abs(state::transform_point(state::coordinate_internal,
          state::coordinate_world,
          state::position()))
      * scale;
  int3 checkerPos = int3(math::fmod(math::round(objectPosition), 2.0));

  color checkerColor =
      (checkerPos.x != checkerPos.y) != (checkerPos.z == 0) ? color1 : color2;

  material mat = material(surface : material_surface(
      scattering : df::diffuse_reflection_bsdf(tint : checkerColor)));
}
in mat;

)"