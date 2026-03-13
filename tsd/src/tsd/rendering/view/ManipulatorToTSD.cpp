// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/view/ManipulatorToTSD.hpp"

namespace tsd::rendering {

void updateCameraObject(
    tsd::scene::Camera &c, const Manipulator &m, bool includeManipulatorMetadata)
{
  c.beginParameterBatch();

  c.setParameter("direction", m.dir());
  c.setParameter("up", m.up());

  if (c.subtype() == scene::tokens::camera::orthographic) {
    c.setParameter("position", m.eye_FixedDistance());
    c.setParameter("height", m.distance() * 0.75f);
  } else {
    c.setParameter("position", m.eye());
  }

  if (includeManipulatorMetadata) {
    c.setMetadataValue("manipulator.at", m.at());
    c.setMetadataValue("manipulator.distance", m.distance());
    c.setMetadataValue("manipulator.fixedDistance", m.fixedDistance());
    c.setMetadataValue("manipulator.azel", m.azel());
    c.setMetadataValue("manipulator.up", int(m.axis()));
  }

  c.endParameterBatch();
}

void updateManipulatorFromCamera(Manipulator &m, const tsd::scene::Camera &c)
{
  auto at = c.getMetadataValue("manipulator.at").getValueOr(m.at());
  auto d = c.getMetadataValue("manipulator.distance").getValueOr(m.distance());
  auto azel = c.getMetadataValue("manipulator.azel").getValueOr(m.azel());
  auto up = c.getMetadataValue("manipulator.up").getValueOr(int(m.axis()));
  auto fd = c.getMetadataValue("manipulator.fixedDistance")
                .getValueOr(m.fixedDistance());

  m.setConfig(at, d, azel);
  m.setAxis(static_cast<UpAxis>(up));
  m.setFixedDistance(fd);
}

} // namespace tsd::rendering
