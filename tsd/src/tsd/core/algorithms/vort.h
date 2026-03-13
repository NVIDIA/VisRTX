// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace tsd::core {

// lambda2: middle eigenvalue of M = S^2 + O^2, where S=(J+J^T)/2, O=(J-J^T)/2.
// M is symmetric with 6 unique entries computed as M = (J^2 + (J^T)^2) / 2.
// Eigenvalues via the trigonometric (Cardano) formula: eig_k = q + 2p*cos(phi +
// 2*pi*k/3). Middle eigenvalue recovered from the trace identity: eig1 = 3q -
// eig0 - eig2.
static double l2(double j00,
    double j01,
    double j02,
    double j10,
    double j11,
    double j12,
    double j20,
    double j21,
    double j22)
{
  // 6 unique entries of symmetric M = (J^2 + (J^T)^2) / 2
  const double m00 = j00 * j00 + j01 * j10 + j02 * j20;
  const double m11 = j10 * j01 + j11 * j11 + j12 * j21;
  const double m22 = j20 * j02 + j21 * j12 + j22 * j22;
  const double m01 = 0.5
      * (j00 * j01 + j01 * j11 + j02 * j21 + j00 * j10 + j10 * j11 + j20 * j12);
  const double m02 = 0.5
      * (j00 * j02 + j01 * j12 + j02 * j22 + j00 * j20 + j10 * j21 + j20 * j22);
  const double m12 = 0.5
      * (j10 * j02 + j11 * j12 + j12 * j22 + j01 * j20 + j11 * j21 + j21 * j22);

  const double q = (m00 + m11 + m22) / 3.0;
  const double a = m00 - q, d = m11 - q, f = m22 - q;
  const double p1 = m01 * m01 + m02 * m02 + m12 * m12;
  if (p1 == 0.0) {
    // Diagonal — middle eigenvalue by inspection
    double e[3] = {a, d, f};
    if (e[0] < e[1])
      std::swap(e[0], e[1]);
    if (e[1] < e[2])
      std::swap(e[1], e[2]);
    if (e[0] < e[1])
      std::swap(e[0], e[1]);
    return q + e[1];
  }
  const double p = std::sqrt((a * a + d * d + f * f + 2.0 * p1) / 6.0);
  const double r = (a * (d * f - m12 * m12) - m01 * (m01 * f - m12 * m02)
                       + m02 * (m01 * m12 - d * m02))
      / (2.0 * p * p * p);
  const double phi = std::acos(std::max(-1.0, std::min(1.0, r))) / 3.0;
  const double e0 = q + 2.0 * p * std::cos(phi);
  const double e2 =
      q + 2.0 * p * std::cos(phi + 2.0943951023931953); // phi + 2*pi/3
  return 3.0 * q - e0 - e2; // middle eigenvalue via trace identity
}

// Q-criterion: 0.5*(||O||^2 - ||S||^2) = -0.5*tr(J^2) = -0.5*(j00^2+j11^2+j22^2
// + 2*(j01*j10+j02*j20+j12*j21))
static double q_crit(double j00,
    double j01,
    double j02,
    double j10,
    double j11,
    double j12,
    double j20,
    double j21,
    double j22)
{
  return -0.5
      * (j00 * j00 + j11 * j11 + j22 * j22
          + 2.0 * (j01 * j10 + j02 * j20 + j12 * j21));
}

// ---------------------------------------------------------------------------
// grad3D — finite differences along one axis of a 3D float field.
//
// axis 0 (x): stride 1,       n=nx, outer=ny*nz passes
// axis 1 (y): stride nx,      n=ny, outer=nx*nz passes
// axis 2 (z): stride nx*ny,   n=nz, outer=nx*ny passes
//
// c[]: coordinate array along the axis (length n).
// Boundary: 1st-order one-sided.  Interior: 2nd-order central.
// ---------------------------------------------------------------------------
static void grad3D(const float *fc,
    double *gc,
    size_t nx,
    size_t ny,
    size_t nz,
    int axis,
    const double *c)
{
  const size_t slab = nx * ny;
  size_t n, s, outer;
  if (axis == 0) {
    n = nx;
    s = 1;
    outer = ny * nz;
  } else if (axis == 1) {
    n = ny;
    s = nx;
    outer = nx * nz;
  } else {
    n = nz;
    s = slab;
    outer = slab;
  }

  auto pass = [n, s, c](const float *f, double *g) {
    g[0] = ((double)f[s] - (double)f[0]) / (c[1] - c[0]);
    for (size_t i = 1; i < n - 1; ++i)
      g[i * s] = ((double)f[(i + 1) * s] - (double)f[(i - 1) * s])
          / (c[i + 1] - c[i - 1]);
    g[(n - 1) * s] = ((double)f[(n - 1) * s] - (double)f[(n - 2) * s])
        / (c[n - 1] - c[n - 2]);
  };

  if (axis == 1) {
    for (size_t z = 0; z < nz; ++z)
      for (size_t x = 0; x < nx; ++x)
        pass(fc + z * slab + x, gc + z * slab + x);
  } else if (axis == 2) {
    for (size_t y = 0; y < ny; ++y)
      for (size_t x = 0; x < nx; ++x)
        pass(fc + y * nx + x, gc + y * nx + x);
  } else { // axis == 0
    for (size_t r = 0; r < outer; ++r)
      pass(fc + r * n * s, gc + r * n * s);
  }
}

// ---------------------------------------------------------------------------
// vort_from_jacobians — compute vortical quantities from pre-computed Jacobian
// components.
//
// Inputs:  u, v, w     — float velocity components (any layout, length len)
//          dux..dwz    — Jacobian entries: d(u,v,w)/d(x,y,z), double, length
//          len
//                        naming: d<vel><dir>, e.g. duy = ∂u/∂y
// Outputs: vorticity, helicity, lambda2, qCriterion — float*, written in-place.
//          Any output pointer may be null; null outputs are simply skipped.
// ---------------------------------------------------------------------------
inline void vort_from_jacobians(const float *u,
    const float *v,
    const float *w,
    const double *dux,
    const double *dvx,
    const double *dwx,
    const double *duy,
    const double *dvy,
    const double *dwy,
    const double *duz,
    const double *dvz,
    const double *dwz,
    float *vorticity,
    float *helicity,
    float *lambda2,
    float *qCriterion,
    size_t len)
{
  for (size_t i = 0; i < len; ++i) {
    if (vorticity || helicity) {
      const double omx = dwy[i] - dvz[i];
      const double omy = duz[i] - dwx[i];
      const double omz = dvx[i] - duy[i];
      const double omag = std::sqrt(omx * omx + omy * omy + omz * omz);
      if (vorticity)
        vorticity[i] = (float)omag;
      if (helicity) {
        const double ui = u[i], vi = v[i], wi = w[i];
        const double h = std::abs(omx * ui + omy * vi + omz * wi);
        const double vmag = std::sqrt(ui * ui + vi * vi + wi * wi);
        helicity[i] = (vmag > 0.0 && omag > 0.0)
            ? (float)(h / (2.0 * vmag * omag))
            : 0.0f;
      }
    }
    if (lambda2 || qCriterion) {
      if (lambda2)
        lambda2[i] = (float)(-std::min(l2(dux[i],
                                           duy[i],
                                           duz[i],
                                           dvx[i],
                                           dvy[i],
                                           dvz[i],
                                           dwx[i],
                                           dwy[i],
                                           dwz[i]),
            0.0));
      if (qCriterion)
        qCriterion[i] = (float)std::max(q_crit(dux[i],
                                            duy[i],
                                            duz[i],
                                            dvx[i],
                                            dvy[i],
                                            dvz[i],
                                            dwx[i],
                                            dwy[i],
                                            dwz[i]),
            0.0);
    }
  }
}

// ---------------------------------------------------------------------------
// vort — compute vortical quantities from float velocity fields.
//
// Inputs:  u, v, w    — float velocity components (x,y,z), row-major [z][y][x]
//          x_, y_, z_ — double coordinate arrays (length nx, ny, nz)
// Outputs: vorticity, helicity, lambda2, qCriterion — float*, written in-place.
//          Any output pointer may be null; null outputs are simply skipped.
//
// Internal gradient arrays are double-precision scratch space (9 × N doubles).
// ---------------------------------------------------------------------------
inline void vort(const float *u,
    const float *v,
    const float *w,
    const double *x_,
    const double *y_,
    const double *z_,
    float *vorticity,
    float *helicity,
    float *lambda2,
    float *qCriterion,
    size_t nx,
    size_t ny,
    size_t nz)
{
  if (!vorticity && !helicity && !lambda2 && !qCriterion)
    return;

  const size_t len = nx * ny * nz;

  // 9 double-precision gradient scratch arrays (shared across u/v/w)
  std::vector<double> dux(len), dvx(len), dwx(len);
  std::vector<double> duy(len), dvy(len), dwy(len);
  std::vector<double> duz(len), dvz(len), dwz(len);

  grad3D(u, dux.data(), nx, ny, nz, 0, x_);
  grad3D(v, dvx.data(), nx, ny, nz, 0, x_);
  grad3D(w, dwx.data(), nx, ny, nz, 0, x_);

  grad3D(u, duy.data(), nx, ny, nz, 1, y_);
  grad3D(v, dvy.data(), nx, ny, nz, 1, y_);
  grad3D(w, dwy.data(), nx, ny, nz, 1, y_);

  grad3D(u, duz.data(), nx, ny, nz, 2, z_);
  grad3D(v, dvz.data(), nx, ny, nz, 2, z_);
  grad3D(w, dwz.data(), nx, ny, nz, 2, z_);

  vort_from_jacobians(u,
      v,
      w,
      dux.data(),
      dvx.data(),
      dwx.data(),
      duy.data(),
      dvy.data(),
      dwy.data(),
      duz.data(),
      dvz.data(),
      dwz.data(),
      vorticity,
      helicity,
      lambda2,
      qCriterion,
      len);
}

} // namespace tsd::core
