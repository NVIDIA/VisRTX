// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// 3x3 symmetric eigenvalue solver — analytical (Cardano/trigonometric method)
// Returns eigenvalues sorted largest→smallest in eigs[0..2]
static void eigen3sym(double A[3][3], double eigs[3])
{
  double p1 = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
  if (p1 == 0.0) {
    // Diagonal matrix — eigenvalues are the diagonal entries
    eigs[0] = A[0][0];
    eigs[1] = A[1][1];
    eigs[2] = A[2][2];
    if (eigs[0] < eigs[1])
      std::swap(eigs[0], eigs[1]);
    if (eigs[1] < eigs[2])
      std::swap(eigs[1], eigs[2]);
    if (eigs[0] < eigs[1])
      std::swap(eigs[0], eigs[1]);
    return;
  }
  double q = (A[0][0] + A[1][1] + A[2][2]) / 3.0;
  double p2 = (A[0][0] - q) * (A[0][0] - q) + (A[1][1] - q) * (A[1][1] - q)
      + (A[2][2] - q) * (A[2][2] - q) + 2 * p1;
  double p = std::sqrt(p2 / 6.0);
  double B[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      B[i][j] = (A[i][j] - (i == j ? q : 0.0)) / p;
  double r = (B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1])
                 - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
                 + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]))
      / 2.0;
  double phi;
  if (r <= -1.0)
    phi = M_PI / 3.0;
  else if (r >= 1.0)
    phi = 0.0;
  else
    phi = std::acos(r) / 3.0;
  eigs[0] = q + 2 * p * std::cos(phi);
  eigs[2] = q + 2 * p * std::cos(phi + 2.0943951023931953); // phi + 2*pi/3
  eigs[1] = 3 * q - eigs[0] - eigs[2];
}

// lambda2: second (middle) eigenvalue of S^2 + O^2
// J is the velocity gradient Jacobian; uses no external dependencies
static double l2(double J[3][3])
{
  double S[3][3], O[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      S[i][j] = 0.5 * (J[i][j] + J[j][i]);
      O[i][j] = S[i][j] - J[j][i]; // O = S - J^T = 0.5*(J - J^T)
    }
  double M[3][3] = {};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        M[i][j] += S[i][k] * S[k][j] + O[i][k] * O[k][j];
  double eigs[3];
  eigen3sym(M, eigs); // sorted largest→smallest; eigs[1] is the middle
  return eigs[1];
}

// Q-criterion: 0.5 * (||O||_F^2 - ||S||_F^2)
static double q_crit(double J[3][3])
{
  double S[3][3], O[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      S[i][j] = 0.5 * (J[i][j] + J[j][i]);
      O[i][j] = S[i][j] - J[j][i];
    }
  double trO2 = 0, trS2 = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      trO2 += O[i][j] * O[i][j];
      trS2 += S[i][j] * S[i][j];
    }
  return 0.5 * (trO2 - trS2);
}

// ---------------------------------------------------------------------------
// Gradient helpers — float* input, double* output (scratch).
// Each gradient function reads float velocity values and writes double-precision
// gradient values so that finite differences retain full precision.
//
// Boundary scheme: 2nd-order one-sided stencil.
// Interior scheme: 2nd-order central difference.
// ---------------------------------------------------------------------------

// Forward declarations
inline void gradX(const float *u,
    double *uGrad,
    double dx,
    size_t nx,
    size_t ny,
    size_t nz);
inline void gradY(const float *v,
    double *vGrad,
    double dy,
    size_t nx,
    size_t ny,
    size_t nz);
inline void gradZ(const float *w,
    double *wGrad,
    const double *z_,
    size_t nx,
    size_t ny,
    size_t nz);
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
    size_t nz);

// ---------------------------------------------------------------------------

// x is the fastest index, so every (z,y) pair is a contiguous row of nx
// floats — collapse both outer loops into a single row index.
inline void gradX(const float *u,
    double *uGrad,
    double dx,
    size_t nx,
    size_t ny,
    size_t nz)
{
  const double inv2dx = 1.0 / (2.0 * dx);
  const int nrows = (int)(ny * nz);
  for (int r = 0; r < nrows; ++r) {
    const float *ur = u     + (size_t)r * nx;
    double      *gr = uGrad + (size_t)r * nx;
    gr[0]      = (-3.0 * ur[0] + 4.0 * ur[1]      - ur[2])        * inv2dx;
    for (size_t c = 1; c < nx - 1; ++c)
      gr[c]    = ((double)ur[c + 1] - (double)ur[c - 1])           * inv2dx;
    gr[nx - 1] = ( 3.0 * ur[nx-1] - 4.0 * ur[nx-2] + ur[nx-3])   * inv2dx;
  }
}

// y has stride nx.  Iterate over every (z,x) pair as the column base;
// the 1D y-pass then uses vc[y*nx] with no extra offset arithmetic.
inline void gradY(const float *v,
    double *vGrad,
    double dy,
    size_t nx,
    size_t ny,
    size_t nz)
{
  const double inv2dy = 1.0 / (2.0 * dy);
  const size_t slab = nx * ny;
  for (int z = 0; z < (int)nz; ++z) {
    for (int x = 0; x < (int)nx; ++x) {
      const float *vc = v     + (size_t)z * slab + x;
      double      *gc = vGrad + (size_t)z * slab + x;
      gc[0]            = (-3.0 * vc[0] + 4.0 * vc[nx]        - vc[2*nx])            * inv2dy;
      for (size_t y = 1; y < ny - 1; ++y)
        gc[y * nx]     = ((double)vc[(y+1)*nx] - (double)vc[(y-1)*nx])               * inv2dy;
      gc[(ny-1) * nx]  = ( 3.0 * vc[(ny-1)*nx] - 4.0 * vc[(ny-2)*nx] + vc[(ny-3)*nx]) * inv2dy;
    }
  }
}

// Non-uniform z spacing: 1st-order one-sided at boundaries, 2nd-order central
// in the interior.
// z has stride slab=nx*ny.  Every (y,x) pair gives a z-column addressable as
// wc[k*slab] — flatten both outer loops into a single index over ny*nx.
inline void gradZ(const float *w,
    double *wGrad,
    const double *z_,
    size_t nx,
    size_t ny,
    size_t nz)
{
  const size_t slab = nx * ny;
  const int ncols = (int)slab;
  for (int yx = 0; yx < ncols; ++yx) {
    const float *wc = w     + yx;
    double      *gc = wGrad + yx;
    gc[0]              = ((double)wc[slab]        - (double)wc[0])           / (z_[1]      - z_[0]);
    for (size_t k = 1; k < nz - 1; ++k)
      gc[k * slab]     = ((double)wc[(k+1)*slab]  - (double)wc[(k-1)*slab]) / (z_[k+1]    - z_[k-1]);
    gc[(nz-1) * slab]  = ((double)wc[(nz-1)*slab] - (double)wc[(nz-2)*slab]) / (z_[nz-1] - z_[nz-2]);
  }
}

// ---------------------------------------------------------------------------
// vort — compute vortical quantities from float velocity fields.
//
// Inputs:  u, v, w    — float velocity components (x,y,z), row-major [z][y][x]
//          x_, y_, z_ — double coordinate arrays (length nx, ny, nz)
// Outputs: vorticity, helicity, lambda2, qCriterion — float*, written in-place.
//          Any output pointer may be null; null outputs are simply skipped.
//          If both lambda2 and qCriterion are null the eigenvalue loop is skipped.
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
  const size_t len = nx * ny * nz;

  // 9 double-precision gradient scratch arrays (shared across u/v/w)
  std::vector<double> dux(len), dvx(len), dwx(len);
  std::vector<double> duy(len), dvy(len), dwy(len);
  std::vector<double> duz(len), dvz(len), dwz(len);

  const double dx = x_[1] - x_[0];
  const double dy = y_[1] - y_[0];

  gradX(u, dux.data(), dx, nx, ny, nz);
  gradX(v, dvx.data(), dx, nx, ny, nz);
  gradX(w, dwx.data(), dx, nx, ny, nz);

  gradY(u, duy.data(), dy, nx, ny, nz);
  gradY(v, dvy.data(), dy, nx, ny, nz);
  gradY(w, dwy.data(), dy, nx, ny, nz);

  gradZ(u, duz.data(), z_, nx, ny, nz);
  gradZ(v, dvz.data(), z_, nx, ny, nz);
  gradZ(w, dwz.data(), z_, nx, ny, nz);

  if (vorticity || helicity) {
    for (size_t i = 0; i < len; ++i) {
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
  }

  // Lambda2 and Q-criterion (expensive: one eigensolve per voxel)
  if (lambda2 || qCriterion) {
    for (size_t i = 0; i < len; ++i) {
      double J[3][3] = {{dux[i], duy[i], duz[i]},
          {dvx[i], dvy[i], dvz[i]},
          {dwx[i], dwy[i], dwz[i]}};
      if (lambda2)
        lambda2[i] = (float)(-std::min(l2(J), 0.0));
      if (qCriterion)
        qCriterion[i] = (float)std::max(q_crit(J), 0.0);
    }
  }

  std::cout << "[vort] Vortical variables computed" << std::endl;
}
