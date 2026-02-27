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

// Forward declarations

inline void gradX(std::vector<double> &u,
    std::vector<double> &uGrad,
    double dx,
    size_t nx,
    size_t ny,
    size_t nz);
inline void gradY(std::vector<double> &v,
    std::vector<double> &vGrad,
    double dy,
    size_t nx,
    size_t ny,
    size_t nz);
inline void gradZ(std::vector<double> &w,
    std::vector<double> &wGrad,
    std::vector<double> &z_,
    size_t nx,
    size_t ny,
    size_t nz);
inline void vort(std::vector<double> &u,
    std::vector<double> &v,
    std::vector<double> &w,
    std::vector<double> &x_,
    std::vector<double> &y_,
    std::vector<double> &z_,
    std::vector<double> &vorticity,
    std::vector<double> &helicity,
    std::vector<double> &lambda2,
    std::vector<double> &qCriterion,
    size_t nx,
    size_t ny,
    size_t nz);

// 2nd-order one-sided BCs, 2nd-order central FD in interior, in x
inline void gradX(std::vector<double> &u,
    std::vector<double> &uGrad,
    double dx,
    size_t nx,
    size_t ny,
    size_t nz)
{
#pragma omp parallel for
  for (auto z = 0; z < (int)nz; ++z) {
    auto off = z * nx * ny;
    for (auto row = 0; row < (int)ny; ++row) {
      // Left boundary: 2nd-order one-sided forward
      uGrad[off + row * nx + 0] =
          (-3.0 * u[off + row * nx + 0] + 4.0 * u[off + row * nx + 1]
              - u[off + row * nx + 2])
          / (2.0 * dx);
      // Right boundary: 2nd-order one-sided backward
      uGrad[off + row * nx + nx - 1] =
          (3.0 * u[off + row * nx + nx - 1] - 4.0 * u[off + row * nx + nx - 2]
              + u[off + row * nx + nx - 3])
          / (2.0 * dx);
      // Interior: 2nd-order central
      for (auto col = 1; col < (int)nx - 1; ++col) {
        uGrad[off + row * nx + col] =
            (u[off + row * nx + col + 1] - u[off + row * nx + col - 1])
            / (2.0 * dx);
      }
    }
  }
}

// 2nd-order one-sided BCs, 2nd-order central FD in interior, in y
inline void gradY(std::vector<double> &v,
    std::vector<double> &vGrad,
    double dy,
    size_t nx,
    size_t ny,
    size_t nz)
{
#pragma omp parallel for
  for (auto z = 0; z < (int)nz; ++z) {
    auto off = z * nx * ny;
    for (auto col = 0; col < (int)nx; ++col) {
      // Bottom boundary (row=0): 2nd-order one-sided forward
      vGrad[0 * nx + col + off] =
          (-3.0 * v[0 * nx + col + off] + 4.0 * v[1 * nx + col + off]
              - v[2 * nx + col + off])
          / (2.0 * dy);
      // Top boundary (row=ny-1): 2nd-order one-sided backward
      vGrad[(ny - 1) * nx + col + off] =
          (3.0 * v[(ny - 1) * nx + col + off]
              - 4.0 * v[(ny - 2) * nx + col + off]
              + v[(ny - 3) * nx + col + off])
          / (2.0 * dy);
      // Interior: 2nd-order central
      for (auto row = 1; row < (int)ny - 1; ++row) {
        vGrad[row * nx + col + off] =
            (v[(row + 1) * nx + col + off] - v[(row - 1) * nx + col + off])
            / (2.0 * dy);
      }
    }
  }
}

// Non-uniform spacing in z (one-sided FD at boundaries)
inline void gradZ(std::vector<double> &w,
    std::vector<double> &wGrad,
    std::vector<double> &z_,
    size_t nx,
    size_t ny,
    size_t nz)
{
  size_t off = nx * ny;
#pragma omp parallel for
  for (auto row = 0; row < (int)ny; ++row) {
    for (auto col = 0; col < (int)nx; ++col) {
      wGrad[(nz - 1) * off + row * nx + col] =
          (w[(nz - 1) * off + row * nx + col]
              - w[(nz - 2) * off + row * nx + col])
          / (z_[nz - 1] - z_[nz - 2]);
      wGrad[0 * off + row * nx + col] =
          (w[1 * off + row * nx + col] - w[0 * off + row * nx + col])
          / (z_[1] - z_[0]);
      // Interior: 2nd-order central
      for (auto z = 1; z < (int)nz - 1; ++z) {
        wGrad[z * off + row * nx + col] =
            (w[(z + 1) * off + row * nx + col]
                - w[(z - 1) * off + row * nx + col])
            / (z_[z + 1] - z_[z - 1]);
      }
    }
  }
}

inline void vort(std::vector<double> &u,
    std::vector<double> &v,
    std::vector<double> &w,
    std::vector<double> &x_,
    std::vector<double> &y_,
    std::vector<double> &z_,
    std::vector<double> &vorticity,
    std::vector<double> &helicity,
    std::vector<double> &lambda2,
    std::vector<double> &qCriterion,
    size_t nx,
    size_t ny,
    size_t nz)
{
  auto len = u.size();
  std::vector<double> dux(len);
  std::vector<double> dvx(len);
  std::vector<double> dwx(len);
  std::vector<double> duy(len);
  std::vector<double> dvy(len);
  std::vector<double> dwy(len);
  std::vector<double> duz(len);
  std::vector<double> dvz(len);
  std::vector<double> dwz(len);

  double dx = x_.at(1) - x_.at(0);
  double dy = y_.at(1) - y_.at(0);

  gradX(u, dux, dx, nx, ny, nz);
  gradX(v, dvx, dx, nx, ny, nz);
  gradX(w, dwx, dx, nx, ny, nz);

  gradY(u, duy, dy, nx, ny, nz);
  gradY(v, dvy, dy, nx, ny, nz);
  gradY(w, dwy, dy, nx, ny, nz);

  gradZ(u, duz, z_, nx, ny, nz);
  gradZ(v, dvz, z_, nx, ny, nz);
  gradZ(w, dwz, z_, nx, ny, nz);

  for (size_t i = 0; i < len; ++i) {
    vorticity[i] =
        std::sqrt((dwy[i] - dvz[i]) * (dwy[i] - dvz[i])
            + (duz[i] - dwx[i]) * (duz[i] - dwx[i])
            + (dvx[i] - duy[i]) * (dvx[i] - duy[i]));

    helicity[i] = std::abs(
        ((dwy[i] - dvz[i]) * u[i]) + ((duz[i] - dwx[i]) * v[i])
        + ((dvx[i] - duy[i]) * w[i]));

    double velMag = std::sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    if (velMag != 0 && vorticity[i] != 0) {
      helicity[i] /= (2 * velMag * vorticity[i]);
    } else {
      helicity[i] = 0;
    }
  }

#pragma omp parallel for
  for (size_t i = 0; i < len; ++i) {
    double J[3][3] = {{dux[i], duy[i], duz[i]},
        {dvx[i], dvy[i], dvz[i]},
        {dwx[i], dwy[i], dwz[i]}};

    lambda2[i] = -std::min(l2(J), 0.0);
    qCriterion[i] = std::max(q_crit(J), 0.0);
  }
  std::cout << "[vort] Vortical variables computed" << std::endl;
}
