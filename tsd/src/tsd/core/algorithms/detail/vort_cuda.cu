// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/algorithms/vort_cuda.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// cuda
#include <cuda_runtime.h>
// std
#include <cmath>
#include <cstdint>

namespace tsd::core {

// ---------------------------------------------------------------------------
// Device functions: lambda2 and Q-criterion (mirrors vort.h)
// ---------------------------------------------------------------------------

__device__ static double l2_d(double j00,
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
    double e0 = a, e1 = d, e2 = f;
    if (e0 < e1) {
      double t = e0;
      e0 = e1;
      e1 = t;
    }
    if (e1 < e2) {
      double t = e1;
      e1 = e2;
      e2 = t;
    }
    if (e0 < e1) {
      double t = e0;
      e0 = e1;
      e1 = t;
    }
    (void)e0;
    (void)e2;
    return q + e1;
  }
  const double p = sqrt((a * a + d * d + f * f + 2.0 * p1) / 6.0);
  const double r = (a * (d * f - m12 * m12) - m01 * (m01 * f - m12 * m02)
                       + m02 * (m01 * m12 - d * m02))
      / (2.0 * p * p * p);
  const double phi = acos(fmax(-1.0, fmin(1.0, r))) / 3.0;
  const double e0 = q + 2.0 * p * cos(phi);
  const double e2 = q + 2.0 * p * cos(phi + 2.0943951023931953);
  return 3.0 * q - e0 - e2;
}

__device__ static double q_crit_d(double j00,
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
// grad1 — one directional gradient for element tid.
//
// i      : index of this element along the chosen axis
// n      : extent along the chosen axis
// s      : stride in the flat array along the chosen axis
// c      : coordinate array along the chosen axis (length n)
//
// Boundary: 1st-order one-sided.  Interior: 2nd-order central.
// ---------------------------------------------------------------------------

__device__ static double grad1(
    const float *fc, size_t tid, size_t i, size_t n, size_t s, const double *c)
{
  if (i == 0)
    return ((double)fc[tid + s] - (double)fc[tid]) / (c[1] - c[0]);
  if (i == n - 1)
    return ((double)fc[tid] - (double)fc[tid - s]) / (c[n - 1] - c[n - 2]);
  return ((double)fc[tid + s] - (double)fc[tid - s]) / (c[i + 1] - c[i - 1]);
}

// ---------------------------------------------------------------------------
// vort_fused_kernel — one thread per voxel.
//
// Computes all 9 Jacobian components inline (register-resident) and
// immediately produces the requested vortical output(s).  Eliminates the
// 9 intermediate gradient arrays and the 9 separate grad3D kernel launches
// that existed in the previous split formulation.
// ---------------------------------------------------------------------------

__global__ static void vort_fused_kernel(const float *u,
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
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= len)
    return;

  const size_t slab = nx * ny;
  const size_t iz = tid / slab;
  const size_t iy = (tid % slab) / nx;
  const size_t ix = tid % nx;

  // All 9 Jacobian components — kept in registers, never written to DRAM.
  // Convention: d<vel><coord>, e.g. duy = du/dy.
  const double dux = grad1(u, tid, ix, nx, 1, x_);
  const double duy = grad1(u, tid, iy, ny, nx, y_);
  const double duz = grad1(u, tid, iz, nz, slab, z_);
  const double dvx = grad1(v, tid, ix, nx, 1, x_);
  const double dvy = grad1(v, tid, iy, ny, nx, y_);
  const double dvz = grad1(v, tid, iz, nz, slab, z_);
  const double dwx = grad1(w, tid, ix, nx, 1, x_);
  const double dwy = grad1(w, tid, iy, ny, nx, y_);
  const double dwz = grad1(w, tid, iz, nz, slab, z_);

  if (vorticity || helicity) {
    const double omx = dwy - dvz;
    const double omy = duz - dwx;
    const double omz = dvx - duy;
    const double omag = sqrt(omx * omx + omy * omy + omz * omz);
    if (vorticity)
      vorticity[tid] = (float)omag;
    if (helicity) {
      const double ui = u[tid], vi = v[tid], wi = w[tid];
      const double h = fabs(omx * ui + omy * vi + omz * wi);
      const double vmag = sqrt(ui * ui + vi * vi + wi * wi);
      helicity[tid] =
          (vmag > 0.0 && omag > 0.0) ? (float)(h / (2.0 * vmag * omag)) : 0.0f;
    }
  }
  if (lambda2)
    lambda2[tid] =
        (float)(-fmin(l2_d(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz), 0.0));
  if (qCriterion)
    qCriterion[tid] =
        (float)fmax(q_crit_d(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz), 0.0);
}

// ---------------------------------------------------------------------------
// vort_cuda — host entry point
// ---------------------------------------------------------------------------

void vort_cuda(const float *u,
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
  const int BLOCK = 256;
  const int grid = (int)((len + BLOCK - 1) / BLOCK);

  tsd::core::logStatus(
      "[vort_cuda] computing on %zux%zux%zu grid (%zu voxels)...",
      nx,
      ny,
      nz,
      len);

  // Device pointers — all null-initialized so cudaFree(nullptr) is safe on
  // any early-exit path.
  float *d_u{}, *d_v{}, *d_w{};
  double *d_x{}, *d_y{}, *d_z{};
  float *d_vort{}, *d_hel{}, *d_l2{}, *d_qc{};

  bool ok = true;
  auto check = [&](cudaError_t e) {
    if (e != cudaSuccess && ok) {
      tsd::core::logError("[vort_cuda] CUDA error: %s", cudaGetErrorString(e));
      ok = false;
    }
  };

  // Allocate and upload velocity fields
  check(cudaMalloc(&d_u, len * sizeof(float)));
  check(cudaMalloc(&d_v, len * sizeof(float)));
  check(cudaMalloc(&d_w, len * sizeof(float)));
  if (ok) {
    check(cudaMemcpy(d_u, u, len * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_v, v, len * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_w, w, len * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Allocate and upload coordinate arrays
  check(cudaMalloc(&d_x, nx * sizeof(double)));
  check(cudaMalloc(&d_y, ny * sizeof(double)));
  check(cudaMalloc(&d_z, nz * sizeof(double)));
  if (ok) {
    check(cudaMemcpy(d_x, x_, nx * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_y, y_, ny * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_z, z_, nz * sizeof(double), cudaMemcpyHostToDevice));
  }

  // Allocate output buffers
  if (vorticity)
    check(cudaMalloc(&d_vort, len * sizeof(float)));
  if (helicity)
    check(cudaMalloc(&d_hel, len * sizeof(float)));
  if (lambda2)
    check(cudaMalloc(&d_l2, len * sizeof(float)));
  if (qCriterion)
    check(cudaMalloc(&d_qc, len * sizeof(float)));

  if (ok) {
    // Single fused kernel: computes all 9 Jacobian components per thread in
    // registers and immediately derives the requested vortical quantities.
    // Eliminates 9x len*sizeof(double) of intermediate global memory traffic.
    vort_fused_kernel<<<grid, BLOCK>>>(
        d_u, d_v, d_w, d_x, d_y, d_z, d_vort, d_hel, d_l2, d_qc, nx, ny, nz);

    check(cudaDeviceSynchronize());
  }

  if (ok) {
    if (vorticity)
      check(cudaMemcpy(
          vorticity, d_vort, len * sizeof(float), cudaMemcpyDeviceToHost));
    if (helicity)
      check(cudaMemcpy(
          helicity, d_hel, len * sizeof(float), cudaMemcpyDeviceToHost));
    if (lambda2)
      check(cudaMemcpy(
          lambda2, d_l2, len * sizeof(float), cudaMemcpyDeviceToHost));
    if (qCriterion)
      check(cudaMemcpy(
          qCriterion, d_qc, len * sizeof(float), cudaMemcpyDeviceToHost));
  }

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_vort);
  cudaFree(d_hel);
  cudaFree(d_l2);
  cudaFree(d_qc);
}

// ---------------------------------------------------------------------------
// Green-Gauss gradient for unstructured meshes
// ---------------------------------------------------------------------------

// Returns the number of triangular faces for a given VTK cell type.
// Supported: TETRA(10)=4, HEX(12)=12, WEDGE(13)=8, PYRAMID(14)=6.
__device__ static int num_tris_for_type(uint8_t ct)
{
  if (ct == 10)
    return 4;
  if (ct == 12)
    return 12;
  if (ct == 13)
    return 8;
  if (ct == 14)
    return 6;
  return 0;
}

// Returns the expected number of vertices for a given VTK cell type.
__device__ static uint32_t expected_nv(uint8_t ct)
{
  if (ct == 10)
    return 4; // TETRA
  if (ct == 12)
    return 8; // HEX
  if (ct == 13)
    return 6; // WEDGE
  if (ct == 14)
    return 5; // PYRAMID
  return 0;
}

// Fills (i0,i1,i2) with the local vertex indices of the k-th triangulated
// face of the cell.  Faces follow VTK element ordering; quad faces are split
// into two triangles (a,b,c)+(a,c,d).  Orientation is corrected per-triangle
// against the cell centroid inside the kernel, so winding order here only
// needs to be internally consistent.
__device__ static void tri_indices(uint8_t ct, int k, int &i0, int &i1, int &i2)
{
  if (ct == 10) { // TETRA: 4 triangular faces
    // VTK faces: {0,1,3},{1,2,3},{2,0,3},{0,2,1}
    const int t[4][3] = {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}, {0, 2, 1}};
    i0 = t[k][0];
    i1 = t[k][1];
    i2 = t[k][2];
  } else if (ct == 12) { // HEX: 6 quad faces → 12 triangles
    // VTK faces: {0,4,7,3},{1,2,6,5},{0,1,5,4},{3,7,6,2},{0,3,2,1},{4,5,6,7}
    const int t[12][3] = {
        {0, 4, 7},
        {0, 7, 3}, // face 0
        {1, 2, 6},
        {1, 6, 5}, // face 1
        {0, 1, 5},
        {0, 5, 4}, // face 2
        {3, 7, 6},
        {3, 6, 2}, // face 3
        {0, 3, 2},
        {0, 2, 1}, // face 4
        {4, 5, 6},
        {4, 6, 7}, // face 5
    };
    i0 = t[k][0];
    i1 = t[k][1];
    i2 = t[k][2];
  } else if (ct == 13) { // WEDGE: 2 tri + 3 quad faces → 8 triangles
    // VTK faces: {0,1,2},{3,5,4},{0,3,4,1},{1,4,5,2},{2,5,3,0}
    const int t[8][3] = {
        {0, 1, 2},
        {3, 5, 4}, // tri faces
        {0, 3, 4},
        {0, 4, 1}, // quad face {0,3,4,1}
        {1, 4, 5},
        {1, 5, 2}, // quad face {1,4,5,2}
        {2, 5, 3},
        {2, 3, 0}, // quad face {2,5,3,0}
    };
    i0 = t[k][0];
    i1 = t[k][1];
    i2 = t[k][2];
  } else { // PYRAMID (14): 1 quad + 4 tri faces → 6 triangles
    // VTK faces: {0,3,2,1},{0,1,4},{1,2,4},{2,3,4},{3,0,4}
    const int t[6][3] = {
        {0, 3, 2},
        {0, 2, 1}, // quad base {0,3,2,1}
        {0, 1, 4},
        {1, 2, 4},
        {2, 3, 4},
        {3, 0, 4},
    };
    i0 = t[k][0];
    i1 = t[k][1];
    i2 = t[k][2];
  }
}

// ---------------------------------------------------------------------------
// gg_cell_kernel — Pass 1: Green-Gauss cell-centered gradient.
// One thread per cell.  Called once per scalar (u, v, w).
// Cell volume is geometry-only (same for all scalars); written every call.
// ---------------------------------------------------------------------------

__global__ static void gg_cell_kernel(const float *pos,
    const float *scalar,
    const uint32_t *conn,
    const uint32_t *cellIdx,
    const uint8_t *cellTypes,
    size_t numCells,
    size_t connSize,
    float3 *cellGrad,
    float *cellVol)
{
  const size_t ci = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ci >= numCells)
    return;

  const uint32_t start = cellIdx[ci];
  const uint32_t end =
      (ci + 1 < numCells) ? cellIdx[ci + 1] : (uint32_t)connSize;
  const uint32_t nv = end - start;
  const uint8_t ct = cellTypes[ci];

  const int ntris = num_tris_for_type(ct);
  if (ntris == 0 || nv != expected_nv(ct) || nv > 8) {
    cellGrad[ci] = {0.f, 0.f, 0.f};
    cellVol[ci] = 0.f;
    return;
  }

  // Load vertex positions and scalar values into registers (max 8 verts).
  float px[8], py[8], pz[8], sv[8];
  float cx = 0.f, cy = 0.f, cz = 0.f; // centroid
  for (uint32_t j = 0; j < nv; ++j) {
    uint32_t vid = conn[start + j];
    px[j] = pos[3 * vid];
    py[j] = pos[3 * vid + 1];
    pz[j] = pos[3 * vid + 2];
    sv[j] = scalar[vid];
    cx += px[j];
    cy += py[j];
    cz += pz[j];
  }
  const float inv_nv = 1.f / (float)nv;
  cx *= inv_nv;
  cy *= inv_nv;
  cz *= inv_nv;

  double gx = 0, gy = 0, gz = 0; // gradient accumulator
  double vol = 0; // divergence-theorem volume accumulator (= 3 * true volume)

  for (int t = 0; t < ntris; ++t) {
    int i0, i1, i2;
    tri_indices(ct, t, i0, i1, i2);

    // Edge vectors
    const double e1x = px[i1] - px[i0], e1y = py[i1] - py[i0],
                 e1z = pz[i1] - pz[i0];
    const double e2x = px[i2] - px[i0], e2y = py[i2] - py[i0],
                 e2z = pz[i2] - pz[i0];

    // Area-weighted face normal N = e1 × e2
    double Nx = e1y * e2z - e1z * e2y;
    double Ny = e1z * e2x - e1x * e2z;
    double Nz = e1x * e2y - e1y * e2x;

    // Correct orientation: N must point away from centroid
    const double d =
        Nx * (px[i0] - cx) + Ny * (py[i0] - cy) + Nz * (pz[i0] - cz);
    if (d < 0.0) {
      Nx = -Nx;
      Ny = -Ny;
      Nz = -Nz;
    }

    // Green-Gauss: ∇f += f_face * N (face-average scalar)
    const double f_face =
        ((double)sv[i0] + (double)sv[i1] + (double)sv[i2]) / 3.0;
    gx += f_face * Nx;
    gy += f_face * Ny;
    gz += f_face * Nz;

    // Divergence-theorem volume: V = (1/3) Σ (face_centroid · N_outward)
    const double fcx = ((double)px[i0] + px[i1] + px[i2]) / 3.0;
    const double fcy = ((double)py[i0] + py[i1] + py[i2]) / 3.0;
    const double fcz = ((double)pz[i0] + pz[i1] + pz[i2]) / 3.0;
    vol += fcx * Nx + fcy * Ny + fcz * Nz;
  }

  // vol = Σ(face_centroid · N) = 3 * true_volume
  const double true_vol = vol / 3.0;
  if (true_vol > 1e-30) {
    cellGrad[ci] = {
        (float)(gx / true_vol), (float)(gy / true_vol), (float)(gz / true_vol)};
  } else {
    cellGrad[ci] = {0.f, 0.f, 0.f};
  }
  cellVol[ci] = (float)true_vol;
}

// ---------------------------------------------------------------------------
// gg_vertex_kernel — Pass 2: Volume-weighted cell→vertex gradient average.
// One thread per vertex.
// ---------------------------------------------------------------------------

__global__ static void gg_vertex_kernel(const uint32_t *vtxCellOffsets,
    const uint32_t *vtxCellList,
    const float3 *cellGrad,
    const float *cellVol,
    size_t numPoints,
    float3 *vertGrad)
{
  const size_t vi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (vi >= numPoints)
    return;

  const uint32_t cstart = vtxCellOffsets[vi];
  const uint32_t cend = vtxCellOffsets[vi + 1];

  double gx = 0, gy = 0, gz = 0, totalVol = 0;
  for (uint32_t k = cstart; k < cend; ++k) {
    const uint32_t ci = vtxCellList[k];
    const double vol = (double)cellVol[ci];
    gx += vol * (double)cellGrad[ci].x;
    gy += vol * (double)cellGrad[ci].y;
    gz += vol * (double)cellGrad[ci].z;
    totalVol += vol;
  }

  if (totalVol > 0.0) {
    vertGrad[vi] = {
        (float)(gx / totalVol), (float)(gy / totalVol), (float)(gz / totalVol)};
  } else {
    vertGrad[vi] = {0.f, 0.f, 0.f};
  }
}

// ---------------------------------------------------------------------------
// vort_jacobian_kernel — Pass 3: Vorticity from pre-computed vertex Jacobian.
// One thread per vertex.
// ---------------------------------------------------------------------------

__global__ static void vort_jacobian_kernel(const float *u,
    const float *v,
    const float *w,
    const float3 *gradU,
    const float3 *gradV,
    const float3 *gradW,
    size_t numPoints,
    float *vorticity,
    float *helicity,
    float *lambda2,
    float *qCriterion)
{
  const size_t vi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (vi >= numPoints)
    return;

  // Jacobian rows: gradU = (du/dx, du/dy, du/dz), etc.
  const double dux = gradU[vi].x, duy = gradU[vi].y, duz = gradU[vi].z;
  const double dvx = gradV[vi].x, dvy = gradV[vi].y, dvz = gradV[vi].z;
  const double dwx = gradW[vi].x, dwy = gradW[vi].y, dwz = gradW[vi].z;

  if (vorticity || helicity) {
    const double omx = dwy - dvz;
    const double omy = duz - dwx;
    const double omz = dvx - duy;
    const double omag = sqrt(omx * omx + omy * omy + omz * omz);
    if (vorticity)
      vorticity[vi] = (float)omag;
    if (helicity) {
      const double ui = u[vi], vi_ = v[vi], wi = w[vi];
      const double h = fabs(omx * ui + omy * vi_ + omz * wi);
      const double vmag = sqrt(ui * ui + vi_ * vi_ + wi * wi);
      helicity[vi] =
          (vmag > 0.0 && omag > 0.0) ? (float)(h / (2.0 * vmag * omag)) : 0.0f;
    }
  }
  if (lambda2)
    lambda2[vi] =
        (float)(-fmin(l2_d(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz), 0.0));
  if (qCriterion)
    qCriterion[vi] =
        (float)fmax(q_crit_d(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz), 0.0);
}

// ---------------------------------------------------------------------------
// vort_cuda_unstructured — host entry point
// ---------------------------------------------------------------------------

void vort_cuda_unstructured(const float *positions,
    const float *u,
    const float *v,
    const float *w,
    const uint32_t *connectivity,
    const uint32_t *cellIndex,
    const uint8_t *cellTypes,
    const uint32_t *vtxCellOffsets,
    const uint32_t *vtxCellList,
    size_t numPoints,
    size_t numCells,
    size_t connSize,
    float *vorticity,
    float *helicity,
    float *lambda2,
    float *qCriterion)
{
  if (!vorticity && !helicity && !lambda2 && !qCriterion)
    return;

  tsd::core::logStatus(
      "[vort_cuda_unstructured] computing on %zu points, %zu cells...",
      numPoints,
      numCells);

  const int BLOCK = 256;
  const int gridP = (int)((numPoints + BLOCK - 1) / BLOCK);
  const int gridC = (int)((numCells + BLOCK - 1) / BLOCK);
  const size_t listSize = vtxCellOffsets[numPoints];

  bool ok = true;
  auto check = [&](cudaError_t e) {
    if (e != cudaSuccess && ok) {
      tsd::core::logError(
          "[vort_cuda_unstructured] CUDA error: %s", cudaGetErrorString(e));
      ok = false;
    }
  };

  // Device pointers — all null-initialized so cudaFree(nullptr) is safe.
  float *d_pos{}, *d_u{}, *d_v{}, *d_w{};
  uint32_t *d_conn{}, *d_cellIdx{};
  uint8_t *d_ct{};
  uint32_t *d_vco{}, *d_vcl{};
  float3 *d_cgU{}, *d_cgV{}, *d_cgW{}; // cell-centered gradients
  float *d_cellVol{};
  float3 *d_vgU{}, *d_vgV{}, *d_vgW{}; // vertex-centered gradients
  float *d_vort{}, *d_hel{}, *d_l2{}, *d_qc{};

  // Allocate topology + scalar inputs
  check(cudaMalloc(&d_pos, numPoints * 3 * sizeof(float)));
  check(cudaMalloc(&d_u, numPoints * sizeof(float)));
  check(cudaMalloc(&d_v, numPoints * sizeof(float)));
  check(cudaMalloc(&d_w, numPoints * sizeof(float)));
  check(cudaMalloc(&d_conn, connSize * sizeof(uint32_t)));
  check(cudaMalloc(&d_cellIdx, numCells * sizeof(uint32_t)));
  check(cudaMalloc(&d_ct, numCells * sizeof(uint8_t)));
  check(cudaMalloc(&d_vco, (numPoints + 1) * sizeof(uint32_t)));
  check(cudaMalloc(&d_vcl, listSize * sizeof(uint32_t)));
  // Intermediate gradient + volume arrays
  check(cudaMalloc(&d_cgU, numCells * sizeof(float3)));
  check(cudaMalloc(&d_cgV, numCells * sizeof(float3)));
  check(cudaMalloc(&d_cgW, numCells * sizeof(float3)));
  check(cudaMalloc(&d_cellVol, numCells * sizeof(float)));
  check(cudaMalloc(&d_vgU, numPoints * sizeof(float3)));
  check(cudaMalloc(&d_vgV, numPoints * sizeof(float3)));
  check(cudaMalloc(&d_vgW, numPoints * sizeof(float3)));
  // Output arrays
  if (vorticity)
    check(cudaMalloc(&d_vort, numPoints * sizeof(float)));
  if (helicity)
    check(cudaMalloc(&d_hel, numPoints * sizeof(float)));
  if (lambda2)
    check(cudaMalloc(&d_l2, numPoints * sizeof(float)));
  if (qCriterion)
    check(cudaMalloc(&d_qc, numPoints * sizeof(float)));

  if (ok) {
    check(cudaMemcpy(d_pos,
        positions,
        numPoints * 3 * sizeof(float),
        cudaMemcpyHostToDevice));
    check(
        cudaMemcpy(d_u, u, numPoints * sizeof(float), cudaMemcpyHostToDevice));
    check(
        cudaMemcpy(d_v, v, numPoints * sizeof(float), cudaMemcpyHostToDevice));
    check(
        cudaMemcpy(d_w, w, numPoints * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_conn,
        connectivity,
        connSize * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_cellIdx,
        cellIndex,
        numCells * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    check(cudaMemcpy(
        d_ct, cellTypes, numCells * sizeof(uint8_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_vco,
        vtxCellOffsets,
        (numPoints + 1) * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_vcl,
        vtxCellList,
        listSize * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
  }

  if (ok) {
    // Pass 1+2 for u: cell-centered GG gradient → volume-weighted vertex avg
    gg_cell_kernel<<<gridC, BLOCK>>>(d_pos,
        d_u,
        d_conn,
        d_cellIdx,
        d_ct,
        numCells,
        connSize,
        d_cgU,
        d_cellVol);
    gg_vertex_kernel<<<gridP, BLOCK>>>(
        d_vco, d_vcl, d_cgU, d_cellVol, numPoints, d_vgU);

    // Pass 1+2 for v (cell volumes are geometry-only, same as u pass)
    gg_cell_kernel<<<gridC, BLOCK>>>(d_pos,
        d_v,
        d_conn,
        d_cellIdx,
        d_ct,
        numCells,
        connSize,
        d_cgV,
        d_cellVol);
    gg_vertex_kernel<<<gridP, BLOCK>>>(
        d_vco, d_vcl, d_cgV, d_cellVol, numPoints, d_vgV);

    // Pass 1+2 for w
    gg_cell_kernel<<<gridC, BLOCK>>>(d_pos,
        d_w,
        d_conn,
        d_cellIdx,
        d_ct,
        numCells,
        connSize,
        d_cgW,
        d_cellVol);
    gg_vertex_kernel<<<gridP, BLOCK>>>(
        d_vco, d_vcl, d_cgW, d_cellVol, numPoints, d_vgW);

    // Pass 3: vorticity/helicity/lambda2/Q from assembled Jacobian
    vort_jacobian_kernel<<<gridP, BLOCK>>>(d_u,
        d_v,
        d_w,
        d_vgU,
        d_vgV,
        d_vgW,
        numPoints,
        d_vort,
        d_hel,
        d_l2,
        d_qc);

    check(cudaDeviceSynchronize());
  }

  if (ok) {
    if (vorticity)
      check(cudaMemcpy(vorticity,
          d_vort,
          numPoints * sizeof(float),
          cudaMemcpyDeviceToHost));
    if (helicity)
      check(cudaMemcpy(
          helicity, d_hel, numPoints * sizeof(float), cudaMemcpyDeviceToHost));
    if (lambda2)
      check(cudaMemcpy(
          lambda2, d_l2, numPoints * sizeof(float), cudaMemcpyDeviceToHost));
    if (qCriterion)
      check(cudaMemcpy(
          qCriterion, d_qc, numPoints * sizeof(float), cudaMemcpyDeviceToHost));
  }

  cudaFree(d_pos);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_conn);
  cudaFree(d_cellIdx);
  cudaFree(d_ct);
  cudaFree(d_vco);
  cudaFree(d_vcl);
  cudaFree(d_cgU);
  cudaFree(d_cgV);
  cudaFree(d_cgW);
  cudaFree(d_cellVol);
  cudaFree(d_vgU);
  cudaFree(d_vgV);
  cudaFree(d_vgW);
  cudaFree(d_vort);
  cudaFree(d_hel);
  cudaFree(d_l2);
  cudaFree(d_qc);
}

} // namespace tsd::core
