# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Transfer function widget — interactive colour-map editor, opacity curve editor,
value-range control, opacity-scale and unit-distance sliders.

Works with any :class:`~tsd_client.client.TSDClient` (or subclass) that has
a scene graph with volumes.

Usage::

    from tsd_client import TSDClient
    from tsd_client.panels import TransferFunctionPanel

    client = TSDClient("host", 12345)
    tf = TransferFunctionPanel(client)
    display(tf)
"""

import json
import logging
from typing import TYPE_CHECKING

import anywidget
import traitlets

from ..utils import resolve_client, lerp as _lerp, clamp as _clamp

if TYPE_CHECKING:
    from ..client import TSDClient

logger = logging.getLogger("tsd_client.transfer_function")

# ---------------------------------------------------------------------------
# Preset colormaps — each returns 256 RGBA tuples (α = 1)
# ---------------------------------------------------------------------------
NUM_SAMPLES = 256


def _color_stops_to_samples(
    stops: list[tuple[float, float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Convert a list of (t, r, g, b, a) stops to NUM_SAMPLES RGBA tuples."""
    stops = sorted(stops, key=lambda s: s[0])
    out: list[tuple[float, float, float, float]] = []
    for i in range(NUM_SAMPLES):
        t = i / (NUM_SAMPLES - 1)
        lo = stops[0]
        hi = stops[-1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                lo = stops[j]
                hi = stops[j + 1]
                break
        span = hi[0] - lo[0]
        f = (t - lo[0]) / span if span > 1e-9 else 0.0
        r = _lerp(lo[1], hi[1], f)
        g = _lerp(lo[2], hi[2], f)
        b = _lerp(lo[3], hi[3], f)
        a = _lerp(lo[4], hi[4], f)
        out.append((_clamp(r), _clamp(g), _clamp(b), _clamp(a)))
    return out


_PRESET_DEFS: dict[str, list[tuple[float, float, float, float, float]]] = {
    # --- Sequential ---
    "Grayscale": [(0, 0, 0, 0, 1), (1, 1, 1, 1, 1)],
    "Viridis": [
        (0.00, 0.267, 0.004, 0.329, 1),
        (0.25, 0.283, 0.140, 0.458, 1),
        (0.50, 0.127, 0.570, 0.551, 1),
        (0.75, 0.454, 0.810, 0.335, 1),
        (1.00, 0.993, 0.906, 0.144, 1),
    ],
    "Plasma": [
        (0.00, 0.050, 0.030, 0.528, 1),
        (0.25, 0.494, 0.012, 0.658, 1),
        (0.50, 0.798, 0.195, 0.474, 1),
        (0.75, 0.973, 0.513, 0.149, 1),
        (1.00, 0.940, 0.975, 0.131, 1),
    ],
    "Inferno": [
        (0.00, 0.001, 0.000, 0.014, 1),
        (0.25, 0.342, 0.063, 0.429, 1),
        (0.50, 0.735, 0.216, 0.330, 1),
        (0.75, 0.978, 0.558, 0.035, 1),
        (1.00, 0.988, 0.998, 0.645, 1),
    ],
    "Magma": [
        (0.00, 0.001, 0.000, 0.014, 1),
        (0.25, 0.320, 0.060, 0.460, 1),
        (0.50, 0.716, 0.215, 0.475, 1),
        (0.75, 0.975, 0.550, 0.390, 1),
        (1.00, 0.987, 0.991, 0.750, 1),
    ],
    "Cividis": [
        (0.00, 0.000, 0.135, 0.305, 1),
        (0.25, 0.257, 0.310, 0.413, 1),
        (0.50, 0.471, 0.480, 0.431, 1),
        (0.75, 0.718, 0.661, 0.380, 1),
        (1.00, 0.995, 0.865, 0.196, 1),
    ],
    "Turbo": [
        (0.00, 0.190, 0.072, 0.232, 1),
        (0.15, 0.169, 0.450, 0.938, 1),
        (0.30, 0.063, 0.794, 0.831, 1),
        (0.45, 0.280, 0.945, 0.440, 1),
        (0.60, 0.693, 0.960, 0.170, 1),
        (0.75, 0.957, 0.740, 0.104, 1),
        (0.90, 0.950, 0.380, 0.070, 1),
        (1.00, 0.480, 0.015, 0.010, 1),
    ],
    # --- Classic ---
    "Jet": [
        (0.00, 0, 0, 0.5, 1),
        (0.11, 0, 0, 1, 1),
        (0.35, 0, 1, 1, 1),
        (0.50, 0, 1, 0, 1),
        (0.65, 1, 1, 0, 1),
        (0.89, 1, 0, 0, 1),
        (1.00, 0.5, 0, 0, 1),
    ],
    "Hot": [
        (0.00, 0.04, 0, 0, 1),
        (0.37, 1, 0, 0, 1),
        (0.73, 1, 1, 0, 1),
        (1.00, 1, 1, 1, 1),
    ],
    "Bone": [
        (0.00, 0, 0, 0, 1),
        (0.37, 0.33, 0.33, 0.44, 1),
        (0.75, 0.66, 0.78, 0.78, 1),
        (1.00, 1, 1, 1, 1),
    ],
    "Copper": [
        (0.00, 0, 0, 0, 1),
        (0.80, 1, 0.63, 0.40, 1),
        (1.00, 1, 0.78, 0.50, 1),
    ],
    "HSV": [
        (0.00, 1, 0, 0, 1),
        (0.17, 1, 1, 0, 1),
        (0.33, 0, 1, 0, 1),
        (0.50, 0, 1, 1, 1),
        (0.67, 0, 0, 1, 1),
        (0.83, 1, 0, 1, 1),
        (1.00, 1, 0, 0, 1),
    ],
    "Rainbow": [
        (0.00, 0.50, 0, 1, 1),
        (0.20, 0, 0, 1, 1),
        (0.40, 0, 1, 0.5, 1),
        (0.60, 1, 1, 0, 1),
        (0.80, 1, 0.5, 0, 1),
        (1.00, 1, 0, 0, 1),
    ],
    # --- Diverging ---
    "Cool\u2013Warm": [
        (0.0, 0.23, 0.30, 0.75, 1),
        (0.5, 0.87, 0.87, 0.87, 1),
        (1.0, 0.71, 0.016, 0.15, 1),
    ],
    "RdBu": [
        (0.0, 0.40, 0.00, 0.05, 1),
        (0.25, 0.84, 0.38, 0.30, 1),
        (0.5, 0.97, 0.97, 0.97, 1),
        (0.75, 0.36, 0.58, 0.81, 1),
        (1.0, 0.02, 0.19, 0.58, 1),
    ],
    "Spectral": [
        (0.00, 0.62, 0.00, 0.26, 1),
        (0.25, 0.96, 0.43, 0.26, 1),
        (0.50, 1.00, 1.00, 0.75, 1),
        (0.75, 0.40, 0.76, 0.65, 1),
        (1.00, 0.37, 0.31, 0.64, 1),
    ],
    "PiYG": [
        (0.0, 0.56, 0.00, 0.32, 1),
        (0.25, 0.87, 0.55, 0.77, 1),
        (0.5, 0.97, 0.97, 0.97, 1),
        (0.75, 0.57, 0.82, 0.31, 1),
        (1.0, 0.15, 0.39, 0.10, 1),
    ],
    # --- Perceptual / scientific ---
    "GnBu": [
        (0.00, 0.97, 0.98, 0.96, 1),
        (0.25, 0.78, 0.91, 0.77, 1),
        (0.50, 0.50, 0.80, 0.77, 1),
        (0.75, 0.25, 0.58, 0.76, 1),
        (1.00, 0.03, 0.25, 0.51, 1),
    ],
    "YlOrRd": [
        (0.00, 1.00, 1.00, 0.80, 1),
        (0.25, 0.99, 0.85, 0.46, 1),
        (0.50, 0.99, 0.55, 0.24, 1),
        (0.75, 0.89, 0.15, 0.14, 1),
        (1.00, 0.50, 0.00, 0.15, 1),
    ],
    "BuPu": [
        (0.00, 0.97, 0.98, 0.99, 1),
        (0.25, 0.70, 0.80, 0.91, 1),
        (0.50, 0.55, 0.59, 0.83, 1),
        (0.75, 0.62, 0.30, 0.69, 1),
        (1.00, 0.30, 0.00, 0.42, 1),
    ],
    "Ice\u2013Fire": [
        (0.0, 0.00, 0.15, 0.35, 1),
        (0.25, 0.20, 0.50, 0.90, 1),
        (0.5, 0.95, 0.95, 0.95, 1),
        (0.75, 0.95, 0.45, 0.10, 1),
        (1.0, 0.35, 0.05, 0.00, 1),
    ],
    "Ocean": [
        (0.00, 0.00, 0.10, 0.15, 1),
        (0.30, 0.00, 0.25, 0.42, 1),
        (0.60, 0.00, 0.55, 0.55, 1),
        (0.85, 0.40, 0.85, 0.75, 1),
        (1.00, 0.90, 1.00, 0.95, 1),
    ],
    # --- Domain-specific ---
    "X-Ray": [(0, 1, 1, 1, 1), (1, 0, 0, 0, 1)],
    "CT Tissue": [
        (0.00, 0.00, 0.00, 0.00, 1),
        (0.20, 0.40, 0.20, 0.15, 1),
        (0.50, 0.85, 0.55, 0.40, 1),
        (0.75, 0.95, 0.85, 0.75, 1),
        (1.00, 1.00, 1.00, 1.00, 1),
    ],
    "CT Bone": [
        (0.00, 0.00, 0.00, 0.00, 1),
        (0.30, 0.30, 0.15, 0.10, 1),
        (0.60, 0.78, 0.70, 0.55, 1),
        (1.00, 1.00, 0.98, 0.90, 1),
    ],
    "Temperature": [
        (0.00, 0.02, 0.00, 0.38, 1),
        (0.20, 0.10, 0.30, 0.85, 1),
        (0.40, 0.30, 0.75, 0.90, 1),
        (0.50, 0.90, 0.95, 0.95, 1),
        (0.60, 0.95, 0.85, 0.30, 1),
        (0.80, 0.95, 0.35, 0.10, 1),
        (1.00, 0.50, 0.00, 0.00, 1),
    ],
    "Black Body": [
        (0.00, 0, 0, 0, 1),
        (0.33, 0.90, 0.15, 0, 1),
        (0.66, 1, 0.90, 0.20, 1),
        (1.00, 1, 1, 1, 1),
    ],
}

PRESETS: dict[str, list[tuple[float, float, float, float]]] = {
    name: _color_stops_to_samples(stops) for name, stops in _PRESET_DEFS.items()
}

DEFAULT_COLOR_STOPS = [
    {"t": 0.0, "r": 0, "g": 0, "b": 0},
    {"t": 1.0, "r": 1, "g": 1, "b": 1},
]

DEFAULT_OPACITY_POINTS = [
    {"x": 0.0, "y": 0.0},
    {"x": 1.0, "y": 1.0},
]

# ---------------------------------------------------------------------------
# ESM front-end
# ---------------------------------------------------------------------------
_ESM = r"""
export function render({ model, el }) {
  /* ---- constants / colours ---- */
  const C = {
    bg: '#252525', border: '#444', text: '#ccc', dim: '#888',
    accent: '#4a9eff', inputBg: '#333', canvasBg: '#1a1a1a',
    opFill: 'rgba(100,180,255,0.25)', opLine: 'rgba(100,180,255,0.85)',
    opPt: '#5af', opPtHover: '#8cf',
    colorBorder: '#fff', colorSel: '#ff0',
  };
  let W = 480;
  const H = 200;

  /* ---- root ---- */
  const root = document.createElement('div');
  root.style.cssText = `
    font-family: 'SF Mono','Fira Code',Consolas,monospace;
    font-size: 12px; color:${C.text}; background:${C.bg};
    border:1px solid ${C.border}; border-radius:6px;
    padding:10px 12px; min-width:280px; user-select:none;
    box-sizing:border-box; width:100%;
  `;

  /* helper */
  const mkRow = () => { const d = document.createElement('div'); d.style.cssText='display:flex;align-items:center;gap:6px;margin-bottom:6px;'; return d; };
  const mkLabel = (t) => { const s=document.createElement('span'); s.style.cssText=`color:${C.text};min-width:80px;`; s.textContent=t; return s; };
  const mkSelect = () => { const s=document.createElement('select'); s.style.cssText=`flex:1;background:${C.inputBg};color:${C.text};border:1px solid ${C.border};border-radius:4px;padding:3px 5px;`; return s; };
  const mkInput = (type,w) => { const i=document.createElement('input'); i.type=type; i.style.cssText=`width:${w||60}px;background:${C.inputBg};color:${C.text};border:1px solid ${C.border};border-radius:4px;padding:3px 5px;text-align:right;`; return i; };
  const mkBtn = (t,bg) => { const b=document.createElement('button'); b.textContent=t; b.style.cssText=`background:${bg||C.accent};color:#fff;border:none;border-radius:4px;padding:5px 10px;cursor:pointer;font-size:12px;`; return b; };
  const mkSlider = (min,max,step,val) => { const s=document.createElement('input'); s.type='range'; s.min=min; s.max=max; s.step=step; s.value=val; s.style.cssText='flex:1;'; return s; };

  /* title + status */
  const titleDiv = document.createElement('div');
  titleDiv.style.cssText = 'font-weight:600;margin-bottom:2px;';
  titleDiv.textContent = 'Transfer Function';
  root.appendChild(titleDiv);
  const statusDiv = document.createElement('div');
  statusDiv.style.cssText = `font-size:11px;color:${C.dim};margin-bottom:8px;`;
  statusDiv.textContent = model.get('_status') || '';
  model.on('change:_status', () => { statusDiv.textContent = model.get('_status')||''; });
  root.appendChild(statusDiv);

  /* ---- volume selector ---- */
  const volRow = mkRow();
  volRow.appendChild(mkLabel('Volume'));
  const volSel = mkSelect();
  volRow.appendChild(volSel);
  root.appendChild(volRow);

  function fillVolumes() {
    volSel.innerHTML = '';
    let list = [];
    try { list = JSON.parse(model.get('_volumes_json')||'[]'); } catch(e){}
    list.forEach(v => { const o=document.createElement('option'); o.value=JSON.stringify(v); o.textContent=v.name||`Volume ${v.index}`; volSel.appendChild(o); });
    if (!list.length) { const o=document.createElement('option'); o.value=''; o.textContent='No volumes'; volSel.appendChild(o); }
    onVolumeChanged();
  }
  model.on('change:_volumes_json', fillVolumes);

  /* ---- preset selector ---- */
  const preRow = mkRow();
  preRow.appendChild(mkLabel('Preset'));
  const preSel = mkSelect();
  preRow.appendChild(preSel);
  root.appendChild(preRow);

  function fillPresets() {
    preSel.innerHTML = '';
    const none = document.createElement('option');
    none.value = ''; none.textContent = '\u2014 custom \u2014';
    preSel.appendChild(none);
    let list = [];
    try { list = JSON.parse(model.get('_presets_list')||'[]'); } catch(e){}
    list.forEach(n => { const o=document.createElement('option'); o.value=n; o.textContent=n; preSel.appendChild(o); });
  }
  model.on('change:_presets_list', fillPresets);

  preSel.addEventListener('change', () => {
    const name = preSel.value;
    if (!name) return;
    model.set('_load_preset', { name, _t: Date.now() });
    model.save_changes();
  });

  /* ---- canvas area ---- */
  const canvasWrap = document.createElement('div');
  canvasWrap.style.cssText = `position:relative;width:100%;height:${H}px;margin:8px 0;`;

  const colorCanvas = document.createElement('canvas');
  colorCanvas.height = H;
  colorCanvas.style.cssText = `position:absolute;left:0;top:0;width:100%;height:${H}px;border-radius:4px;`;
  canvasWrap.appendChild(colorCanvas);

  const opCanvas = document.createElement('canvas');
  opCanvas.height = H;
  opCanvas.style.cssText = `position:absolute;left:0;top:0;width:100%;height:${H}px;border-radius:4px;cursor:crosshair;`;
  canvasWrap.appendChild(opCanvas);

  root.appendChild(canvasWrap);

  /* ---- colour stops row ---- */
  const colorStopsRow = document.createElement('div');
  colorStopsRow.style.cssText = `position:relative;width:100%;height:20px;margin-bottom:4px;`;
  root.appendChild(colorStopsRow);

  const colorHint = document.createElement('div');
  colorHint.style.cssText = `font-size:10px;color:${C.dim};margin-bottom:6px;`;
  colorHint.textContent = 'Dbl-click bar to add stop \u00b7 dbl-click stop to remove \u00b7 click stop to pick colour \u00b7 drag to move';
  root.appendChild(colorHint);

  /* ---- value range ---- */
  const vrRow = mkRow();
  vrRow.appendChild(mkLabel('Value range'));
  const vrMin = mkInput('number',70); vrMin.step='any';
  const vrMax = mkInput('number',70); vrMax.step='any';
  vrRow.appendChild(vrMin);
  const vrDash = document.createElement('span'); vrDash.textContent='\u2013'; vrRow.appendChild(vrDash);
  vrRow.appendChild(vrMax);
  const vrReset = mkBtn('Reset','#555');
  vrRow.appendChild(vrReset);
  root.appendChild(vrRow);

  /* ---- opacity scale ---- */
  const opRow = mkRow();
  opRow.appendChild(mkLabel('Opacity'));
  const opSlider = mkSlider(0, 1, 0.01, 1);
  opRow.appendChild(opSlider);
  const opVal = document.createElement('span'); opVal.style.cssText=`min-width:36px;text-align:right;color:${C.text};`; opVal.textContent='1.00';
  opRow.appendChild(opVal);
  root.appendChild(opRow);

  /* ---- unit distance ---- */
  const udRow = mkRow();
  udRow.appendChild(mkLabel('Unit dist.'));
  const udInput = mkInput('number',70); udInput.step='any'; udInput.min='0.001'; udInput.value='1.0';
  udRow.appendChild(udInput);
  root.appendChild(udRow);

  /* (no explicit apply row — changes are sent automatically) */

  /* ================================================================
   *  State
   * ================================================================ */
  const DEFAULT_COLOR_STOPS = [
    {t:0,r:0,g:0,b:0},{t:1,r:1,g:1,b:1}
  ];
  const DEFAULT_OPACITY_PTS = [
    {x:0,y:0},{x:1,y:1}
  ];

  let colorStops = JSON.parse(JSON.stringify(DEFAULT_COLOR_STOPS));
  let opacityPts = JSON.parse(JSON.stringify(DEFAULT_OPACITY_PTS));
  let serverValueRange = null;
  let curVolIdx = -1;
  const savedTF = new Map();

  function saveCurrentTF() {
    if (curVolIdx < 0) return;
    savedTF.set(curVolIdx, {
      colorStops: JSON.parse(JSON.stringify(colorStops)),
      opacityPts: JSON.parse(JSON.stringify(opacityPts)),
    });
  }

  function onVolumeChanged() {
    _suppressAuto = true;
    saveCurrentTF();
    try {
      const v = JSON.parse(volSel.value);
      const newIdx = v.index;
      if (v.valueRange) { vrMin.value=v.valueRange[0]; vrMax.value=v.valueRange[1]; serverValueRange=v.valueRange; }
      if (v.opacity !== undefined) { opSlider.value=v.opacity; opVal.textContent=Number(v.opacity).toFixed(2); }
      if (v.unitDistance !== undefined) { udInput.value=v.unitDistance; }

      const prev = savedTF.get(newIdx);
      if (prev) {
        colorStops = prev.colorStops;
        opacityPts = prev.opacityPts;
      } else {
        if (v.colorStops && v.colorStops.length >= 2) {
          colorStops = v.colorStops.map(s => ({t:s.t,r:s.r,g:s.g,b:s.b}));
        } else {
          colorStops = JSON.parse(JSON.stringify(DEFAULT_COLOR_STOPS));
        }
        const ocp = v.opacityControlPoints || v.opacityPoints;
        if (ocp && ocp.length >= 2) {
          opacityPts = ocp.map(p => Array.isArray(p) ? {x:p[0],y:p[1]} : {x:p.x,y:p.y});
        } else {
          opacityPts = JSON.parse(JSON.stringify(DEFAULT_OPACITY_PTS));
        }
      }
      curVolIdx = newIdx;
    } catch(e){}
    drawAll();
    _suppressAuto = false;
  }
  volSel.addEventListener('change', onVolumeChanged);

  vrReset.addEventListener('click', () => {
    if (serverValueRange) { vrMin.value=serverValueRange[0]; vrMax.value=serverValueRange[1]; autoApply(); }
  });
  vrMin.addEventListener('change', autoApply);
  vrMax.addEventListener('change', autoApply);
  udInput.addEventListener('change', autoApply);

  opSlider.addEventListener('input', () => { opVal.textContent = Number(opSlider.value).toFixed(2); autoApply(); });

  /* ================================================================
   *  Load preset from Python
   * ================================================================ */
  model.on('change:_preset_colors', () => {
    try {
      const stops = JSON.parse(model.get('_preset_colors')||'[]');
      if (stops.length >= 2) {
        colorStops = stops;
        drawAll();
      }
    } catch(e){}
  });

  /* ================================================================
   *  Drawing
   * ================================================================ */
  function drawColorBar() {
    const ctx = colorCanvas.getContext('2d');
    const sorted = [...colorStops].sort((a,b) => a.t - b.t);
    if (sorted.length < 2) { ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H); return; }
    const grad = ctx.createLinearGradient(0,0,W,0);
    sorted.forEach(s => {
      grad.addColorStop(s.t, `rgb(${Math.round(s.r*255)},${Math.round(s.g*255)},${Math.round(s.b*255)})`);
    });
    ctx.fillStyle = grad;
    ctx.fillRect(0,0,W,H);
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(0,0,W,H);
  }

  function drawOpacity() {
    const ctx = opCanvas.getContext('2d');
    ctx.clearRect(0,0,W,H);
    const pts = [...opacityPts].sort((a,b) => a.x - b.x);
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x*W, H);
    pts.forEach(p => ctx.lineTo(p.x*W, H - p.y*H));
    ctx.lineTo(pts[pts.length-1].x*W, H);
    ctx.closePath();
    ctx.fillStyle = C.opFill;
    ctx.fill();
    ctx.beginPath();
    pts.forEach((p,i) => { if(i===0) ctx.moveTo(p.x*W, H-p.y*H); else ctx.lineTo(p.x*W, H-p.y*H); });
    ctx.strokeStyle = C.opLine; ctx.lineWidth = 2; ctx.stroke();
    pts.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x*W, H-p.y*H, 5, 0, Math.PI*2);
      ctx.fillStyle = C.opPt; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
    });
  }

  function drawColorStops() {
    colorStopsRow.innerHTML = '';
    const sorted = [...colorStops].sort((a,b) => a.t - b.t);
    sorted.forEach((s, idx) => {
      const d = document.createElement('div');
      d.style.cssText = `position:absolute;left:${s.t*W - 6}px;top:0;width:12px;height:18px;border-radius:3px;border:2px solid ${C.colorBorder};cursor:pointer;`;
      d.style.background = `rgb(${Math.round(s.r*255)},${Math.round(s.g*255)},${Math.round(s.b*255)})`;
      d.title = `t=${s.t.toFixed(3)} \u00b7 dbl-click to remove \u00b7 click to edit colour`;

      let dragging = false, didDrag = false, startX = 0, startT = 0;
      d.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        dragging = true; didDrag = false; startX = e.clientX; startT = s.t;
        e.preventDefault();
        const onMove = (ev) => {
          if (!dragging) return;
          const dx = ev.clientX - startX;
          if (Math.abs(dx) > 2) didDrag = true;
          s.t = Math.max(0, Math.min(1, startT + dx/W));
          drawAll();
        };
        const onUp = () => { dragging = false; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
      });

      const picker = document.createElement('input');
      picker.type = 'color';
      picker.value = rgbToHex(s.r, s.g, s.b);
      picker.style.cssText = 'position:absolute;opacity:0;width:0;height:0;';
      d.appendChild(picker);
      d.addEventListener('click', (e) => {
        if (didDrag) return;
        picker.click();
      });
      picker.addEventListener('input', () => {
        const c = hexToRgb(picker.value);
        s.r = c.r; s.g = c.g; s.b = c.b;
        drawAll();
      });

      d.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        if (colorStops.length > 2) {
          const i = colorStops.indexOf(s);
          if (i >= 0) colorStops.splice(i, 1);
          drawAll();
        }
      });

      colorStopsRow.appendChild(d);
    });
  }

  let _suppressAuto = false;
  function drawAll() { drawColorBar(); drawOpacity(); drawColorStops(); if (!_suppressAuto) autoApply(); }

  colorStopsRow.addEventListener('dblclick', (e) => {
    const rect = colorStopsRow.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / W));
    const sorted = [...colorStops].sort((a,b) => a.t - b.t);
    let lo = sorted[0], hi = sorted[sorted.length-1];
    for (let i=0;i<sorted.length-1;i++) {
      if (sorted[i].t <= t && sorted[i+1].t >= t) { lo=sorted[i]; hi=sorted[i+1]; break; }
    }
    const span = hi.t - lo.t;
    const f = span > 1e-9 ? (t - lo.t)/span : 0;
    colorStops.push({
      t, r: lo.r+(hi.r-lo.r)*f, g: lo.g+(hi.g-lo.g)*f, b: lo.b+(hi.b-lo.b)*f
    });
    drawAll();
  });

  /* ---- opacity interaction ---- */
  let dragIdx = -1;
  function getOpPtAt(mx, my) {
    const sorted = [...opacityPts];
    for (let i=0;i<sorted.length;i++) {
      const px = sorted[i].x*W, py = H - sorted[i].y*H;
      if (Math.hypot(mx-px, my-py) < 8) return i;
    }
    return -1;
  }

  opCanvas.addEventListener('mousedown', (e) => {
    const rect = opCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    dragIdx = getOpPtAt(mx, my);
    if (dragIdx >= 0) e.preventDefault();
  });

  opCanvas.addEventListener('mousemove', (e) => {
    if (dragIdx < 0) return;
    const rect = opCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let nx = Math.max(0, Math.min(1, mx/W));
    let ny = Math.max(0, Math.min(1, 1 - my/H));
    const sorted = [...opacityPts].sort((a,b)=>a.x-b.x);
    const sortIdx = sorted.indexOf(opacityPts[dragIdx]);
    if (sortIdx === 0) nx = 0;
    if (sortIdx === sorted.length-1) nx = 1;
    opacityPts[dragIdx].x = nx;
    opacityPts[dragIdx].y = ny;
    drawAll();
  });

  opCanvas.addEventListener('mouseup', () => { dragIdx = -1; });
  opCanvas.addEventListener('mouseleave', () => { dragIdx = -1; });

  opCanvas.addEventListener('dblclick', (e) => {
    const rect = opCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const idx = getOpPtAt(mx, my);
    if (idx >= 0) {
      const sorted = [...opacityPts].sort((a,b)=>a.x-b.x);
      if (sorted.indexOf(opacityPts[idx]) > 0 && sorted.indexOf(opacityPts[idx]) < sorted.length-1) {
        opacityPts.splice(idx, 1);
      }
    } else {
      const nx = Math.max(0, Math.min(1, mx/W));
      const ny = Math.max(0, Math.min(1, 1-my/H));
      opacityPts.push({x:nx, y:ny});
    }
    drawAll();
  });

  /* ---- Apply (debounced, always automatic) ---- */
  let _applyTimer = null;
  function doApply() {
    let volIdx = -1;
    try { volIdx = JSON.parse(volSel.value).index; } catch(e){}
    if (volIdx < 0) return;
    curVolIdx = volIdx;
    saveCurrentTF();
    const sortedColors = [...colorStops].sort((a,b) => a.t - b.t);
    const sortedOp = [...opacityPts].sort((a,b) => a.x - b.x);
    model.set('_apply', {
      volumeIndex: volIdx,
      colorStops: sortedColors.map(s => ({t:s.t,r:s.r,g:s.g,b:s.b})),
      opacityPoints: sortedOp.map(p => ({x:p.x,y:p.y})),
      valueRange: [parseFloat(vrMin.value)||0, parseFloat(vrMax.value)||1],
      opacity: parseFloat(opSlider.value),
      unitDistance: parseFloat(udInput.value)||1,
      _t: Date.now()
    });
    model.save_changes();
  }
  function autoApply() {
    if (_applyTimer) clearTimeout(_applyTimer);
    _applyTimer = setTimeout(doApply, 120);
  }

  /* ---- helpers ---- */
  function rgbToHex(r,g,b) {
    const h = (v) => Math.round(v*255).toString(16).padStart(2,'0');
    return `#${h(r)}${h(g)}${h(b)}`;
  }
  function hexToRgb(hex) {
    const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex);
    return m ? {r:parseInt(m[1],16)/255,g:parseInt(m[2],16)/255,b:parseInt(m[3],16)/255} : {r:0,g:0,b:0};
  }

  /* ---- responsive resize ---- */
  function resizeCanvases() {
    const newW = canvasWrap.clientWidth;
    if (newW > 0 && newW !== W) {
      W = newW;
      colorCanvas.width = W;
      opCanvas.width = W;
      _suppressAuto = true;
      drawAll();
      _suppressAuto = false;
    }
  }
  const ro = new ResizeObserver(() => resizeCanvases());
  ro.observe(canvasWrap);

  /* ---- mount ---- */
  el.appendChild(root);
  fillVolumes();
  fillPresets();
  requestAnimationFrame(() => {
    resizeCanvases();
    _suppressAuto = true;
    drawAll();
    _suppressAuto = false;
  });
}
"""


class TransferFunctionPanel(anywidget.AnyWidget):
    """Interactive transfer-function editor for any TSD-based renderer.

    Features:

    - Volume selector (populated from the scene graph)
    - Colour-map: draggable colour stops with colour picker, preset loader
    - Opacity curve: draggable control points, add/remove by double-click
    - Value range, opacity scale, unit distance
    - Apply sends sampled RGBA + control points + scalar params to the server
    """

    _esm = traitlets.Unicode(_ESM).tag(sync=True)
    _volumes_json = traitlets.Unicode("[]").tag(sync=True)
    _presets_list = traitlets.Unicode("[]").tag(sync=True)
    _preset_colors = traitlets.Unicode("[]").tag(sync=True)
    _status = traitlets.Unicode("").tag(sync=True)
    _apply = traitlets.Dict({}).tag(sync=True)
    _load_preset = traitlets.Dict({}).tag(sync=True)

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        self._client = resolve_client(client)
        self._requested = False
        self._presets_list = json.dumps(list(PRESETS.keys()))
        self.observe(self._on_apply, names=["_apply"])
        self.observe(self._on_load_preset, names=["_load_preset"])

        prev_on_connect = client.on_connect

        def _chain_on_connect():
            if prev_on_connect:
                prev_on_connect()
            if not self._requested:
                self._request_volumes()

        client.on_connect = _chain_on_connect

        if self._client.connected and not self._requested:
            self._request_volumes()

    def _request_volumes(self):
        self._requested = True
        self._status = "Requesting volumes\u2026"
        try:
            sg = self._client.scene_graph
            if sg is None:
                self._volumes_json = "[]"
                self._status = "No volumes"
                return
            detailed = []
            for vol in sg.volumes:
                idx = vol.object_index
                info = None
                request_fn = getattr(self._client, "request_volume_info", None)
                if request_fn is not None:
                    try:
                        info = request_fn(idx, timeout=3.0)
                    except Exception:
                        pass
                if not info or not isinstance(info, dict):
                    info = sg.volume_info(idx)
                if info:
                    detailed.append(_normalize_tf_volume_info(info))
            self._volumes_json = json.dumps(detailed)
            self._status = f"{len(detailed)} volume(s)"
        except Exception:
            self._status = "Error"

    def _on_load_preset(self, change):
        cmd = change.get("new", {})
        if not cmd:
            return
        name = cmd.get("name")
        if not name or name not in _PRESET_DEFS:
            return
        stops = _PRESET_DEFS[name]
        color_stops = [
            {"t": s[0], "r": s[1], "g": s[2], "b": s[3]} for s in stops
        ]
        self._preset_colors = json.dumps(color_stops)
        self._status = f"Loaded: {name}"

    def _on_apply(self, change):
        cmd = change.get("new", {})
        if not cmd or not self._client.connected:
            return
        vol_idx = cmd.get("volumeIndex")
        if vol_idx is None:
            return

        sg = self._client.cached_scene_graph() or self._client.scene_graph
        if sg is None:
            return

        color_stops = cmd.get("colorStops", [])
        opacity_points = cmd.get("opacityPoints", [])
        value_range = cmd.get("valueRange")
        opacity_val = cmd.get("opacity")
        unit_dist = cmd.get("unitDistance")

        if len(color_stops) < 2 or len(opacity_points) < 2:
            return

        sorted_colors = sorted(color_stops, key=lambda s: s["t"])
        sorted_opacity = sorted(opacity_points, key=lambda p: p["x"])

        rgba_samples = []
        for i in range(NUM_SAMPLES):
            t = i / (NUM_SAMPLES - 1)
            lo_c = sorted_colors[0]
            hi_c = sorted_colors[-1]
            for j in range(len(sorted_colors) - 1):
                if sorted_colors[j]["t"] <= t <= sorted_colors[j + 1]["t"]:
                    lo_c = sorted_colors[j]
                    hi_c = sorted_colors[j + 1]
                    break
            span_c = hi_c["t"] - lo_c["t"]
            fc = (t - lo_c["t"]) / span_c if span_c > 1e-9 else 0.0
            r = _clamp(_lerp(lo_c["r"], hi_c["r"], fc))
            g = _clamp(_lerp(lo_c["g"], hi_c["g"], fc))
            b = _clamp(_lerp(lo_c["b"], hi_c["b"], fc))

            lo_o = sorted_opacity[0]
            hi_o = sorted_opacity[-1]
            for j in range(len(sorted_opacity) - 1):
                if sorted_opacity[j]["x"] <= t <= sorted_opacity[j + 1]["x"]:
                    lo_o = sorted_opacity[j]
                    hi_o = sorted_opacity[j + 1]
                    break
            span_o = hi_o["x"] - lo_o["x"]
            fo = (t - lo_o["x"]) / span_o if span_o > 1e-9 else 0.0
            a = _clamp(_lerp(lo_o["y"], hi_o["y"], fo))

            rgba_samples.append((r, g, b, a))

        op_xy = [(p["x"], p["y"]) for p in sorted_opacity]

        vr = tuple(value_range) if value_range else None
        sg.set_volume_tf(
            vol_idx,
            rgba_samples,
            op_xy,
            value_range=vr,
            opacity=opacity_val,
            unit_distance=unit_dist,
        )
        logger.info("Applied TF to volume %s", vol_idx)
        self._status = "Applied"


def _normalize_tf_volume_info(info: dict) -> dict:
    """Ensure valueRange and other TF fields are in the shape the frontend expects."""
    out = dict(info)
    vr = out.get("valueRange")
    if vr is not None and isinstance(vr, (list, tuple)) and len(vr) >= 2:
        out["valueRange"] = [float(vr[0]), float(vr[1])]
    elif out.get("valueRange") is None:
        out["valueRange"] = [0.0, 1.0]
    return out
