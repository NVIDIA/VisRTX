# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Transfer function widget – interactive colour-map editor, opacity curve editor,
value-range control, opacity-scale and unit-distance sliders.

Use with a connected TSDViewer (or subclass)::

    from tsdviewer import TSDViewer, TransferFunctionWidget
    viewer = TSDViewer("host", 12345)
    viewer.connect()
    tf = TransferFunctionWidget(viewer)
    display(viewer)
    display(tf)
"""

import json
import logging
from typing import TYPE_CHECKING

import anywidget
import traitlets

if TYPE_CHECKING:
    from .viewer import TSDViewer

logger = logging.getLogger("tsdviewer.transfer_function")

# ---------------------------------------------------------------------------
# Preset colormaps – each returns 256 RGBA tuples (α = 1)
# ---------------------------------------------------------------------------
NUM_SAMPLES = 256


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _color_stops_to_samples(
    stops: list[tuple[float, float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Convert a list of (t, r, g, b, a) stops to NUM_SAMPLES RGBA tuples."""
    stops = sorted(stops, key=lambda s: s[0])
    out: list[tuple[float, float, float, float]] = []
    for i in range(NUM_SAMPLES):
        t = i / (NUM_SAMPLES - 1)
        # find surrounding stops
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


# Preset definitions (stops: t, r, g, b, a)
_PRESET_DEFS: dict[str, list[tuple[float, float, float, float, float]]] = {
    "Grayscale": [(0, 0, 0, 0, 1), (1, 1, 1, 1, 1)],
    "Jet": [
        (0.00, 0, 0, 0.5, 1),
        (0.11, 0, 0, 1, 1),
        (0.35, 0, 1, 1, 1),
        (0.50, 0, 1, 0, 1),
        (0.65, 1, 1, 0, 1),
        (0.89, 1, 0, 0, 1),
        (1.00, 0.5, 0, 0, 1),
    ],
    "Viridis": [
        (0.00, 0.267, 0.004, 0.329, 1),
        (0.25, 0.283, 0.140, 0.458, 1),
        (0.50, 0.127, 0.570, 0.551, 1),
        (0.75, 0.454, 0.810, 0.335, 1),
        (1.00, 0.993, 0.906, 0.144, 1),
    ],
    "Cool–Warm": [
        (0.0, 0.23, 0.30, 0.75, 1),
        (0.5, 0.87, 0.87, 0.87, 1),
        (1.0, 0.71, 0.016, 0.15, 1),
    ],
    "Inferno": [
        (0.00, 0.001, 0.000, 0.014, 1),
        (0.25, 0.342, 0.063, 0.429, 1),
        (0.50, 0.735, 0.216, 0.330, 1),
        (0.75, 0.978, 0.558, 0.035, 1),
        (1.00, 0.988, 0.998, 0.645, 1),
    ],
}

PRESETS: dict[str, list[tuple[float, float, float, float]]] = {
    name: _color_stops_to_samples(stops) for name, stops in _PRESET_DEFS.items()
}

# Default colour control points (JSON):  [{t, r, g, b}]
# (alpha is always 1 in colour stops – separate opacity curve)
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
  const preApply = mkBtn('Load');
  preRow.appendChild(preApply);
  root.appendChild(preRow);

  function fillPresets() {
    preSel.innerHTML = '';
    let list = [];
    try { list = JSON.parse(model.get('_presets_list')||'[]'); } catch(e){}
    list.forEach(n => { const o=document.createElement('option'); o.value=n; o.textContent=n; preSel.appendChild(o); });
  }
  model.on('change:_presets_list', fillPresets);

  preApply.addEventListener('click', () => {
    const name = preSel.value;
    if (!name) return;
    model.set('_load_preset', { name, _t: Date.now() });
    model.save_changes();
  });

  /* ---- canvas area ---- */
  const canvasWrap = document.createElement('div');
  canvasWrap.style.cssText = `position:relative;width:100%;height:${H}px;margin:8px 0;`;

  /* colour-bar canvas (background gradient) */
  const colorCanvas = document.createElement('canvas');
  colorCanvas.height = H;
  colorCanvas.style.cssText = `position:absolute;left:0;top:0;width:100%;height:${H}px;border-radius:4px;`;
  canvasWrap.appendChild(colorCanvas);

  /* opacity overlay canvas */
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
  colorHint.textContent = 'Dbl-click bar to add stop · dbl-click stop to remove · click stop to pick colour · drag to move';
  root.appendChild(colorHint);

  /* ---- value range ---- */
  const vrRow = mkRow();
  vrRow.appendChild(mkLabel('Value range'));
  const vrMin = mkInput('number',70); vrMin.step='any';
  const vrMax = mkInput('number',70); vrMax.step='any';
  vrRow.appendChild(vrMin);
  const vrDash = document.createElement('span'); vrDash.textContent='–'; vrRow.appendChild(vrDash);
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

  /* ---- apply ---- */
  const actRow = mkRow();
  actRow.style.justifyContent = 'flex-end';
  const autoLbl = document.createElement('label');
  autoLbl.style.cssText = `display:flex;align-items:center;gap:4px;color:${C.text};cursor:pointer;margin-right:auto;`;
  const autoCb = document.createElement('input');
  autoCb.type = 'checkbox';
  autoCb.style.cursor = 'pointer';
  autoLbl.appendChild(autoCb);
  autoLbl.appendChild(document.createTextNode('Auto-update'));
  actRow.appendChild(autoLbl);
  const applyBtn = mkBtn('Apply');
  actRow.appendChild(applyBtn);
  root.appendChild(actRow);

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

  function onVolumeChanged() {
    try {
      const v = JSON.parse(volSel.value);
      if (v.valueRange) { vrMin.value=v.valueRange[0]; vrMax.value=v.valueRange[1]; serverValueRange=v.valueRange; }
      if (v.opacity !== undefined) { opSlider.value=v.opacity; opVal.textContent=Number(v.opacity).toFixed(2); }
      if (v.unitDistance !== undefined) { udInput.value=v.unitDistance; }
      if (v.opacityPoints && v.opacityPoints.length >= 2) {
        opacityPts = v.opacityPoints.map(p => ({x:p[0],y:p[1]}));
      }
    } catch(e){}
    drawAll();
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
    /* dim overlay for alpha visual */
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(0,0,W,H);
  }

  function drawOpacity() {
    const ctx = opCanvas.getContext('2d');
    ctx.clearRect(0,0,W,H);
    const pts = [...opacityPts].sort((a,b) => a.x - b.x);
    if (pts.length < 2) return;
    /* filled area */
    ctx.beginPath();
    ctx.moveTo(pts[0].x*W, H);
    pts.forEach(p => ctx.lineTo(p.x*W, H - p.y*H));
    ctx.lineTo(pts[pts.length-1].x*W, H);
    ctx.closePath();
    ctx.fillStyle = C.opFill;
    ctx.fill();
    /* line */
    ctx.beginPath();
    pts.forEach((p,i) => { if(i===0) ctx.moveTo(p.x*W, H-p.y*H); else ctx.lineTo(p.x*W, H-p.y*H); });
    ctx.strokeStyle = C.opLine; ctx.lineWidth = 2; ctx.stroke();
    /* points */
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
      d.title = `t=${s.t.toFixed(3)} · dbl-click to remove · click to edit colour`;

      /* drag */
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

      /* colour edit */
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

      /* double-click to remove (keep at least 2) */
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

  /* ---- add colour stop on clicking the bar area ---- */
  colorStopsRow.addEventListener('dblclick', (e) => {
    const rect = colorStopsRow.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / W));
    /* interpolate colour at t */
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
    const sorted = [...opacityPts]; // keep original order
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
    /* first and last points (by x) stay pinned horizontally */
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

  /* double-click to add opacity point */
  opCanvas.addEventListener('dblclick', (e) => {
    const rect = opCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const idx = getOpPtAt(mx, my);
    if (idx >= 0) {
      /* remove (keep endpoints) */
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

  /* ---- Apply ---- */
  function doApply() {
    let volIdx = -1;
    try { volIdx = JSON.parse(volSel.value).index; } catch(e){}
    if (volIdx < 0) return;
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
  function autoApply() { if (autoCb.checked) doApply(); }
  applyBtn.addEventListener('click', doApply);

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
  /* defer initial draw so layout has resolved the actual width */
  requestAnimationFrame(() => {
    resizeCanvases();
    _suppressAuto = true;
    drawAll();
    _suppressAuto = false;
  });
}
"""


class TransferFunctionWidget(anywidget.AnyWidget):
    """
    Interactive transfer-function editor.

    Features:
    - Volume selector (populated from the server)
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

    def __init__(self, viewer: "TSDViewer", **kwargs):
        super().__init__(**kwargs)
        self._viewer = viewer
        self._client = viewer._client
        self._requested = False
        self._presets_list = json.dumps(list(PRESETS.keys()))
        self.observe(self._on_apply, names=["_apply"])
        self.observe(self._on_load_preset, names=["_load_preset"])
        viewer.observe(self._when_viewer_connected, names=["_status"])
        if self._client.connected and not self._requested:
            self._request_volumes()

    # ---- connection --------------------------------------------------------

    def _when_viewer_connected(self, change):
        new = change.get("new", "") or ""
        if "Connected" in new and self._client.connected and not self._requested:
            self._request_volumes()

    def _request_volumes(self):
        self._requested = True
        self._status = "Requesting volumes…"
        try:
            vol_list = self._client.request_volume_list(timeout=5.0)
            if not vol_list:
                self._volumes_json = "[]"
                self._status = "No volumes"
                return
            detailed = []
            for v in vol_list:
                idx = v.get("index")
                if idx is None:
                    continue
                try:
                    info = self._client.request_volume_info(idx, timeout=5.0)
                    detailed.append(info)
                except TimeoutError:
                    detailed.append(v)
            self._volumes_json = json.dumps(detailed)
            self._status = f"{len(detailed)} volume(s)"
        except TimeoutError:
            self._status = "Timeout"
        except Exception:
            self._status = "Error"

    # ---- preset loader -----------------------------------------------------

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
        self._status = f"Loaded preset: {name}"

    # ---- apply -------------------------------------------------------------

    def _on_apply(self, change):
        cmd = change.get("new", {})
        if not cmd or not self._client.connected:
            return
        vol_idx = cmd.get("volumeIndex")
        if vol_idx is None:
            return

        color_stops = cmd.get("colorStops", [])
        opacity_points = cmd.get("opacityPoints", [])
        value_range = cmd.get("valueRange")
        opacity_val = cmd.get("opacity")
        unit_dist = cmd.get("unitDistance")

        if len(color_stops) < 2 or len(opacity_points) < 2:
            return

        # Build 256 RGBA samples by sampling the colour ramp and opacity
        sorted_colors = sorted(color_stops, key=lambda s: s["t"])
        sorted_opacity = sorted(opacity_points, key=lambda p: p["x"])

        rgba_samples = []
        for i in range(NUM_SAMPLES):
            t = i / (NUM_SAMPLES - 1)
            # Colour interpolation
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

            # Opacity interpolation (piecewise linear)
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
        self._client.set_volume_tf(
            vol_idx,
            rgba_samples,
            op_xy,
            value_range=vr,
            opacity=opacity_val,
            unit_distance=unit_dist,
        )
        logger.debug("Applied TF to volume %s", vol_idx)
        self._status = "Applied"
