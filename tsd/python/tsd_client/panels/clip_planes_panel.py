# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Clip planes panel — volume selector and X/Y/Z range sliders (two knobs per
axis, -1 to +1), plus per-plane normal/distance editing.

Works with any :class:`~tsd_client.client.TSDClient` (or subclass) that has
a scene graph with volumes.

Usage::

    from tsd_client import TSDClient
    from tsd_client.panels import ClipPlanesPanel

    client = TSDClient("host", 12345)
    panel = ClipPlanesPanel(client)
    display(panel)
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import anywidget
import traitlets

from ..utils import resolve_client, run_on_kernel_loop

if TYPE_CHECKING:
    from ..client import TSDClient

logger = logging.getLogger("tsd_client.clip_planes_panel")

# ---------------------------------------------------------------------------
# ESM front-end
# ---------------------------------------------------------------------------

_CLIP_PLANES_ESM = r"""
export function render({ model, el }) {
  const C = {
    bg:       '#1e1e1e',
    border:   '#444',
    section:  '#252525',
    header:   '#2a2a2a',
    text:     '#ccc',
    textDim:  '#888',
    accent:   '#4a9eff',
    inputBg:  '#333',
    btnBg:    '#333',
    btnHover: '#444',
  };

  const root = document.createElement('div');
  root.style.cssText = `
    font-family: 'SF Mono','Fira Code','Consolas', monospace;
    font-size: 12px; color: ${C.text};
    background: ${C.bg};
    border: 1px solid ${C.border};
    border-radius: 6px;
    max-width: 380px;
    overflow: hidden;
    user-select: none;
  `;

  const titleBar = document.createElement('div');
  titleBar.style.cssText = `
    padding: 8px 12px; font-size: 13px; font-weight: 600;
    border-bottom: 1px solid ${C.border};
    display: flex; align-items: center; gap: 6px;
  `;
  titleBar.textContent = '\u2702 Clip Planes';

  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '\u21BB';
  refreshBtn.title = 'Refresh volume list';
  refreshBtn.style.cssText = `
    margin-left: auto; background: ${C.btnBg}; color: ${C.text};
    border: 1px solid ${C.border}; border-radius: 4px;
    padding: 2px 7px; cursor: pointer; font-size: 12px;
  `;
  refreshBtn.addEventListener('mouseenter', () => refreshBtn.style.background = C.btnHover);
  refreshBtn.addEventListener('mouseleave', () => refreshBtn.style.background = C.btnBg);
  titleBar.appendChild(refreshBtn);
  root.appendChild(titleBar);

  const container = document.createElement('div');
  container.style.padding = '4px 12px 8px';
  root.appendChild(container);
  el.appendChild(root);

  function makeAxisRow(axisLabel, rangeMin, rangeMax, step, lowVal, highVal, onChange) {
    const row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 8px; margin: 6px 0; flex-wrap: wrap;';

    const axisLbl = document.createElement('span');
    axisLbl.textContent = axisLabel;
    axisLbl.style.cssText = `min-width: 28px; font-size: 11px; font-weight: 600; color: ${C.text}`;

    function oneSlider(side, value, accent) {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'display: flex; align-items: center; gap: 4px; flex: 1; min-width: 120px;';

      const sideLbl = document.createElement('span');
      sideLbl.textContent = side === 'min' ? '\u2212' : '+';
      sideLbl.style.cssText = `width: 14px; font-size: 11px; color: ${C.textDim}; text-align: center;`;
      const sl = document.createElement('input');
      sl.type = 'range';
      sl.min = rangeMin; sl.max = rangeMax; sl.step = step;
      sl.value = value;
      sl.style.cssText = `flex: 1; min-width: 60px; accent-color: ${accent};`;
      const valLbl = document.createElement('span');
      valLbl.style.cssText = `min-width: 40px; text-align: right; font-size: 11px; font-variant-numeric: tabular-nums; color: ${C.textDim};`;
      valLbl.textContent = Number(value).toFixed(2);
      sl.addEventListener('input', () => { valLbl.textContent = Number(sl.value).toFixed(2); });
      wrap.append(sideLbl, sl, valLbl);
      return { wrap, sl, valLbl };
    }

    const loAccent = '#5a9eff';
    const hiAccent = '#e8a030';
    const loPart = oneSlider('min', lowVal, loAccent);
    const hiPart = oneSlider('max', highVal, hiAccent);

    function sync() {
      let l = parseFloat(loPart.sl.value), h = parseFloat(hiPart.sl.value);
      if (l > h) {
        l = h; loPart.sl.value = l; loPart.valLbl.textContent = l.toFixed(2);
      }
      if (h < l) {
        h = l; hiPart.sl.value = h; hiPart.valLbl.textContent = h.toFixed(2);
      }
      onChange(l, h);
    }

    loPart.sl.addEventListener('input', sync);
    hiPart.sl.addEventListener('input', sync);

    row.append(axisLbl, loPart.wrap, hiPart.wrap);
    return {
      row,
      setRange(low, high) {
        loPart.sl.value = Math.min(low, high);
        hiPart.sl.value = Math.max(low, high);
        loPart.valLbl.textContent = Number(loPart.sl.value).toFixed(2);
        hiPart.valLbl.textContent = Number(hiPart.sl.value).toFixed(2);
      },
    };
  }

  function sendClip(volIndex, clipXMin, clipXMax, clipYMin, clipYMax, clipZMin, clipZMax) {
    model.set('_clip_cmd', {
      volIndex,
      clipXMin, clipXMax, clipYMin, clipYMax, clipZMin, clipZMax,
      _t: Date.now(),
    });
    model.save_changes();
  }

  function sendPlane(volIndex, planeIndex, nx, ny, nz, d) {
    model.set('_clip_cmd', {
      volIndex,
      planeIndex,
      nx: Number(nx), ny: Number(ny), nz: Number(nz), d: Number(d),
      _t: Date.now(),
    });
    model.save_changes();
  }

  const vols = model.get('_clip_volumes') || [];
  let selectedVol = vols[0] || null;
  let selectedPlaneIndex = 0;
  let clipXMin = -1.0, clipXMax = 1.0, clipYMin = -1.0, clipYMax = 1.0, clipZMin = -1.0, clipZMax = 1.0;
  const planeValues = [[-1,0,0,0], [1,0,0,0], [0,-1,0,0], [0,1,0,0]];

  function getVolPlane(vol, i) {
    if (!vol || !vol['clipPlane' + i]) return planeValues[i];
    const p = vol['clipPlane' + i];
    return Array.isArray(p) ? p : (p && p.length === 4 ? [p[0],p[1],p[2],p[3]] : planeValues[i]);
  }

  const volRow = document.createElement('div');
  volRow.style.cssText = 'display: flex; align-items: center; gap: 6px; margin: 8px 0;';

  const volLbl = document.createElement('label');
  volLbl.textContent = 'Volume';
  volLbl.style.cssText = `min-width: 60px; font-size: 11px; color: ${C.text}`;

  const sel = document.createElement('select');
  sel.style.cssText = `
    flex: 1; background: ${C.inputBg}; color: ${C.text};
    border: 1px solid ${C.border}; border-radius: 3px;
    padding: 4px 6px; font-size: 11px; font-family: inherit;
  `;

  function buildVolumeOptions() {
    const v = model.get('_clip_volumes') || [];
    sel.innerHTML = '';
    v.forEach((vol, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = vol.name || ('Volume ' + (vol.index != null ? vol.index : i));
      sel.appendChild(opt);
    });
    selectedVol = v[sel.value !== '' ? parseInt(sel.value, 10) : 0] || null;
    if (selectedVol) {
      for (let i = 0; i < 4; i++) planeValues[i] = getVolPlane(selectedVol, i);
      updatePlaneInputs();
    }
  }

  function doSend() {
    if (selectedVol) sendClip(selectedVol.index, clipXMin, clipXMax, clipYMin, clipYMax, clipZMin, clipZMax);
  }

  sel.addEventListener('change', () => {
    const v = model.get('_clip_volumes') || [];
    selectedVol = v[parseInt(sel.value, 10)] || null;
    if (selectedVol) {
      for (let i = 0; i < 4; i++) planeValues[i] = getVolPlane(selectedVol, i);
      updatePlaneInputs();
    }
    doSend();
  });

  volRow.append(volLbl, sel);
  container.appendChild(volRow);

  const planeRow = document.createElement('div');
  planeRow.style.cssText = 'display: flex; align-items: center; gap: 6px; margin: 8px 0;';
  const planeLbl = document.createElement('label');
  planeLbl.textContent = 'Clip plane';
  planeLbl.style.cssText = `min-width: 60px; font-size: 11px; color: ${C.text}`;
  const planeSel = document.createElement('select');
  planeSel.style.cssText = `
    flex: 1; background: ${C.inputBg}; color: ${C.text};
    border: 1px solid ${C.border}; border-radius: 3px;
    padding: 4px 6px; font-size: 11px; font-family: inherit;
  `;
  ['0 (X\u2212)', '1 (X+)', '2 (Y\u2212)', '3 (Y+)'].forEach((label, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = label;
    planeSel.appendChild(o);
  });
  planeRow.append(planeLbl, planeSel);
  container.appendChild(planeRow);

  const planeInputRow = document.createElement('div');
  planeInputRow.style.cssText = 'display: flex; align-items: center; gap: 6px; margin: 6px 0; flex-wrap: wrap;';
  const planeInputs = [];
  ['nx', 'ny', 'nz', 'd'].forEach((id, i) => {
    const lbl = document.createElement('span');
    lbl.textContent = id;
    lbl.style.cssText = `min-width: 18px; font-size: 10px; color: ${C.textDim};`;
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = 0.01;
    inp.style.cssText = `width: 56px; background: ${C.inputBg}; color: ${C.text}; border: 1px solid ${C.border}; border-radius: 3px; padding: 2px 4px; font-size: 11px;`;
    planeInputRow.append(lbl, inp);
    planeInputs.push(inp);
  });

  function updatePlaneInputs() {
    const p = planeValues[selectedPlaneIndex] || planeValues[0];
    planeInputs.forEach((inp, i) => { inp.value = (p[i] != null ? p[i] : 0); });
  }

  function onPlaneInputChange() {
    const p = planeValues[selectedPlaneIndex];
    p[0] = parseFloat(planeInputs[0].value) || 0;
    p[1] = parseFloat(planeInputs[1].value) || 0;
    p[2] = parseFloat(planeInputs[2].value) || 0;
    p[3] = parseFloat(planeInputs[3].value) || 0;
    if (selectedVol) sendPlane(selectedVol.index, selectedPlaneIndex, p[0], p[1], p[2], p[3]);
  }

  planeInputs.forEach(inp => inp.addEventListener('change', onPlaneInputChange));
  planeInputs.forEach(inp => inp.addEventListener('input', onPlaneInputChange));

  planeSel.addEventListener('change', () => {
    selectedPlaneIndex = parseInt(planeSel.value, 10);
    updatePlaneInputs();
  });

  container.appendChild(planeInputRow);

  const xRow = makeAxisRow('X', -1, 1, 0.01, -1.0, 1.0, (lo, hi) => {
    clipXMin = lo; clipXMax = hi;
    doSend();
  });
  const yRow = makeAxisRow('Y', -1, 1, 0.01, -1.0, 1.0, (lo, hi) => {
    clipYMin = lo; clipYMax = hi;
    doSend();
  });
  const zRow = makeAxisRow('Z', -1, 1, 0.01, -1.0, 1.0, (lo, hi) => {
    clipZMin = lo; clipZMax = hi;
    doSend();
  });

  container.appendChild(xRow.row);
  container.appendChild(yRow.row);
  container.appendChild(zRow.row);

  const hint = document.createElement('div');
  hint.style.cssText = `font-size: 10px; color: ${C.textDim}; padding: 6px 0; margin-top: 4px;`;
  hint.textContent = 'Select a clip plane to edit (nx, ny, nz, d). Or use X/Y/Z sliders for axis-aligned clipping.';
  container.appendChild(hint);

  model.on('change:_clip_volumes', buildVolumeOptions);
  buildVolumeOptions();

  model.on('change:_clip_cmd', () => {});

  refreshBtn.addEventListener('click', () => {
    model.set('_clip_refresh', Date.now());
    model.save_changes();
  });
}
"""


# ---------------------------------------------------------------------------
# Python widget
# ---------------------------------------------------------------------------


class ClipPlanesPanel(anywidget.AnyWidget):
    """Panel to select a volume and set axis-aligned clip ranges (X, Y, Z).

    Each axis has a range slider with two knobs (-1 to +1). Volume is clipped
    outside the selected range. Four planes are sent: Xmin, Xmax, Ymin, Ymax
    (Z range is shown in UI but only 4 planes exist on the server).
    """

    _esm = traitlets.Unicode(_CLIP_PLANES_ESM).tag(sync=True)
    _clip_volumes = traitlets.List([]).tag(sync=True)
    _clip_cmd = traitlets.Dict({}).tag(sync=True)
    _clip_refresh = traitlets.Float(0.0).tag(sync=True)

    def __init__(
        self,
        client,
        auto_refresh: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = resolve_client(client)
        self.observe(self._handle_clip_cmd, names=["_clip_cmd"])
        self.observe(self._handle_refresh, names=["_clip_refresh"])
        if auto_refresh:
            self.refresh()

    def refresh(self):
        """Rebuild volume list with clip plane state (background thread)."""
        if not self._client.connected:
            self._clip_volumes = []
            return

        def _do():
            try:
                sg = self._client.scene_graph
                if sg is None:
                    options: list[dict] = []
                else:
                    options = []
                    for v in sg.volumes:
                        idx = v.object_index
                        info = None
                        request_fn = getattr(self._client, "request_volume_info", None)
                        if request_fn is not None:
                            try:
                                info = request_fn(idx, timeout=3.0)
                            except Exception:
                                pass
                        if not info or not isinstance(info, dict):
                            info = sg.volume_info(idx) or {}
                        field = info.get("field") or {}
                        entry = {
                            "index": idx,
                            "name": info.get("name") or v.name or f"Volume {idx}",
                        }
                        for i in range(4):
                            key = f"clipPlane{i}"
                            val = field.get(key) or info.get(key)
                            if val is not None:
                                if isinstance(val, (list, tuple)) and len(val) >= 4:
                                    entry[key] = [
                                        float(val[0]),
                                        float(val[1]),
                                        float(val[2]),
                                        float(val[3]),
                                    ]
                                else:
                                    entry[key] = val
                        options.append(entry)

                def _apply():
                    self._clip_volumes = options

                if not run_on_kernel_loop(_apply):
                    _apply()
            except Exception:
                logger.exception("Failed to fetch volume list for clip panel")

                def _clear():
                    self._clip_volumes = []

                if not run_on_kernel_loop(_clear):
                    _clear()

        threading.Thread(target=_do, daemon=True, name="tsd-clip-refresh").start()

    def _handle_refresh(self, change):
        if not change.get("new"):
            return
        self.refresh()

    def _handle_clip_cmd(self, change):
        cmd = change.get("new", {})
        if not cmd or "volIndex" not in cmd:
            return
        sg = self._client.cached_scene_graph() or self._client.scene_graph
        if sg is None:
            return
        vol_index = int(cmd["volIndex"])
        try:
            if "planeIndex" in cmd and all(
                k in cmd for k in ("nx", "ny", "nz", "d")
            ):
                plane_index = int(cmd["planeIndex"])
                if 0 <= plane_index <= 3:
                    vec = (
                        float(cmd["nx"]),
                        float(cmd["ny"]),
                        float(cmd["nz"]),
                        float(cmd["d"]),
                    )
                    sg.set_object_parameter(
                        sg.ANARI_VOLUME,
                        vol_index,
                        f"clipPlane{plane_index}",
                        vec,
                        param_type="float32_vec4",
                    )
                    logger.debug(
                        "Set clipPlane%d for vol %s to %s",
                        plane_index,
                        vol_index,
                        vec,
                    )
                return
            x_min = float(cmd.get("clipXMin", -1.0))
            x_max = float(cmd.get("clipXMax", 1.0))
            y_min = float(cmd.get("clipYMin", -1.0))
            y_max = float(cmd.get("clipYMax", 1.0))
            sg.set_object_parameter(
                sg.ANARI_VOLUME,
                vol_index,
                "clipPlane0",
                (-1.0, 0.0, 0.0, x_min),
                param_type="float32_vec4",
            )
            sg.set_object_parameter(
                sg.ANARI_VOLUME,
                vol_index,
                "clipPlane1",
                (1.0, 0.0, 0.0, -x_max),
                param_type="float32_vec4",
            )
            sg.set_object_parameter(
                sg.ANARI_VOLUME,
                vol_index,
                "clipPlane2",
                (0.0, -1.0, 0.0, y_min),
                param_type="float32_vec4",
            )
            sg.set_object_parameter(
                sg.ANARI_VOLUME,
                vol_index,
                "clipPlane3",
                (0.0, 1.0, 0.0, -y_max),
                param_type="float32_vec4",
            )
            logger.debug("Set axis-aligned clip planes for vol %s", vol_index)
        except Exception:
            logger.exception(
                "Failed to set clip planes for volume %s", vol_index
            )
