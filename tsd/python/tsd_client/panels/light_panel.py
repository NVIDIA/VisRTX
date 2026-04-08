# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Light panel — lists all lights in the scene with editable parameters.

Lights are discovered from the scene graph object pool first. When
``cleanupScene()`` on the server strips lights from the objectDB, the
panel falls back to scanning the layer tree for ANARI_LIGHT references
and sends raw parameter change messages.

Usage::

    from tsd_client import TSDClient
    from tsd_client.panels import LightPanel

    client = TSDClient("host", 12345)
    panel = LightPanel(client)
    display(panel)
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import TYPE_CHECKING

import anywidget
import traitlets

from ..anari_types import ANARI_LIGHT, anari_type_name
from ..utils import resolve_client, run_on_kernel_loop, datanode_to_attr as _datanode_to_attr_base
from ..scene import build_parameter_change_payload

if TYPE_CHECKING:
    from ..client import TSDClient

logger = logging.getLogger("tsd_client.light_panel")

# ---------------------------------------------------------------------------
# ESM front-end
# ---------------------------------------------------------------------------

_LIGHT_PANEL_ESM = r"""
export function render({ model, el }) {
  const C = {
    bg:     '#1e1e1e',
    border: '#444',
    header: '#2a2a2a',
    text:   '#ccc',
    dim:    '#888',
    accent: '#4a9eff',
    input:  '#333',
    hover:  '#2f2f2f',
    light:  '#f0c040',
    meta:   '#6a9955',
  };

  const root = document.createElement('div');
  root.style.cssText = `
    font-family: 'SF Mono','Fira Code','Consolas', monospace;
    font-size: 12px; color: ${C.text}; background: ${C.bg};
    border: 1px solid ${C.border}; border-radius: 6px;
    overflow: hidden; max-height: 600px; overflow-y: auto;
  `;

  const titleBar = document.createElement('div');
  titleBar.style.cssText = `
    display: flex; align-items: center; padding: 8px 12px;
    background: ${C.header}; border-bottom: 1px solid ${C.border};
  `;
  const titleEl = document.createElement('span');
  titleEl.style.cssText = 'flex: 1; font-weight: 600; font-size: 13px;';
  titleEl.textContent = 'Lights';
  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '\u21BB';
  refreshBtn.title = 'Refresh';
  refreshBtn.style.cssText = `
    background: none; border: 1px solid ${C.border}; color: ${C.dim};
    border-radius: 3px; padding: 2px 6px; cursor: pointer; font-size: 14px;
  `;
  refreshBtn.addEventListener('click', () => {
    model.set('_refresh_cmd', { _t: Date.now() });
    model.save_changes();
  });
  titleBar.append(titleEl, refreshBtn);

  const container = document.createElement('div');
  container.style.padding = '2px 0';
  root.append(titleBar, container);
  el.appendChild(root);

  const openLights = {};

  const inpCss = `background:${C.input};color:${C.text};border:1px solid ${C.border};` +
    'border-radius:3px;padding:2px 4px;font-size:11px;font-family:inherit;';

  function send(light, name, type, value) {
    model.set('_param_cmd', {
      lightIndex: light.index,
      inObjectDB: light.inObjectDB,
      name, type, value,
      _t: Date.now(),
    });
    model.save_changes();
    titleEl.textContent = `Lights \u2014 sent ${name}`;
    clearTimeout(send._tid);
    send._tid = setTimeout(() => titleEl.textContent = 'Lights', 1200);
  }

  function buildUI() {
    const lights = model.get('_lights_data') || [];
    container.innerHTML = '';
    if (!lights.length) {
      const msg = document.createElement('div');
      msg.style.cssText = `padding: 16px; text-align: center; color: ${C.dim};`;
      msg.textContent = 'No lights found';
      container.appendChild(msg);
      return;
    }
    for (const light of lights) {
      renderLight(light);
    }
  }

  function renderLight(light) {
    const key = `light_${light.index}`;
    const attrs = light.attrs || [];
    const isOpen = openLights[key] || false;

    const hdr = document.createElement('div');
    hdr.style.cssText = `
      display: flex; align-items: center; gap: 6px;
      padding: 6px 12px; cursor: pointer; user-select: none;
    `;
    hdr.addEventListener('mouseenter', () => hdr.style.background = C.hover);
    hdr.addEventListener('mouseleave', () => hdr.style.background = '');

    const arrow = document.createElement('span');
    arrow.style.cssText = `font-size: 10px; width: 12px; flex-shrink: 0; color: ${C.dim};`;
    arrow.textContent = isOpen ? '\u25BC' : '\u25B6';

    const icon = document.createElement('span');
    icon.textContent = '\u2600';
    icon.style.flexShrink = '0';

    const nameEl = document.createElement('span');
    nameEl.textContent = light.name || `Light ${light.index}`;
    nameEl.style.cssText = 'flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';

    const badge = document.createElement('span');
    badge.textContent = (light.subtype || 'LIGHT').toUpperCase();
    badge.style.cssText = `
      font-size: 9px; padding: 1px 5px; border-radius: 3px;
      background: ${C.input}; color: ${C.light}; flex-shrink: 0;
    `;

    hdr.append(arrow, icon, nameEl, badge);
    container.appendChild(hdr);

    const content = document.createElement('div');
    content.style.display = isOpen ? 'block' : 'none';
    hdr.addEventListener('click', () => {
      const nowOpen = content.style.display === 'none';
      content.style.display = nowOpen ? 'block' : 'none';
      arrow.textContent = nowOpen ? '\u25BC' : '\u25B6';
      openLights[key] = nowOpen;
    });

    for (const attr of attrs) {
      renderAttr(attr, content, light);
    }
    container.appendChild(content);
  }

  function renderAttr(attr, parent, light) {
    const row = document.createElement('div');
    row.style.cssText = `
      display: flex; align-items: center; gap: 6px;
      padding: 2px 8px 2px 44px; font-size: 11px;
    `;

    const label = document.createElement('span');
    label.textContent = attr.name;
    label.title = attr.description || (attr.name + ' (' + attr.type + ')');
    label.style.cssText = `
      min-width: 120px; max-width: 160px; color: ${C.dim}; flex-shrink: 0;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    `;
    if (attr.category === 'metadata') {
      label.style.fontStyle = 'italic';
      label.style.color = C.meta;
    }
    row.appendChild(label);

    const ro = attr.readOnly
      || ['object_ref','array','binary','unknown'].includes(attr.type);
    if (ro) {
      const val = document.createElement('span');
      val.style.cssText = `font-size: 10px; color: ${C.dim};
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;`;
      val.textContent = fmtRO(attr);
      row.appendChild(val);
    } else {
      row.appendChild(mkEditor(attr, light));
    }
    parent.appendChild(row);
  }

  function fmtRO(a) {
    const v = a.value;
    if (a.type === 'object_ref') return '\u2192 ' + v;
    if (Array.isArray(v))
      return '[' + v.map(x => typeof x === 'number' ? x.toPrecision(4) : x).join(', ') + ']';
    if (typeof v === 'number') return v.toPrecision(6);
    return String(v != null ? v : '');
  }

  function mkEditor(attr, light) {
    const { name, type, value } = attr;

    if (type === 'bool') {
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = !!value;
      cb.style.accentColor = C.accent;
      cb.addEventListener('change', () => send(light, name, 'bool', cb.checked));
      return cb;
    }

    if (type === 'float32' && attr.min != null && attr.max != null) {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'display:flex;align-items:center;gap:4px;flex:1;';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = attr.min; slider.max = attr.max;
      slider.step = attr.step || 0.001;
      slider.value = value != null ? value : 0;
      slider.style.cssText = 'flex:1;accent-color:' + C.accent;
      const lbl = document.createElement('span');
      lbl.style.cssText = `min-width:50px;text-align:right;font-size:10px;color:${C.dim};font-variant-numeric:tabular-nums;`;
      lbl.textContent = Number(slider.value).toPrecision(4);
      slider.addEventListener('input', () => lbl.textContent = Number(slider.value).toPrecision(4));
      slider.addEventListener('change', () => send(light, name, 'float32', parseFloat(slider.value)));
      wrap.append(slider, lbl);
      return wrap;
    }

    if (type === 'float32' || type === 'int32') {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = type === 'int32' ? 1 : 0.01;
      inp.value = value != null ? value : 0;
      inp.style.cssText = 'width:80px;' + inpCss;
      inp.addEventListener('change', () => {
        send(light, name, type,
          type === 'int32' ? parseInt(inp.value, 10) : parseFloat(inp.value));
      });
      return inp;
    }

    if (type === 'direction_azel') {
      const arr = Array.isArray(value) ? value : [0, 0];
      const labs = ['Azimuth', 'Elevation'];
      const wrap = document.createElement('div');
      wrap.style.cssText = 'display:flex;flex-direction:column;gap:4px;flex:1;';
      const sliders = [];
      const doSend = () => send(light, name, 'direction_azel',
        sliders.map(s => parseFloat(s.value) || 0));
      for (let i = 0; i < 2; i++) {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:4px;';
        const l = document.createElement('span');
        l.textContent = labs[i];
        l.style.cssText = `font-size:9px;color:${C.dim};min-width:48px;`;
        const sl = document.createElement('input');
        sl.type = 'range'; sl.min = -180; sl.max = 180; sl.step = 1;
        sl.value = arr[i] != null ? arr[i] : 0;
        sl.style.cssText = 'flex:1;accent-color:' + C.accent;
        const lbl = document.createElement('span');
        lbl.style.cssText = `min-width:36px;text-align:right;font-size:10px;color:${C.dim};font-variant-numeric:tabular-nums;`;
        lbl.textContent = Math.round(sl.value) + '\u00B0';
        sl.addEventListener('input', () => lbl.textContent = Math.round(sl.value) + '\u00B0');
        sl.addEventListener('change', doSend);
        sliders.push(sl);
        row.append(l, sl, lbl);
        wrap.appendChild(row);
      }
      return wrap;
    }

    if (type.startsWith('float32_vec')) {
      const n = parseInt(type.slice(-1));
      const arr = Array.isArray(value) ? value : Array(n).fill(0);
      const labs = ['x','y','z','w'];
      const inputs = [];
      const flex = document.createElement('div');
      flex.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;';
      const doSend = () => send(light, name, type, inputs.map(i => parseFloat(i.value) || 0));
      for (let i = 0; i < n; i++) {
        const w = document.createElement('div');
        w.style.cssText = 'display:flex;align-items:center;gap:2px;';
        const l = document.createElement('span');
        l.textContent = labs[i];
        l.style.cssText = 'font-size:9px;color:' + C.dim + ';';
        const inp = document.createElement('input');
        inp.type = 'number'; inp.step = 0.01;
        inp.value = arr[i] != null ? arr[i] : 0;
        inp.style.cssText = 'width:54px;font-size:10px;' + inpCss;
        inp.addEventListener('change', doSend);
        inputs.push(inp);
        w.append(l, inp);
        flex.appendChild(w);
      }
      return flex;
    }

    if (type === 'string' && attr.stringValues && attr.stringValues.length) {
      const sel = document.createElement('select');
      sel.style.cssText = 'flex:1;' + inpCss;
      for (const opt of attr.stringValues) {
        const o = document.createElement('option');
        o.value = opt; o.textContent = opt;
        if (opt === value) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => send(light, name, 'string', sel.value));
      return sel;
    }

    const inp = document.createElement('input');
    inp.type = 'text';
    inp.value = value != null ? String(value) : '';
    inp.style.cssText = 'flex:1;' + inpCss;
    inp.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      let v = inp.value;
      if (type === 'float32') v = parseFloat(v);
      else if (type === 'int32') v = parseInt(v, 10);
      send(light, name, type || 'string', v);
    });
    return inp;
  }

  model.on('change:_lights_data', buildUI);
  buildUI();
}
"""


# ---------------------------------------------------------------------------
# Python widget
# ---------------------------------------------------------------------------

class LightPanel(anywidget.AnyWidget):
    """Jupyter widget listing all lights in the scene with editable parameters."""

    _esm = traitlets.Unicode(_LIGHT_PANEL_ESM).tag(sync=True)
    _lights_data = traitlets.List(traitlets.Dict()).tag(sync=True)
    _param_cmd = traitlets.Dict().tag(sync=True)
    _refresh_cmd = traitlets.Dict().tag(sync=True)

    def __init__(self, client, *, auto_refresh: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._client: TSDClient = resolve_client(client)
        self._light_ensured: set[int] = set()
        self.observe(self._handle_param_cmd, names=["_param_cmd"])
        self.observe(self._handle_refresh_cmd, names=["_refresh_cmd"])
        if auto_refresh:
            self.refresh()

    # -- Refresh -------------------------------------------------------------

    def refresh(self) -> None:
        """Collect lights from the scene graph and update the widget."""
        client = self._client
        if not client.connected:
            run_on_kernel_loop(lambda: setattr(self, "_lights_data", []))
            return

        def _do():
            try:
                sg = client.scene_graph
                if sg is None:
                    lights_json: list[dict] = []
                else:
                    lights_json = self._collect_lights(sg)
                    for lj in lights_json:
                        idx = lj.get("index", 0)
                        if idx not in self._light_ensured:
                            self._ensure_light_exists(client, idx)
            except Exception:
                logger.exception("Failed to collect lights")
                lights_json = []
            run_on_kernel_loop(lambda lj=lights_json: setattr(self, "_lights_data", lj))

        threading.Thread(target=_do, daemon=True, name="tsd-light-refresh").start()

    # -- Light collection ----------------------------------------------------

    def _collect_lights(self, sg) -> list[dict]:
        """Build the lights list with 3-tier fallback:
        1. ObjectDB pool  2. Layer tree scan  3. Default Light 0
        """
        db_lights = sg.lights
        if db_lights:
            logger.debug("Found %d light(s) in objectDB", len(db_lights))
            return [self._light_from_object(obj) for obj in db_lights]

        tree_lights = self._scan_layer_tree_for_lights(sg)
        if tree_lights:
            logger.debug("Found %d light(s) in layer tree", len(tree_lights))
            return tree_lights

        logger.debug("No lights in objectDB or layer tree; using default Light 0")
        return [self._default_light()]

    @staticmethod
    def _default_light() -> dict:
        """Fallback entry for when the server strips lights from the scene
        data.  Provides the standard directional light parameters so the
        user can still control them via raw parameter messages."""
        return {
            "index": 0,
            "name": "Light 0",
            "subtype": "directional",
            "inObjectDB": False,
            "attrs": [
                {"name": "direction", "type": "direction_azel",
                 "value": [0.0, -60.0], "readOnly": False,
                 "description": "Light direction (azimuth / elevation in degrees)"},
                {"name": "color", "type": "float32_vec3",
                 "value": [1.0, 0.98, 0.95], "readOnly": False,
                 "description": "Light color (RGB)"},
                {"name": "irradiance", "type": "float32",
                 "value": 30.0, "readOnly": False, "min": 0.0, "max": 200.0,
                 "description": "Light irradiance"},
            ],
        }

    def _light_from_object(self, obj) -> dict:
        """Build a light dict from an ObjectInfo in the objectDB."""
        attrs = self._object_attrs(obj)
        return {
            "index": obj.object_index,
            "name": obj.name or f"Light {obj.object_index}",
            "subtype": obj.subtype if obj.subtype and obj.subtype != "none" else "",
            "inObjectDB": True,
            "attrs": attrs,
        }

    def _scan_layer_tree_for_lights(self, sg) -> list[dict]:
        """Fallback: find lights by scanning layer tree nodes."""
        found: list[dict] = []
        seen: set[int] = set()
        for lname in sg.layer_names:
            root = sg.layer_tree(lname)
            if root is None:
                continue
            self._walk_for_lights(root, sg, found, seen)
        return found

    def _walk_for_lights(self, node, sg, found: list[dict], seen: set[int]):
        if (node.object_type == ANARI_LIGHT
                and node.object_index is not None
                and node.object_index not in seen):
            seen.add(node.object_index)
            obj = sg.get_object(ANARI_LIGHT, node.object_index)
            if obj is not None:
                found.append(self._light_from_object(obj))
            else:
                fallback = self._default_light()
                fallback["index"] = node.object_index
                fallback["name"] = node.name or f"Light {node.object_index}"
                found.append(fallback)
        for child in node.children:
            self._walk_for_lights(child, sg, found, seen)

    # -- Attribute extraction (mirrors DataTreePanel logic) ------------------

    @staticmethod
    def _object_attrs(obj) -> list[dict]:
        attrs: list[dict] = []
        for name, node in obj.parameters.items():
            a = LightPanel._datanode_to_attr(name, node, read_only=False)
            if a is not None:
                if name == "direction" and a["type"] == "float32_vec2":
                    a["type"] = "direction_azel"
                    a["description"] = "Light direction (azimuth / elevation in degrees)"
                attrs.append(a)
        for name, node in obj.metadata.items():
            a = LightPanel._datanode_to_attr(name, node, read_only=True)
            if a is not None:
                a["category"] = "metadata"
                attrs.append(a)
        return attrs

    @staticmethod
    def _datanode_to_attr(name: str, node, read_only: bool) -> dict | None:
        """Delegate to the shared :func:`~tsd_client.utils.datanode_to_attr`."""
        return _datanode_to_attr_base(name, node, read_only)

    # -- Command handlers ----------------------------------------------------

    @staticmethod
    def _azel_to_dir(az_deg: float, el_deg: float) -> tuple[float, float, float]:
        """Convert azimuth/elevation (degrees) to a unit direction vector.
        Matches ``tsd::math::azelToDir`` in TSDMath.hpp."""
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        x = math.sin(az) * math.cos(el)
        y = math.sin(el)
        z = math.cos(az) * math.cos(el)
        return (x, y, z)

    def _handle_param_cmd(self, change):
        cmd = change.get("new", {})
        if not cmd:
            return
        light_idx = cmd.get("lightIndex")
        name = cmd.get("name")
        ptype = cmd.get("type")
        value = cmd.get("value")
        if light_idx is None or not name or not ptype:
            return

        if ptype == "direction_azel" and isinstance(value, list) and len(value) == 2:
            az, el = float(value[0]), float(value[1])
            value = self._azel_to_dir(az, el)
            logger.info("direction azel (%.1f, %.1f) → vec3 %s", az, el, value)
            ptype = "float32_vec3"
            name = "direction"

        if ptype in ("float32_vec2", "float32_vec3", "float32_vec4") and isinstance(
            value, list
        ):
            value = tuple(float(x) for x in value)

        self._send_raw(self._client, int(light_idx), name, ptype, value)

    def _ensure_light_exists(self, client, light_idx: int):
        """Re-create the light in the server's TSD object pool.

        ``cleanupScene()`` on the server can remove light objects from the
        pool even while the ANARI device still renders them.  Sending a
        ``SERVER_ADD_OBJECT`` message restores the object so that subsequent
        ``ParameterChange`` messages can find it.

        A ``SERVER_UPDATE_LAYER`` is also sent for every layer to trigger
        ``signalLayerStructureChanged`` on the server, which forces the
        ANARI world rebuild to pick up the restored light object.

        Only sent once per (panel, index) pair.
        """
        if light_idx in self._light_ensured:
            return
        from ..datatree import DataTree
        from ..protocol import MessageType

        sg = client.cached_scene_graph()
        wire_size_t = sg._wire_size_t if sg is not None else 8

        tree = DataTree()
        tree._size_t_bytes = wire_size_t
        tree.root["self"].set_object(ANARI_LIGHT, light_idx)
        tree.root["subtype"].set_string("directional")
        tree.root["name"].set_string("Sun")

        params = tree.root["parameters"]
        d = params["direction"]
        d["value"].set_vec2((0.0, 300.0))
        d["enabled"].set_bool(True)
        d["usage"].set_int(2)  # ParameterUsageHint::DIRECTION
        d["min"].set_vec2((0.0, 0.0))
        d["max"].set_vec2((360.0, 360.0))
        c = params["color"]
        c["value"].set_vec3((1.0, 0.98, 0.95))
        c["enabled"].set_bool(True)
        c["usage"].set_int(1)  # ParameterUsageHint::COLOR
        irr = params["irradiance"]
        irr["value"].set_float(30.0)
        irr["enabled"].set_bool(True)

        client.send(MessageType.SERVER_ADD_OBJECT, tree.to_bytes())
        logger.info("Sent SERVER_ADD_OBJECT to recreate light[%d]", light_idx)

        if sg is not None:
            for lname in sg.layer_names:
                client.send_layer_update(sg, lname)
            logger.info("Sent layer updates for %s", sg.layer_names)

        self._light_ensured.add(light_idx)
        logger.info("Light[%d] ensured in server pool", light_idx)

    def _send_raw(self, client, light_idx: int, name: str, ptype: str, value):
        """Send a parameter change directly, bypassing the scene graph."""
        from ..protocol import MessageType

        try:
            wire_size_t = 8
            sg = client.cached_scene_graph()
            if sg is not None:
                wire_size_t = sg._wire_size_t
            payload = build_parameter_change_payload(
                ANARI_LIGHT, light_idx, [(name, ptype, value)],
                wire_size_t=wire_size_t,
            )
            client.send(MessageType.SERVER_SET_OBJECT_PARAMETER, payload)
            logger.info(
                "Sent light[%d].%s (%s) = %s  [%d bytes, size_t=%d]",
                light_idx, name, ptype, value, len(payload), wire_size_t,
            )
        except Exception:
            logger.exception("Failed to send raw light param %d.%s", light_idx, name)

    def _handle_refresh_cmd(self, change):
        self.refresh()
