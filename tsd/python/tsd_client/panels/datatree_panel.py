# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
DataTree panel — hierarchical tree view of the full TSD scene structure.

Displays the complete layer → transform → object hierarchy from the
server's scene graph, exposing every attribute (parameters, metadata,
object references, array summaries) of each scene object.

The panel is fully generic: the title, parameter slider hints, and
read-only parameter set are all configurable.

Usage::

    from tsd_client import TSDClient
    from tsd_client.panels import DataTreePanel

    client = TSDClient("host", 12345)
    panel = DataTreePanel(client, title="My Scene")
    display(panel)
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import anywidget
import traitlets

from ..utils import resolve_client, guess_type as _guess_type, run_on_kernel_loop, datanode_to_attr as _datanode_to_attr_base

if TYPE_CHECKING:
    from ..client import TSDClient

logger = logging.getLogger("tsd_client.datatree_panel")

# ---------------------------------------------------------------------------
# ESM front-end — recursive tree view
# ---------------------------------------------------------------------------

_DATATREE_PANEL_ESM = r"""
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
    xform:  '#e8a84c',
    obj:    '#66b3ff',
    layer:  '#76c893',
    grp:    '#999',
    ref:    '#b894e8',
    meta:   '#6a9955',
  };

  const ICONS = {
    layer:        '\uD83D\uDCC2',
    transform:    '\u21BB',
    group:        '\uD83D\uDCC1',
    geometry:     '\u25B3',
    material:     '\u25C9',
    surface:      '\u25A0',
    volume:       '\u2B22',
    spatialField: '\u2261',
    light:        '\u2600',
    camera:       '\uD83C\uDFA5',
    sampler:      '\u25A8',
    renderer:     '\u25CE',
    array:        '\u2395',
    array1d:      '\u2395',
    array2d:      '\u2395',
    array3d:      '\u2395',
    _default:     '\u25CF',
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
  titleEl.textContent = model.get('_panel_title') || 'Scene Tree';
  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '\u21BB';
  refreshBtn.title = 'Refresh';
  refreshBtn.style.cssText = `
    background: none; border: 1px solid ${C.border}; color: ${C.dim};
    border-radius: 3px; padding: 2px 6px; cursor: pointer; font-size: 14px;
  `;
  refreshBtn.addEventListener('click', () => {
    model.set('_dt_refresh_cmd', { _t: Date.now() });
    model.save_changes();
  });
  titleBar.append(titleEl, refreshBtn);

  const container = document.createElement('div');
  container.style.padding = '2px 0';
  root.append(titleBar, container);
  el.appendChild(root);

  const openPaths = {};

  function buildTree() {
    const layers = model.get('_dt_layers') || [];
    container.innerHTML = '';
    if (!layers.length) {
      const msg = document.createElement('div');
      msg.style.cssText = `padding: 16px; text-align: center; color: ${C.dim};`;
      msg.textContent = model.get('_empty_message') || 'No objects loaded';
      container.appendChild(msg);
      return;
    }
    for (const layer of layers) {
      renderNode(layer, container, 0, layer.name);
    }
  }

  function badgeColor(node) {
    if (node.objectTypeName) return C.obj;
    const nt = node.nodeType;
    if (nt === 'transform') return C.xform;
    if (nt === 'layer') return C.layer;
    return C.grp;
  }

  function renderNode(node, parent, depth, path) {
    const children = node.children || [];
    const attrs = node.attrs || [];
    const expandable = children.length > 0 || attrs.length > 0;
    const isOpen = openPaths[path] || false;

    const hdr = document.createElement('div');
    hdr.style.cssText = `
      display: flex; align-items: center; gap: 4px;
      padding: 3px 8px 3px ${8 + depth * 16}px;
      user-select: none;
    `;
    if (expandable) {
      hdr.style.cursor = 'pointer';
      hdr.addEventListener('mouseenter', () => hdr.style.background = C.hover);
      hdr.addEventListener('mouseleave', () => hdr.style.background = '');
    }
    if (node.enabled === false) hdr.style.opacity = '0.4';

    const arrow = document.createElement('span');
    arrow.style.cssText = `font-size: 10px; width: 12px; flex-shrink: 0; color: ${C.dim};`;
    arrow.textContent = expandable ? (isOpen ? '\u25BC' : '\u25B6') : ' ';

    const typeKey = node.objectTypeName || node.nodeType || '_default';
    const iconEl = document.createElement('span');
    iconEl.textContent = ICONS[typeKey] || ICONS._default;
    iconEl.style.flexShrink = '0';

    const nameEl = document.createElement('span');
    nameEl.textContent = node.name || '(unnamed)';
    nameEl.style.cssText = 'flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';

    const badgeText = node.subtype
      ? node.subtype.toUpperCase()
      : (node.objectTypeName || node.nodeType || '').toUpperCase();
    const badge = document.createElement('span');
    badge.textContent = badgeText;
    badge.style.cssText = `
      font-size: 9px; padding: 1px 5px; border-radius: 3px;
      background: ${C.input}; color: ${badgeColor(node)}; flex-shrink: 0;
    `;

    hdr.append(arrow, iconEl, nameEl, badge);
    parent.appendChild(hdr);

    const content = document.createElement('div');
    content.style.display = isOpen ? 'block' : 'none';
    if (expandable) {
      hdr.addEventListener('click', () => {
        const nowOpen = content.style.display === 'none';
        content.style.display = nowOpen ? 'block' : 'none';
        arrow.textContent = nowOpen ? '\u25BC' : '\u25B6';
        openPaths[path] = nowOpen;
      });
    }

    for (const attr of attrs) renderAttr(attr, content, depth + 1, node);
    for (let i = 0; i < children.length; i++) {
      const c = children[i];
      renderNode(c, content, depth + 1, path + '/' + (c.name || i));
    }

    parent.appendChild(content);
  }

  function renderAttr(attr, parent, depth, node) {
    const row = document.createElement('div');
    row.style.cssText = `
      display: flex; align-items: center; gap: 6px;
      padding: 2px 8px 2px ${8 + depth * 16 + 16}px;
      font-size: 11px;
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
      val.style.cssText = `font-size: 10px; color: ${
        attr.type === 'object_ref' ? C.ref : C.dim
      }; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;`;
      val.textContent = fmtRO(attr);
      row.appendChild(val);
    } else {
      row.appendChild(mkEditor(attr, node));
    }

    parent.appendChild(row);
  }

  function fmtRO(a) {
    if (a.type === 'object_ref') return '\u2192 ' + a.value;
    const v = a.value;
    if (Array.isArray(v))
      return '[' + v.map(x => typeof x === 'number' ? x.toPrecision(4) : x).join(', ') + ']';
    if (typeof v === 'number') return v.toPrecision(6);
    return String(v != null ? v : '');
  }

  function send(node, name, type, value, attr) {
    const cmd = {
      objectType: node.objectType,
      objectIndex: node.objectIndex,
      layerName: node.layerName,
      nodeName: node.nodeName,
      name, type, value,
      _t: Date.now(),
    };
    if (attr && attr.isSRT) cmd.isSRT = true;
    model.set('_dt_param_cmd', cmd);
    model.save_changes();
  }

  const inpCss = `background:${C.input};color:${C.text};border:1px solid ${C.border};` +
    'border-radius:3px;padding:2px 4px;font-size:11px;font-family:inherit;';

  function mkEditor(attr, node) {
    const { name, type, value } = attr;

    if (type === 'bool') {
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = !!value;
      cb.style.accentColor = C.accent;
      cb.addEventListener('change', () => send(node, name, 'bool', cb.checked, attr));
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
      slider.addEventListener('change', () => send(node, name, 'float32', parseFloat(slider.value), attr));
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
        send(node, name, type,
          type === 'int32' ? parseInt(inp.value, 10) : parseFloat(inp.value), attr);
      });
      return inp;
    }

    if (type.startsWith('float32_vec')) {
      const n = parseInt(type.slice(-1));
      const arr = Array.isArray(value) ? value : Array(n).fill(0);
      const labs = ['x','y','z','w'];
      const inputs = [];
      const flex = document.createElement('div');
      flex.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;';
      const doSend = () => send(node, name, type, inputs.map(i => parseFloat(i.value) || 0), attr);
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
      sel.addEventListener('change', () => send(node, name, 'string', sel.value, attr));
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
      send(node, name, type || 'string', v, attr);
    });
    return inp;
  }

  model.on('change:_dt_layers', buildTree);
  model.on('change:_panel_title', () => {
    titleEl.textContent = model.get('_panel_title') || 'Scene Tree';
  });
  buildTree();
}
"""


# ---------------------------------------------------------------------------
# Python widget
# ---------------------------------------------------------------------------


class DataTreePanel(anywidget.AnyWidget):
    """Hierarchical tree panel showing the full scene structure.

    Displays layers, transforms, and objects from the client's cached
    :class:`~tsd_client.scene.SceneGraph`, exposing every attribute
    (parameters, metadata, object references, array summaries) with the
    scene hierarchy preserved.

    Parameters
    ----------
    client : TSDClient
        Connected client instance.
    title : str
        Panel title displayed in the header.
    param_hints : dict[str, tuple[float, float, float]]
        Mapping from parameter name to ``(min, max, step)`` for slider
        ranges.  Parameters not listed here show as text inputs.
    read_only_params : set[str]
        Parameter names that should be displayed as read-only.
    empty_message : str
        Message shown when the scene is empty.
    auto_refresh : bool
        If True (default), populate from the SceneGraph on creation.
    """

    _esm = traitlets.Unicode(_DATATREE_PANEL_ESM).tag(sync=True)
    _dt_layers = traitlets.List([]).tag(sync=True)
    _dt_param_cmd = traitlets.Dict({}).tag(sync=True)
    _dt_refresh_cmd = traitlets.Dict({}).tag(sync=True)
    _panel_title = traitlets.Unicode("Scene Tree").tag(sync=True)
    _empty_message = traitlets.Unicode("No objects loaded").tag(sync=True)

    def __init__(
        self,
        client,
        *,
        title: str = "Scene Tree",
        param_hints: dict[str, tuple[float, float, float]] | None = None,
        read_only_params: set[str] | None = None,
        empty_message: str = "No objects loaded",
        auto_refresh: bool = True,
        **kwargs,
    ):
        # Accept (and ignore) legacy kwargs for backward compat
        kwargs.pop("field_icons", None)
        kwargs.pop("field_order", None)
        super().__init__(**kwargs)
        self._client = resolve_client(client)
        self._panel_title = title
        self._empty_message = empty_message
        self._param_hints: dict[str, tuple[float, float, float]] = dict(
            param_hints or {}
        )
        self._read_only_params: frozenset[str] = frozenset(
            read_only_params or set()
        )

        self.observe(self._handle_param_cmd, names=["_dt_param_cmd"])
        self.observe(self._handle_refresh_cmd, names=["_dt_refresh_cmd"])
        if auto_refresh:
            self.refresh()

    # -- Public API ----------------------------------------------------------

    def refresh(self):
        """Rebuild the tree from the client's SceneGraph."""
        if not self._client.connected:
            self._dt_layers = []
            return

        client = self._client

        def _do():
            try:
                sg = client.scene_graph
                if sg is None:
                    layers: list[dict] = []
                else:
                    layers = self._build_hierarchy(sg)

                def _apply():
                    self._dt_layers = layers

                if not run_on_kernel_loop(_apply):
                    _apply()
            except Exception:
                logger.exception("DataTreePanel refresh failed")

                def _clear():
                    self._dt_layers = []

                if not run_on_kernel_loop(_clear):
                    _clear()

        threading.Thread(
            target=_do, daemon=True, name="tsd-datatree-refresh"
        ).start()

    # -- Hierarchy building --------------------------------------------------

    def _build_hierarchy(self, sg) -> list[dict]:
        """Walk the SceneGraph layer tree and build a JSON-serializable tree."""
        from ..anari_types import ANARI_UNKNOWN

        result: list[dict] = []
        for lname in sg.layer_names:
            root = sg.layer_tree(lname)
            if root is None:
                continue
            layer: dict = {
                "name": lname,
                "layerName": lname,
                "nodeName": lname,
                "nodeType": "layer",
                "attrs": [],
                "children": [
                    self._node_to_dict(c, sg, lname) for c in root.children
                ],
            }
            if root.is_transform and root.transform_srt:
                layer["attrs"] = self._srt_attrs(root.transform_srt)
            result.append(layer)
        return result

    def _node_to_dict(self, node, sg, layer_name: str) -> dict:
        """Convert a LayerNodeInfo to a JSON-serializable dict, recursively."""
        from ..anari_types import ANARI_UNKNOWN, anari_type_name

        d: dict = {
            "name": node.name,
            "layerName": layer_name,
            "nodeName": node.name,
            "enabled": node.enabled,
            "attrs": [],
            "children": [
                self._node_to_dict(c, sg, layer_name) for c in node.children
            ],
        }

        if node.is_transform:
            d["nodeType"] = "transform"
            if node.transform_srt:
                d["attrs"] = self._srt_attrs(node.transform_srt)
        elif node.object_type != ANARI_UNKNOWN and node.object_index is not None:
            d["nodeType"] = "object"
        else:
            d["nodeType"] = "group"

        if node.object_type != ANARI_UNKNOWN and node.object_index is not None:
            d["objectType"] = node.object_type
            d["objectIndex"] = node.object_index
            d["objectTypeName"] = anari_type_name(node.object_type)

            obj = sg.get_object(node.object_type, node.object_index)
            if obj is not None:
                if not d["name"] and obj.name:
                    d["name"] = obj.name
                sub = obj.subtype
                if sub and sub != "none":
                    d["subtype"] = sub
                obj_attrs = self._object_attrs(obj)
                d["attrs"] = d.get("attrs", []) + obj_attrs
                ref_children = self._linked_object_children(obj, sg)
                if ref_children:
                    d["children"] = d.get("children", []) + ref_children

        return d

    def _linked_object_children(self, obj, sg) -> list[dict]:
        """Build child dicts for objects referenced by this object's params."""
        from ..anari_types import anari_type_name

        children: list[dict] = []
        for pname, pnode in obj.parameters.items():
            value_child = pnode.child("value")
            vnode = value_child if value_child is not None else pnode
            if vnode.object_index is None:
                continue
            ref_obj = sg.get_object(vnode.dtype, vnode.object_index)
            if ref_obj is None:
                continue
            sub = ref_obj.subtype
            child: dict = {
                "name": ref_obj.name or f"{anari_type_name(vnode.dtype)}@{vnode.object_index}",
                "nodeType": "object",
                "objectType": vnode.dtype,
                "objectIndex": vnode.object_index,
                "objectTypeName": anari_type_name(vnode.dtype),
                "enabled": True,
                "attrs": self._object_attrs(ref_obj),
                "children": [],
            }
            if sub and sub != "none":
                child["subtype"] = sub
            children.append(child)
        return children

    # -- Attribute extraction ------------------------------------------------

    @staticmethod
    def _srt_attrs(srt: tuple) -> list[dict]:
        """Convert a 9-float SRT tuple into three editable vec3 attrs."""
        return [
            {
                "name": "scale",
                "type": "float32_vec3",
                "value": [float(srt[0]), float(srt[1]), float(srt[2])],
                "isSRT": True,
            },
            {
                "name": "rotation",
                "type": "float32_vec3",
                "value": [float(srt[3]), float(srt[4]), float(srt[5])],
                "isSRT": True,
            },
            {
                "name": "translation",
                "type": "float32_vec3",
                "value": [float(srt[6]), float(srt[7]), float(srt[8])],
                "isSRT": True,
            },
        ]

    def _object_attrs(self, obj) -> list[dict]:
        """Extract ALL attributes (params + metadata) from an ObjectInfo."""
        from ..anari_types import anari_type_name  # noqa: F811

        attrs: list[dict] = []
        for name, node in obj.parameters.items():
            a = self._datanode_to_attr(name, node, read_only=False)
            if a is not None:
                attrs.append(a)
        for name, node in obj.metadata.items():
            a = self._datanode_to_attr(name, node, read_only=True)
            if a is not None:
                a["category"] = "metadata"
                attrs.append(a)
        return attrs

    def _datanode_to_attr(
        self, name: str, node, read_only: bool
    ) -> dict | None:
        """Convert a TSD parameter DataNode into a JSON-serializable attr dict.

        Delegates to :func:`~tsd_client.utils.datanode_to_attr` and applies
        panel-specific overrides (``read_only_params``, ``param_hints``).
        """
        if name in self._read_only_params:
            read_only = True

        entry = _datanode_to_attr_base(name, node, read_only)
        if entry is None:
            return None

        if not entry.get("readOnly") and name in self._param_hints:
            lo, hi, step = self._param_hints[name]
            entry["min"] = lo
            entry["max"] = hi
            entry["step"] = step

        return entry

    # -- Command handlers ----------------------------------------------------

    _SRT_NAMES = {"scale", "rotation", "translation"}

    def _handle_param_cmd(self, change):
        cmd = change.get("new", {})
        if not cmd:
            return
        name = cmd.get("name")
        ptype = cmd.get("type")
        value = cmd.get("value")
        if not name or not ptype:
            return

        if ptype in ("float32_vec2", "float32_vec3", "float32_vec4") and isinstance(
            value, list
        ):
            value = tuple(float(x) for x in value)

        client = self._client
        sg = client.cached_scene_graph() or client.scene_graph
        if sg is None:
            return

        if cmd.get("isSRT") and name in self._SRT_NAMES:
            layer_name = cmd.get("layerName")
            node_name = cmd.get("nodeName")
            if layer_name and node_name:
                self._handle_srt_change(sg, client, layer_name, node_name, name, value)
            return

        obj_type = cmd.get("objectType")
        obj_idx = cmd.get("objectIndex")
        if obj_type is None or obj_idx is None:
            return

        if name == "visible" and ptype == "bool":
            self._handle_visible_toggle(sg, client, int(obj_type), int(obj_idx), bool(value))
            return

        try:
            sg.set_object_parameter(
                int(obj_type), int(obj_idx), name, value, param_type=ptype
            )
            logger.debug(
                "Set %d[%d].%s (%s) = %s", obj_type, obj_idx, name, ptype, value
            )
        except Exception:
            logger.exception("Failed to set %d[%d].%s", obj_type, obj_idx, name)

    def _handle_srt_change(self, sg, client, layer_name: str, node_name: str,
                           component: str, value):
        """Update a single SRT component (scale/rotation/translation) on a
        layer tree transform node and send the layer update."""
        try:
            info = sg.find_node(layer_name, node_name)
            if info is None or not info.is_transform or info.transform_srt is None:
                logger.warning("Transform node %r not found in layer %r",
                               node_name, layer_name)
                return
            srt = list(info.transform_srt)
            vec = tuple(float(x) for x in value) if not isinstance(value, tuple) else value
            if component == "scale":
                srt[0:3] = vec
            elif component == "rotation":
                srt[3:6] = vec
            elif component == "translation":
                srt[6:9] = vec
            sg._apply_srt(info, tuple(srt[0:3]), tuple(srt[3:6]), tuple(srt[6:9]))
            client.send_layer_update(sg, layer_name)
            logger.debug("Updated %s on %r/%r = %s", component, layer_name, node_name, vec)
        except Exception:
            logger.exception("Failed to update SRT %s on %r/%r",
                             component, layer_name, node_name)

    def _handle_visible_toggle(self, sg, client, obj_type: int, obj_idx: int, visible: bool):
        """Toggle visibility via the layer tree node's ``enabled`` flag."""
        try:
            layer_name = sg.set_object_visible(obj_type, obj_idx, visible)
            if layer_name is None:
                logger.warning("No layer node found for %d[%d]", obj_type, obj_idx)
                return
            client.send_layer_update(sg, layer_name)
            logger.debug(
                "Toggled visible=%s for %d[%d] (layer %r)",
                visible, obj_type, obj_idx, layer_name,
            )
        except Exception:
            logger.exception("Failed to toggle visible for %d[%d]", obj_type, obj_idx)

    def _handle_refresh_cmd(self, change):
        self.refresh()
