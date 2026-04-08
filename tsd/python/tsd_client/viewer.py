# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base interactive viewer widget for TSD render servers.

:class:`TSDViewer` provides the generic AnyWidget-based rendering canvas,
orbit/dolly/pan mouse controls, camera animation (turntable, rock,
keyframes with Catmull-Rom), a detach-to-popup feature, and an optional
scene-time toolbar.

Downstream projects subclass ``TSDViewer`` and override :meth:`_send_view`
to route camera state to their server, and :meth:`_setup_rendering` to
register application-specific message handlers.

Usage::

    from tsd_client.viewer import TSDViewer

    class MyViewer(TSDViewer):
        def _send_view(self):
            self.client.send_view(
                self._azimuth, self._elevation, self._distance, self._lookat
            )

    viewer = MyViewer(client)
    viewer
"""

from __future__ import annotations

import math
import struct
import time
import threading
import logging
from io import BytesIO
from typing import Any

import anywidget
import traitlets

from .client import TSDClient
from .protocol import MessageType
from .scene import build_parameter_change_payload, is_valid_object_pool_index

logger = logging.getLogger("tsd_client.viewer")

from .utils import (
    lerp as _lerp,
    smoothstep as _smoothstep,
    normalize3 as _normalize3,
    run_on_kernel_loop as _run_on_kernel_loop,
)


def _normalize_frame_to_jpeg(payload: bytes, width: int, height: int) -> bytes:
    """Return *payload* as JPEG bytes for the browser canvas.

    TSD servers typically send raw RGBA (or occasionally PNG). The viewer
    front-end decodes JPEG only, so convert when needed.

    Color buffers from VisRTX / OpenGL-style pipelines are usually stored with
    the **first row = bottom** of the viewport; PIL and HTML canvas expect the
    **first row = top**, so we flip vertically whenever we decode raster data.
    """
    from PIL import Image

    def _to_jpeg_rgb_top_down(im: Image.Image) -> bytes:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        out = BytesIO()
        im.convert("RGB").save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue()

    if not payload:
        return payload
    if len(payload) >= 3 and payload[:3] == b"\xff\xd8\xff":
        try:
            im = Image.open(BytesIO(payload))
            im.load()
            return _to_jpeg_rgb_top_down(im)
        except Exception:
            return payload

    try:
        im = Image.open(BytesIO(payload))
        im.load()
        return _to_jpeg_rgb_top_down(im)
    except Exception:
        pass

    w, h = width, height
    if w > 0 and h > 0:
        n_rgba = w * h * 4
        if len(payload) >= n_rgba:
            try:
                im = Image.frombytes("RGBA", (w, h), bytes(payload[:n_rgba]))
                return _to_jpeg_rgb_top_down(im)
            except Exception:
                pass
        n_rgb = w * h * 3
        if len(payload) >= n_rgb:
            try:
                im = Image.frombytes("RGB", (w, h), bytes(payload[:n_rgb]))
                return _to_jpeg_rgb_top_down(im)
            except Exception:
                pass

    logger.warning(
        "Unrecognized frame buffer (%d bytes); expected JPEG/PNG or "
        "RGB/RGBA for %dx%d",
        len(payload),
        w,
        h,
    )
    return payload


def _unpack_size_t_payload(payload: bytes) -> int | None:
    """Little-endian ``size_t`` from a short message (4 or 8 bytes)."""
    if len(payload) >= 8:
        return struct.unpack_from("<Q", payload, 0)[0]
    if len(payload) >= 4:
        return struct.unpack_from("<I", payload, 0)[0]
    return None


def _vec_cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


_vec_norm3 = _normalize3


def _tsd_azel_world_dir(az_rad: float, el_rad: float) -> tuple[float, float, float]:
    """Unit direction from camera toward ``lookat`` (VisRTX ``Manipulator``, ``UpAxis::POS_Y``).

    Matches ``Manipulator::azelToDirection`` + ``update()``:
    internal angles are ``radians(-azimuth_deg)`` and ``radians(-elevation_deg)``,
    then world components ``(sin(az)cos(el), sin(el), cos(az)cos(el))``.
    """
    x = math.sin(az_rad) * math.cos(el_rad)
    y = math.cos(az_rad) * math.cos(el_rad)
    z = math.sin(el_rad)
    wx, wy, wz = x, z, y
    return _vec_norm3((wx, wy, wz))


def _tsd_manipulator_perspective_pose(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    lookat: list[float] | tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """ANARI ``position`` / ``direction`` / ``up`` matching ``updateCameraParametersPerspective``."""
    az = math.radians(-azimuth_deg)
    el = math.radians(-elevation_deg)
    d = abs(float(distance))
    lx, ly, lz = float(lookat[0]), float(lookat[1]), float(lookat[2])

    w = _tsd_azel_world_dir(az, el)
    pos = (lx - w[0] * d, ly - w[1] * d, lz - w[2] * d)

    to_local = (-w[0], -w[1], -w[2])
    w_alt = _tsd_azel_world_dir(az, el + 3.0)
    alt_to_local = (-w_alt[0], -w_alt[1], -w_alt[2])
    from_local = (w[0] * d, w[1] * d, w[2] * d)

    camera_right = _vec_cross(to_local, alt_to_local)
    camera_up = _vec_cross(camera_right, from_local)
    cr_s = sum(c * c for c in camera_right)
    if cr_s < 1e-30:
        up = (0.0, 1.0, 0.0)
        right = _vec_norm3(_vec_cross(up, (-w[0], -w[1], -w[2])))
    else:
        right = _vec_norm3(camera_right)
        up = _vec_norm3(camera_up)

    return pos, w, up


def _tsd_manipulator_right_up(
    azimuth_deg: float, elevation_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Screen pan basis matching ``Manipulator::pan`` (``m_right``, ``m_up``)."""
    az = math.radians(-azimuth_deg)
    el = math.radians(-elevation_deg)
    w = _tsd_azel_world_dir(az, el)
    to_local = (-w[0], -w[1], -w[2])
    w_alt = _tsd_azel_world_dir(az, el + 3.0)
    alt_to_local = (-w_alt[0], -w_alt[1], -w_alt[2])
    from_local = w
    camera_right = _vec_cross(to_local, alt_to_local)
    camera_up = _vec_cross(camera_right, from_local)
    return _vec_norm3(camera_right), _vec_norm3(camera_up)


def orbit_camera_pose(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    lookat: list[float] | tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Convert orbit controls to ANARI camera ``position``, ``direction``, ``up``.

    Uses the same spherical mapping and ``up`` construction as VisRTX
    ``tsd::rendering::Manipulator`` with default ``UpAxis::POS_Y`` so remote
    views match the stock TCV / tsd interactive clients.
    """
    return _tsd_manipulator_perspective_pose(
        azimuth_deg, elevation_deg, distance, lookat
    )


def _lerp_camera(c0: dict, c1: dict, t: float) -> dict:
    """Linearly interpolate between two camera dicts."""
    return {
        "azimuth": _lerp(c0["azimuth"], c1["azimuth"], t),
        "elevation": _lerp(c0["elevation"], c1["elevation"], t),
        "distance": _lerp(c0["distance"], c1["distance"], t),
        "lookat": [_lerp(a, b, t) for a, b in zip(c0["lookat"], c1["lookat"])],
    }


# ---------------------------------------------------------------------------
# ESM — full viewer front-end
# ---------------------------------------------------------------------------

_TSD_VIEWER_ESM = """
export function render({ model, el }) {
  const width  = model.get('_width');
  const height = model.get('_height');
  let activeWidth = width;
  let activeHeight = height;

  const COLORS = {
    bg:       '#1a1a1a',
    border:   '#444',
    toolbar:  '#252525',
    text:     '#ccc',
    textDim:  '#888',
    accent:   '#4a9eff',
    btnBg:    '#333',
    btnHover: '#444',
    btnActive:'#4a9eff',
  };

  const wrapper = document.createElement('div');
  wrapper.style.display = 'inline-block';
  wrapper.style.fontFamily = "'SF Mono', 'Fira Code', 'Consolas', monospace";
  wrapper.style.position = 'relative';
  wrapper.style.zIndex = '1';

  const dpr = Math.max(1, window.devicePixelRatio || 1);

  const canvas = document.createElement('canvas');
  canvas.width  = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width  = width + 'px';
  canvas.style.height = height + 'px';
  canvas.style.display = 'block';
  canvas.style.cursor = 'grab';
  canvas.style.border = '1px solid ' + COLORS.border;
  canvas.style.borderRadius = '6px 6px 0 0';
  canvas.style.background = COLORS.bg;
  canvas.style.touchAction = 'none';
  canvas.tabIndex = 0;
  canvas.style.outline = 'none';
  canvas.addEventListener('mousedown', () => {
    try { canvas.focus({ preventScroll: true }); } catch (e) { canvas.focus(); }
  });

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#666';
  ctx.font = '14px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Connecting...', width / 2, height / 2);

  // -- Toolbar --------------------------------------------------------

  const toolbar = document.createElement('div');
  toolbar.style.cssText = `
    display: flex; align-items: center; gap: 6px;
    padding: 6px 8px;
    background: ${COLORS.toolbar};
    border: 1px solid ${COLORS.border}; border-top: none;
    border-radius: 0;
    font-size: 12px; color: ${COLORS.text};
    user-select: none;
  `;

  function makeBtn(label, title) {
    const b = document.createElement('button');
    b.textContent = label;
    b.title = title || '';
    b.style.cssText = `
      background: ${COLORS.btnBg}; color: ${COLORS.text};
      border: 1px solid ${COLORS.border}; border-radius: 4px;
      padding: 3px 8px; cursor: pointer; font-size: 12px;
      font-family: inherit; line-height: 1.3;
    `;
    b.addEventListener('mouseenter', () => b.style.background = COLORS.btnHover);
    b.addEventListener('mouseleave', () => {
      b.style.background = b.dataset.active === '1' ? COLORS.btnActive : COLORS.btnBg;
    });
    return b;
  }

  const playBtn   = makeBtn('\\u25B6', 'Play / Pause animation');
  const modeBtn   = makeBtn('Turntable', 'Toggle animation mode');
  const recBtn    = makeBtn('\\u23FA REC', 'Record camera keyframe');

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1'; speedSlider.max = '120'; speedSlider.value = '30';
  speedSlider.style.cssText = 'width: 80px; accent-color: ' + COLORS.accent;
  speedSlider.title = 'Animation speed';

  const speedLabel = document.createElement('span');
  speedLabel.style.cssText = 'color: ' + COLORS.textDim + '; min-width: 42px';
  speedLabel.textContent = '30\\u00b0/s';

  const sep = () => {
    const d = document.createElement('span');
    d.style.cssText = 'width:1px; height:16px; background:' + COLORS.border;
    return d;
  };

  const detachable = model.get('_detachable');
  let detachBtn = null;
  if (detachable) {
    detachBtn = makeBtn('\\u29C9', 'Detach viewer into separate window');
  }

  toolbar.append(playBtn, modeBtn, sep(), speedSlider, speedLabel, sep(), recBtn);
  if (detachBtn) toolbar.append(sep(), detachBtn);

  // -- Scene-time toolbar (optional) ----------------------------------

  const showTimeBar = model.get('_show_time_toolbar');
  let timeBar = null;

  if (showTimeBar) {
    timeBar = document.createElement('div');
    timeBar.style.cssText = `
      display: flex; align-items: center; gap: 6px;
      padding: 5px 8px;
      background: ${COLORS.toolbar};
      border: 1px solid ${COLORS.border}; border-top: none;
      font-size: 12px; color: ${COLORS.text};
      user-select: none;
    `;

    const timeLbl = document.createElement('span');
    timeLbl.style.color = COLORS.textDim;
    timeLbl.textContent = 'Time';

    const timePlayBtn = makeBtn('\\u25B6', 'Play / Pause scene time');
    const timeStepBack = makeBtn('\\u23EA', 'Step backward');
    const timeStepFwd  = makeBtn('\\u23E9', 'Step forward');

    const timeSlider = document.createElement('input');
    timeSlider.type = 'range';
    timeSlider.min = '0'; timeSlider.max = '1'; timeSlider.step = '0.001';
    timeSlider.value = '0';
    timeSlider.style.cssText = 'flex: 1; accent-color: ' + COLORS.accent;
    timeSlider.title = 'Scene animation time (0 \\u2013 1)';

    const timeValue = document.createElement('span');
    timeValue.style.cssText =
      'color:' + COLORS.textDim +
      '; min-width: 90px; text-align: right; font-variant-numeric: tabular-nums';
    timeValue.textContent = 't=0.000';

    const timeSpeedSlider = document.createElement('input');
    timeSpeedSlider.type = 'range';
    timeSpeedSlider.min = '0.01'; timeSpeedSlider.max = '1';
    timeSpeedSlider.step = '0.01'; timeSpeedSlider.value = '0.10';
    timeSpeedSlider.style.cssText = 'width: 60px; accent-color: ' + COLORS.accent;
    timeSpeedSlider.title = 'Playback speed (units / sec)';

    const timeSpeedLbl = document.createElement('span');
    timeSpeedLbl.style.cssText =
      'color:' + COLORS.textDim + '; min-width: 50px';
    timeSpeedLbl.textContent = '0.10/s';

    timeBar.append(
      timeLbl, timePlayBtn, timeStepBack, timeStepFwd, sep(),
      timeSlider, timeValue, sep(), timeSpeedSlider, timeSpeedLbl,
    );

    // -- Scene-time toolbar logic -------------------------------------

    function syncTimePlayBtn() {
      const playing = model.get('_scene_time_playing');
      timePlayBtn.textContent = playing ? '\\u23F8' : '\\u25B6';
      timePlayBtn.dataset.active = playing ? '1' : '0';
      timePlayBtn.style.background = playing ? COLORS.btnActive : COLORS.btnBg;
    }

    function updateTimeLabel() {
      const t = parseFloat(model.get('_scene_time') || 0);
      timeSlider.value = t;
      const info = model.get('_scene_time_info') || {};
      const maxSteps = info.maxSteps || 0;
      if (maxSteps > 1) {
        const step = Math.min(Math.floor(t * maxSteps), maxSteps - 1);
        timeValue.textContent = 'Step ' + step + '/' + (maxSteps - 1);
      } else {
        timeValue.textContent = 't=' + t.toFixed(3);
      }
    }

    timePlayBtn.addEventListener('click', () => {
      const next = !model.get('_scene_time_playing');
      model.set('_scene_time_cmd', {
        cmd: next ? 'play' : 'pause',
        speed: parseFloat(timeSpeedSlider.value),
        _t: Date.now(),
      });
      model.save_changes();
    });

    timeStepBack.addEventListener('click', () => {
      model.set('_scene_time_cmd', { cmd: 'step', delta: -1, _t: Date.now() });
      model.save_changes();
    });

    timeStepFwd.addEventListener('click', () => {
      model.set('_scene_time_cmd', { cmd: 'step', delta: 1, _t: Date.now() });
      model.save_changes();
    });

    timeSlider.addEventListener('input', () => {
      model.set('_scene_time_cmd', {
        cmd: 'seek', time: parseFloat(timeSlider.value), _t: Date.now(),
      });
      model.save_changes();
    });

    timeSpeedSlider.addEventListener('input', () => {
      const v = parseFloat(timeSpeedSlider.value);
      timeSpeedLbl.textContent = v.toFixed(2) + '/s';
      if (model.get('_scene_time_playing')) {
        model.set('_scene_time_cmd', { cmd: 'speed', speed: v, _t: Date.now() });
        model.save_changes();
      }
    });

    model.on('change:_scene_time', updateTimeLabel);
    model.on('change:_scene_time_info', updateTimeLabel);
    model.on('change:_scene_time_playing', syncTimePlayBtn);
    syncTimePlayBtn();
    updateTimeLabel();
  }

  // -- Status bar -----------------------------------------------------

  const status = document.createElement('div');
  status.style.cssText = `
    font-size: 11px; color: ${COLORS.textDim};
    padding: 3px 8px; background: ${COLORS.toolbar};
    border: 1px solid ${COLORS.border}; border-top: none;
    border-radius: 0 0 6px 6px;
  `;
  status.textContent = model.get('_status');

  wrapper.append(canvas, toolbar);
  if (timeBar) wrapper.appendChild(timeBar);
  wrapper.appendChild(status);
  el.appendChild(wrapper);

  // -- Toolbar logic --------------------------------------------------

  const modes = ['turntable', 'rock'];
  let modeIdx = 0;

  function syncPlayBtn() {
    const playing = model.get('_anim_playing');
    playBtn.textContent = playing ? '\\u23F8' : '\\u25B6';
    playBtn.dataset.active = playing ? '1' : '0';
    playBtn.style.background = playing ? COLORS.btnActive : COLORS.btnBg;
  }

  playBtn.addEventListener('click', () => {
    const next = !model.get('_anim_playing');
    model.set('_anim_cmd', {
      cmd: next ? 'play' : 'stop',
      mode: modes[modeIdx],
      speed: parseFloat(speedSlider.value),
      _t: Date.now(),
    });
    model.save_changes();
  });

  modeBtn.addEventListener('click', () => {
    modeIdx = (modeIdx + 1) % modes.length;
    modeBtn.textContent = modes[modeIdx].charAt(0).toUpperCase() + modes[modeIdx].slice(1);
    if (model.get('_anim_playing')) {
      model.set('_anim_cmd', {
        cmd: 'play', mode: modes[modeIdx],
        speed: parseFloat(speedSlider.value), _t: Date.now(),
      });
      model.save_changes();
    }
  });

  speedSlider.addEventListener('input', () => {
    const v = parseFloat(speedSlider.value);
    speedLabel.textContent = v + '\\u00b0/s';
    if (model.get('_anim_playing')) {
      model.set('_anim_cmd', {
        cmd: 'speed', speed: v, _t: Date.now(),
      });
      model.save_changes();
    }
  });

  recBtn.addEventListener('click', () => {
    recBtn.style.background = '#c44';
    setTimeout(() => recBtn.style.background = COLORS.btnBg, 300);
    model.set('_anim_cmd', { cmd: 'record', _t: Date.now() });
    model.save_changes();
  });

  model.on('change:_anim_playing', syncPlayBtn);
  syncPlayBtn();

  // -- Detach state ---------------------------------------------------

  let detached = false;
  let popupWin = null;
  let activeCtx = ctx;

  function showPlaceholder() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#666';
    ctx.font = '14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('\\u29C9 Detached \\u2014 rendering in separate window', width / 2, height / 2);
    ctx.fillText('Click \\u29C9 to reattach', width / 2, height / 2 + 22);
  }

  // -- Frame display --------------------------------------------------

  const img = new window.Image();

  function updateFrame() {
    const data = model.get('_frame_jpeg');
    if (data && data.byteLength > 0) {
      const blob = new Blob([data], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      img.onload = () => {
        activeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        activeCtx.drawImage(img, 0, 0, activeWidth, activeHeight);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    }
  }

  model.on('change:_frame_jpeg', updateFrame);
  model.on('change:_status', () => {
    status.textContent = model.get('_status');
  });

  // -- Mouse interaction ----------------------------------------------

  function setupMouseHandlers(targetCanvas) {
    let mouseDown = false;
    let btn = -1;
    let lastX = 0;
    let lastY = 0;

    targetCanvas.addEventListener('mousedown', (e) => {
      e.preventDefault();
      mouseDown = true;
      btn = e.button;
      lastX = e.offsetX;
      lastY = e.offsetY;
      targetCanvas.style.cursor = 'grabbing';
    });

    targetCanvas.addEventListener('mouseup', (e) => {
      e.preventDefault();
      mouseDown = false;
      btn = -1;
      targetCanvas.style.cursor = 'grab';
    });

    targetCanvas.addEventListener('mouseleave', () => {
      mouseDown = false;
      btn = -1;
      targetCanvas.style.cursor = 'grab';
    });

    targetCanvas.addEventListener('mousemove', (e) => {
      if (!mouseDown) return;
      e.preventDefault();
      const dw = activeWidth > 0 ? activeWidth : (width > 0 ? width : 1);
      const dh = activeHeight > 0 ? activeHeight : (height > 0 ? height : 1);
      const dx = (e.offsetX - lastX) / dw;
      const dy = (e.offsetY - lastY) / dh;
      lastX = e.offsetX;
      lastY = e.offsetY;

      let action = 'none';
      if (btn === 0 && !e.shiftKey && !e.altKey)        action = 'orbit';
      else if (btn === 2 || (btn === 0 && e.shiftKey))  action = 'dolly';
      else if (btn === 1 || (btn === 0 && e.altKey))    action = 'pan';

      if (action !== 'none') {
        model.set('_mouse_event', { action, dx, dy, _t: Date.now() });
        model.save_changes();
      }
    });

    targetCanvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const dy = e.deltaY > 0 ? 0.01 : -0.01;
      model.set('_mouse_event', { action: 'dolly', dx: 0, dy, _t: Date.now() });
      model.save_changes();
    }, { passive: false });

    targetCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  setupMouseHandlers(canvas);

  // -- Detach / Reattach ----------------------------------------------

  function reattach() {
    if (!detached) return;
    activeCtx = ctx;
    detached = false;
    if (popupWin && !popupWin.closed) popupWin.close();
    popupWin = null;
    if (activeWidth !== width || activeHeight !== height) {
      activeWidth = width;
      activeHeight = height;
      model.set('_resize_event', { width: width, height: height, _t: Date.now() });
      model.save_changes();
    }
    if (detachBtn) {
      detachBtn.dataset.active = '0';
      detachBtn.style.background = COLORS.btnBg;
    }
  }

  function detachViewer() {
    if (detached) { reattach(); return; }

    const popW = width + 40;
    const popH = height + 60;
    popupWin = window.open('', 'TSD_Viewer_' + Date.now(),
      'width=' + popW + ',height=' + popH + ',resizable=yes,scrollbars=no');
    if (!popupWin) {
      alert('Popup blocked \\u2014 please allow popups for this site.');
      return;
    }

    const doc = popupWin.document;
    doc.title = 'TSD Viewer';
    doc.body.style.cssText =
      'margin:0; padding:20px; background:' + COLORS.bg +
      '; display:flex; justify-content:center; align-items:flex-start;' +
      " font-family:'SF Mono','Fira Code','Consolas',monospace;";

    const popCanvas = doc.createElement('canvas');
    popCanvas.width = width * dpr;
    popCanvas.height = height * dpr;
    popCanvas.style.width = width + 'px';
    popCanvas.style.height = height + 'px';
    popCanvas.style.display = 'block';
    popCanvas.style.cursor = 'grab';
    popCanvas.style.border = '1px solid ' + COLORS.border;
    popCanvas.style.borderRadius = '6px';
    popCanvas.style.background = COLORS.bg;

    const popCtx = popCanvas.getContext('2d');
    popCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    doc.body.appendChild(popCanvas);
    setupMouseHandlers(popCanvas);

    let resizeTimer = 0;
    popupWin.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        const pad = 40;
        const newW = Math.max(160, popupWin.innerWidth - pad);
        const newH = Math.max(120, popupWin.innerHeight - pad);
        if (newW === activeWidth && newH === activeHeight) return;
        activeWidth = newW;
        activeHeight = newH;
        popCanvas.width = newW * dpr;
        popCanvas.height = newH * dpr;
        popCanvas.style.width = newW + 'px';
        popCanvas.style.height = newH + 'px';
        popCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        model.set('_resize_event', { width: newW, height: newH, _t: Date.now() });
        model.save_changes();
      }, 150);
    });

    activeCtx = popCtx;
    detached = true;
    showPlaceholder();

    if (detachBtn) {
      detachBtn.dataset.active = '1';
      detachBtn.style.background = COLORS.btnActive;
    }

    popupWin.addEventListener('beforeunload', () => {
      activeCtx = ctx;
      detached = false;
      popupWin = null;
      if (activeWidth !== width || activeHeight !== height) {
        activeWidth = width;
        activeHeight = height;
        model.set('_resize_event', { width: width, height: height, _t: Date.now() });
        model.save_changes();
      }
      if (detachBtn) {
        detachBtn.dataset.active = '0';
        detachBtn.style.background = COLORS.btnBg;
      }
    });
  }

  if (detachBtn) {
    detachBtn.addEventListener('click', detachViewer);
  }

  return () => {
    if (popupWin && !popupWin.closed) popupWin.close();
  };
}
"""


class TSDViewer(anywidget.AnyWidget):
    """Interactive Jupyter viewer for any TSD-based render server.

    Displays streamed rendering with orbit/dolly/pan mouse controls, camera
    animation (turntable, rock, keyframe), toolbar controls, and optional
    detach-to-popup.

    Subclasses **must** override :meth:`_send_view` to route camera state
    to the server.

    Parameters
    ----------
    client : TSDClient
        Connected client instance.
    width, height : int
        Viewport resolution in pixels.
    detachable : bool
        If True, show a button to pop the viewer into a separate window.
    show_time_toolbar : bool
        If True, display a scene-time playback toolbar.
    """

    _esm = traitlets.Unicode(_TSD_VIEWER_ESM).tag(sync=True)

    # Synced traits
    _frame_jpeg = traitlets.Bytes(b"").tag(sync=True)
    _width = traitlets.Int(800).tag(sync=True)
    _height = traitlets.Int(600).tag(sync=True)
    _status = traitlets.Unicode("Disconnected").tag(sync=True)
    _mouse_event = traitlets.Dict({}).tag(sync=True)
    _anim_playing = traitlets.Bool(False).tag(sync=True)
    _anim_cmd = traitlets.Dict({}).tag(sync=True)
    _detachable = traitlets.Bool(True).tag(sync=True)
    _resize_event = traitlets.Dict({}).tag(sync=True)
    _show_time_toolbar = traitlets.Bool(False).tag(sync=True)

    # Scene-time synced traits
    _scene_time = traitlets.Float(0.0).tag(sync=True)
    _scene_time_playing = traitlets.Bool(False).tag(sync=True)
    _scene_time_cmd = traitlets.Dict({}).tag(sync=True)
    _scene_time_info = traitlets.Dict({}).tag(sync=True)

    def __init__(
        self,
        client: TSDClient,
        width: int = 800,
        height: int = 600,
        detachable: bool = True,
        show_time_toolbar: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = client
        self._width = width
        self._height = height
        self._detachable = detachable
        self._show_time_toolbar = show_time_toolbar

        # Camera state (orbit decomposition)
        self._azimuth = 0.0
        self._elevation = 20.0
        self._distance = 5.0
        self._camera_fitted = False
        self._lookat = [0.0, 0.0, 0.0]

        # From ``CLIENT_RECEIVE_CURRENT_CAMERA`` (pool index); may stay None
        self._camera_pool_index: int | None = None

        # Frame state
        self._frame_count = 0
        self._prev_width = width
        self._prev_height = height
        self._frame_buf_w = 0
        self._frame_buf_h = 0

        # Camera animation state
        self._anim_thread: threading.Thread | None = None
        self._anim_stop = threading.Event()
        self._anim_mode = "turntable"
        self._anim_speed = 30.0
        self._anim_rock_amplitude = 60.0
        self._anim_rock_origin = 0.0
        self._keyframes: list[dict] = []

        # Scene-time animation state
        self._scene_time_thread: threading.Thread | None = None
        self._scene_time_stop = threading.Event()
        self._scene_time_speed = 0.10
        self._scene_time_value = 0.0
        self._time_info: dict = {}

        # React to JS events
        self.observe(self._handle_mouse_event, names=["_mouse_event"])
        self.observe(self._handle_anim_cmd, names=["_anim_cmd"])
        self.observe(self._handle_resize_event, names=["_resize_event"])
        self.observe(self._handle_scene_time_cmd, names=["_scene_time_cmd"])

        self._setup_and_start()

    # ========================================================================
    # Connection / setup
    # ========================================================================

    def _setup_and_start(self):
        """Register base message handlers and start rendering."""
        if not self.client.connected:
            return

        try:
            self._status = f"Connecting to {self.client.host}:{self.client.port}..."

            self.client.on_disconnect = self._handle_server_disconnect
            self.client.register_handler(
                MessageType.CLIENT_RECEIVE_FRAME_BUFFER_COLOR, self._on_frame
            )
            self.client.register_handler(
                MessageType.CLIENT_RECEIVE_FRAME_CONFIG, self._on_frame_config
            )
            self.client.register_handler(
                MessageType.CLIENT_RECEIVE_SCENE,
                lambda _t, _p: logger.debug("Received scene (%d bytes)", len(_p)),
            )
            self.client.register_handler(
                MessageType.CLIENT_SCENE_TRANSFER_BEGIN, lambda _t, _p: None
            )
            self.client.register_handler(
                MessageType.PING, lambda _t, _p: None
            )
            self.client.register_handler(
                MessageType.ERROR,
                lambda _t, p: logger.error(
                    "Server error: %s", p.decode("utf-8", errors="replace")
                ),
            )
            self.client.register_handler(
                MessageType.CLIENT_RECEIVE_CURRENT_CAMERA,
                self._on_receive_current_camera,
            )

            self._setup_rendering()

            self.client.send_frame_config(self._width, self._height)
            self.client.start_rendering()
            self.client.send(MessageType.SERVER_REQUEST_CURRENT_CAMERA)
            self._status = f"Connected to {self.client.host}:{self.client.port}"

        except Exception as exc:
            self._status = f"Connection failed: {exc}"
            logger.error("Connection failed: %s", exc)

    def _setup_rendering(self):
        """Hook for subclasses to register additional handlers.

        Called after base handlers are registered but before
        ``start_rendering``.  Override to add application-specific
        handlers, request initial state, etc.
        """

    def _on_receive_current_camera(self, _msg_type: int, payload: bytes):
        """Record the active camera object pool index from the server."""
        raw = _unpack_size_t_payload(payload)
        self._camera_pool_index = (
            int(raw) if is_valid_object_pool_index(raw) else None
        )
        self._on_camera_pool_index_changed()

    def _on_camera_pool_index_changed(self) -> None:
        """Hook after ``CLIENT_RECEIVE_CURRENT_CAMERA``; subclasses may push pose."""

    def _stop_local_playback(self) -> None:
        """Stop camera / scene-time threads without touching the socket."""
        try:
            self.pause_scene_time()
        except Exception:
            pass
        try:
            self.stop_animation()
        except Exception:
            pass

    def close(self):
        """Release widget comms and stop local animation threads.

        Does **not** stop server rendering or disconnect the TCP client, so
        other widgets (e.g. panels) sharing the same :class:`~tsd_client.client.TSDClient`
        keep working. Use :meth:`disconnect` for a full teardown.
        """
        self._stop_local_playback()
        super().close()

    def disconnect(self):
        """Stop playback, tell the server to pause rendering, and close TCP."""
        self._stop_local_playback()
        try:
            self.client.stop_rendering()
        except Exception:
            pass
        self.client.disconnect()
        self._status = "Disconnected"

    def reconnect(self):
        """Disconnect, reconnect the client, and restart rendering."""
        self.disconnect()
        self.client.connect()
        self._setup_and_start()

    # ========================================================================
    # Frame handler
    # ========================================================================

    def _on_frame(self, _msg_type: int, payload: bytes):
        if not payload:
            return
        buf = bytes(payload)
        fw = self._frame_buf_w or self._width
        fh = self._frame_buf_h or self._height

        def apply():
            jpeg = _normalize_frame_to_jpeg(buf, fw, fh)
            if not jpeg:
                return
            self._frame_count += 1
            self._frame_jpeg = jpeg
            tags = self._build_status_tags()
            self._status = (
                f"Connected | Frame #{self._frame_count}"
                f" | {self._width}\u00d7{self._height}"
                f"{tags}"
            )

        if _run_on_kernel_loop(apply):
            return
        apply()

    def _build_status_tags(self) -> str:
        """Override to append extra info to the status bar."""
        tags = ""
        if self._anim_playing:
            tags += f" | \u25B6 {self._anim_mode} {self._anim_speed:.0f}\u00b0/s"
        if self._scene_time_playing:
            tags += f" | \u23F5 t={self._scene_time_value:.3f}"
        return tags

    def _on_frame_config(self, _msg_type: int, payload: bytes):
        if len(payload) >= 8:
            w, h = struct.unpack("<2I", payload[:8])
            self._frame_buf_w = int(w)
            self._frame_buf_h = int(h)
            logger.info("Server frame config: %dx%d", w, h)

    def _handle_server_disconnect(self):
        def apply():
            self.pause_scene_time()
            self.stop_animation()
            self._status = "Server disconnected"

        if _run_on_kernel_loop(apply):
            return
        apply()

    # ========================================================================
    # Camera (orbit-style)
    # ========================================================================

    @property
    def camera(self) -> dict:
        """Current orbit camera state as a dict."""
        return {
            "azimuth": self._azimuth,
            "elevation": self._elevation,
            "distance": self._distance,
            "lookat": list(self._lookat),
        }

    @camera.setter
    def camera(self, value: dict):
        if "azimuth" in value:
            self._azimuth = float(value["azimuth"])
        if "elevation" in value:
            self._elevation = float(value["elevation"])
        if "distance" in value:
            self._distance = float(value["distance"])
        if "lookat" in value:
            self._lookat = [float(v) for v in value["lookat"]]
        self._send_view()

    def _send_view(self):
        """Send current camera state to the server.

        **Must be overridden by subclasses.**  The orbit state is
        available via ``self._azimuth``, ``self._elevation``,
        ``self._distance``, and ``self._lookat``.
        """
        raise NotImplementedError(
            "Subclass must implement _send_view() to send camera state to the server"
        )

    # ========================================================================
    # Camera animation API
    # ========================================================================

    def turntable(self, speed: float = 30.0):
        """Start a continuous turntable orbit animation.

        Parameters
        ----------
        speed : float
            Rotation speed in degrees per second.
        """
        self._start_animation("turntable", speed=speed)

    def rock(self, speed: float = 20.0, amplitude: float = 60.0):
        """Start a rocking (oscillating) animation.

        Parameters
        ----------
        speed : float
            Oscillation speed in degrees per second.
        amplitude : float
            Half-range of the oscillation in degrees.
        """
        self._anim_rock_amplitude = amplitude
        self._anim_rock_origin = self._azimuth
        self._start_animation("rock", speed=speed)

    def animate_to(self, target: dict, duration: float = 2.0):
        """Smoothly animate the camera to a target state.

        Parameters
        ----------
        target : dict
            Target camera state (azimuth, elevation, distance, lookat).
        duration : float
            Transition duration in seconds.
        """
        full_target = self.camera
        full_target.update({k: v for k, v in target.items() if k in full_target})
        self._start_animation("transition", target=full_target, duration=duration)

    def record_keyframe(self):
        """Record the current camera pose as a keyframe."""
        kf = self.camera
        self._keyframes.append(kf)
        n = len(self._keyframes)
        logger.info("Recorded keyframe #%d: %s", n, kf)
        self._status_suffix(f"Keyframe #{n} recorded")

    def clear_keyframes(self):
        """Remove all recorded keyframes."""
        self._keyframes.clear()
        self._status_suffix("Keyframes cleared")

    def play_keyframes(
        self, duration: float = 10.0, loop: bool = True, smooth: bool = True
    ):
        """Play back recorded keyframes as a camera path.

        Parameters
        ----------
        duration : float
            Total playback duration in seconds.
        loop : bool
            Whether to loop continuously.
        smooth : bool
            Use smooth (Catmull-Rom) interpolation between keyframes.
        """
        if len(self._keyframes) < 2:
            self._status_suffix(
                "Need at least 2 keyframes (have %d)" % len(self._keyframes)
            )
            return
        self._start_animation(
            "keyframes",
            keyframes=list(self._keyframes),
            duration=duration,
            loop=loop,
            smooth=smooth,
        )

    @property
    def keyframes(self) -> list[dict]:
        """List of recorded camera keyframes."""
        return list(self._keyframes)

    def stop_animation(self):
        """Stop any running camera animation."""
        self._anim_stop.set()
        if self._anim_thread and self._anim_thread.is_alive():
            self._anim_thread.join(timeout=2.0)
        self._anim_thread = None
        self._anim_playing = False

    # -- Camera animation internals ----------------------------------------

    def _start_animation(self, mode: str, **params: Any):
        self.stop_animation()
        self._anim_mode = mode
        self._anim_speed = params.get("speed", self._anim_speed)
        self._anim_stop.clear()
        self._anim_playing = True
        self._anim_thread = threading.Thread(
            target=self._animation_loop,
            kwargs=params,
            daemon=True,
            name="tsd-anim",
        )
        self._anim_thread.start()

    def _animation_loop(self, **params: Any):
        fps = 30
        interval = 1.0 / fps
        last = time.monotonic()

        transition_start = self.camera if self._anim_mode == "transition" else None
        transition_target = params.get("target")
        transition_duration = params.get("duration", 2.0)
        transition_t0 = time.monotonic()

        keyframes = params.get("keyframes", [])
        kf_duration = params.get("duration", 10.0)
        kf_loop = params.get("loop", True)
        kf_smooth = params.get("smooth", True)
        kf_t0 = time.monotonic()

        rock_t0 = time.monotonic()

        while not self._anim_stop.is_set():
            now = time.monotonic()
            dt = now - last
            last = now

            if self._anim_mode == "turntable":
                self._azimuth += self._anim_speed * dt
                self._send_view()

            elif self._anim_mode == "rock":
                elapsed = now - rock_t0
                phase = math.sin(
                    elapsed * self._anim_speed * math.pi / 180.0
                )
                self._azimuth = (
                    self._anim_rock_origin + phase * self._anim_rock_amplitude
                )
                self._send_view()

            elif self._anim_mode == "transition":
                t = (now - transition_t0) / max(transition_duration, 0.001)
                if t >= 1.0:
                    self._apply_camera(transition_target)
                    self._send_view()
                    break
                cam = _lerp_camera(
                    transition_start, transition_target, _smoothstep(t)
                )
                self._apply_camera(cam)
                self._send_view()

            elif self._anim_mode == "keyframes":
                elapsed = now - kf_t0
                n = len(keyframes)
                total = kf_duration
                if kf_loop:
                    elapsed = elapsed % total
                elif elapsed >= total:
                    self._apply_camera(keyframes[-1])
                    self._send_view()
                    break

                t_norm = elapsed / total
                seg_float = t_norm * (n - 1)
                seg = int(seg_float)
                seg = min(seg, n - 2)
                seg_t = seg_float - seg

                if kf_smooth:
                    cam = self._catmull_rom_interp(keyframes, seg, _smoothstep(seg_t))
                else:
                    cam = _lerp_camera(keyframes[seg], keyframes[seg + 1], seg_t)

                self._apply_camera(cam)
                self._send_view()

            sleep_time = interval - (time.monotonic() - now)
            if sleep_time > 0:
                self._anim_stop.wait(timeout=sleep_time)

        self._anim_playing = False

    @staticmethod
    def _catmull_rom_interp(keyframes: list[dict], seg: int, t: float) -> dict:
        """Catmull-Rom spline interpolation for smooth keyframe paths."""
        n = len(keyframes)
        p0 = keyframes[max(seg - 1, 0)]
        p1 = keyframes[seg]
        p2 = keyframes[min(seg + 1, n - 1)]
        p3 = keyframes[min(seg + 2, n - 1)]

        def cr(a, b, c, d, t):
            return b + 0.5 * t * (
                (c - a) + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a))
            )

        return {
            "azimuth": cr(p0["azimuth"], p1["azimuth"], p2["azimuth"], p3["azimuth"], t),
            "elevation": cr(p0["elevation"], p1["elevation"], p2["elevation"], p3["elevation"], t),
            "distance": cr(p0["distance"], p1["distance"], p2["distance"], p3["distance"], t),
            "lookat": [
                cr(p0["lookat"][i], p1["lookat"][i], p2["lookat"][i], p3["lookat"][i], t)
                for i in range(3)
            ],
        }

    def _apply_camera(self, cam: dict):
        """Apply a camera dict to internal state without sending to server."""
        self._azimuth = cam["azimuth"]
        self._elevation = cam["elevation"]
        self._distance = cam["distance"]
        self._lookat = list(cam["lookat"])

    # -- JS toolbar command handler ----------------------------------------

    def _handle_anim_cmd(self, change):
        cmd = change.get("new", {})
        if not cmd:
            return
        action = cmd.get("cmd")

        if action == "play":
            mode = cmd.get("mode", "turntable")
            speed = cmd.get("speed", 30.0)
            if mode == "turntable":
                self.turntable(speed=speed)
            elif mode == "rock":
                self.rock(speed=speed, amplitude=self._anim_rock_amplitude)
        elif action == "stop":
            self.stop_animation()
        elif action == "speed":
            self._anim_speed = cmd.get("speed", self._anim_speed)
        elif action == "record":
            self.record_keyframe()

    # -- Viewport resize ---------------------------------------------------

    def _handle_resize_event(self, change):
        event = change.get("new", {})
        if not event:
            return
        new_w = int(event.get("width", self._width))
        new_h = int(event.get("height", self._height))
        if new_w == self._width and new_h == self._height:
            return
        self._prev_width = self._width
        self._prev_height = self._height
        self._width = new_w
        self._height = new_h
        if self.client.connected:
            self.client.send_frame_config(new_w, new_h)
        logger.info("Viewport resized to %dx%d", new_w, new_h)

    # -- Mouse handling ----------------------------------------------------

    def _apply_mouse_event_dict(self, event: dict) -> None:
        """Apply one browser mouse/wheel event to orbit state and notify server."""
        if not event or not self.client.connected:
            return

        if self._anim_playing:
            self.stop_animation()

        action = event.get("action", "none")
        try:
            dx = float(event.get("dx", 0.0))
            dy = float(event.get("dy", 0.0))
        except (TypeError, ValueError):
            dx, dy = 0.0, 0.0
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return

        updated = False
        if action == "orbit":
            self._azimuth += dx * 180.0
            self._elevation += dy * 180.0
            self._elevation = max(-89.0, min(89.0, self._elevation))
            updated = True
        elif action == "dolly":
            self._distance *= 1.0 + dy * 2.0
            self._distance = max(0.001, self._distance)
            updated = True
        elif action == "pan":
            self._do_pan(dx, dy)
            updated = True

        if updated:
            try:
                self._send_view()
            except NotImplementedError:
                pass

    def _handle_mouse_event(self, change):
        # Read the live trait; ``change.new`` is occasionally stale with comms.
        event = self._mouse_event if isinstance(self._mouse_event, dict) else {}
        if not event:
            event = change.get("new") or {}
        if not isinstance(event, dict):
            return

        snap = dict(event)

        def apply():
            if not self.client.connected:
                return
            self._apply_mouse_event_dict(snap)

        if _run_on_kernel_loop(apply):
            return
        apply()

    def _do_pan(self, dx: float, dy: float):
        """Translate the lookat point in screen-space directions."""
        right, up = _tsd_manipulator_right_up(self._azimuth, self._elevation)
        scale = self._distance * 0.5
        # Same linear combination as ``Manipulator::pan``: ``-dx * right + dy * up``.
        for i in range(3):
            self._lookat[i] += (-right[i] * dx + up[i] * dy) * scale

    def _status_suffix(self, msg: str):
        """Flash a message on the status bar."""
        base = f"Connected to {self.client.host}:{self.client.port}"
        self._status = f"{base} | {msg}"

    # ========================================================================
    # Scene-time animation API
    # ========================================================================

    @property
    def scene_time(self) -> float:
        """Current normalised scene time [0..1]."""
        return self._scene_time_value

    @scene_time.setter
    def scene_time(self, value: float):
        self.set_scene_time(value)

    def set_scene_time(self, t: float):
        """Set the scene animation time and send to the server.

        Parameters
        ----------
        t : float
            Normalised time in [0, 1].
        """
        t = max(0.0, min(1.0, float(t)))
        self._scene_time_value = t
        self._scene_time = t
        self._send_scene_time(t)

    def _send_scene_time(self, t: float):
        """Send scene time to the server.

        The default sends ``SERVER_UPDATE_TIME`` with a float32 payload.
        Override if a different format is needed.
        """
        import struct as _struct

        self.client.send(
            MessageType.SERVER_UPDATE_TIME,
            _struct.pack("<f", t),
        )

    def play_scene_time(self, speed: float | None = None):
        """Start scene-time playback (loops 0 -> 1 -> 0 ...).

        Parameters
        ----------
        speed : float or None
            Normalised-time units per second.
        """
        if speed is not None:
            self._scene_time_speed = float(speed)
        self.pause_scene_time()
        self._scene_time_stop.clear()
        self._scene_time_playing = True
        self._scene_time_thread = threading.Thread(
            target=self._scene_time_loop,
            daemon=True,
            name="tsd-scene-time",
        )
        self._scene_time_thread.start()

    def pause_scene_time(self):
        """Pause scene-time playback."""
        self._scene_time_stop.set()
        if self._scene_time_thread and self._scene_time_thread.is_alive():
            self._scene_time_thread.join(timeout=2.0)
        self._scene_time_thread = None
        self._scene_time_playing = False

    def step_scene_time(self, delta: int = 1):
        """Step scene time by *delta* steps.

        Parameters
        ----------
        delta : int
            Number of steps (positive = forward, negative = backward).
        """
        max_steps = self._time_info.get("maxSteps", 0)
        step_size = 1.0 / max_steps if max_steps > 1 else 0.01
        new_t = max(0.0, min(1.0, self._scene_time_value + delta * step_size))
        self.set_scene_time(new_t)

    # -- Scene-time internals ----------------------------------------------

    def _scene_time_loop(self):
        fps = 30
        interval = 1.0 / fps
        last = time.monotonic()

        while not self._scene_time_stop.is_set():
            now = time.monotonic()
            dt = now - last
            last = now

            self._scene_time_value += self._scene_time_speed * dt
            if self._scene_time_value > 1.0:
                self._scene_time_value %= 1.0
            elif self._scene_time_value < 0.0:
                self._scene_time_value = 0.0

            self._scene_time = self._scene_time_value
            self._send_scene_time(self._scene_time_value)

            sleep_time = interval - (time.monotonic() - now)
            if sleep_time > 0:
                self._scene_time_stop.wait(timeout=sleep_time)

        self._scene_time_playing = False

    def _handle_scene_time_cmd(self, change):
        cmd = change.get("new", {})
        if not cmd:
            return
        action = cmd.get("cmd")

        if action == "play":
            self.play_scene_time(speed=cmd.get("speed", self._scene_time_speed))
        elif action == "pause":
            self.pause_scene_time()
        elif action == "seek":
            self.set_scene_time(cmd.get("time", 0.0))
        elif action == "step":
            self.step_scene_time(delta=cmd.get("delta", 1))
        elif action == "speed":
            self._scene_time_speed = cmd.get("speed", self._scene_time_speed)

    # -- Lifecycle ---------------------------------------------------------

    def __del__(self):
        # Never call disconnect() here: a replaced notebook widget is GC'd while
        # a new viewer (and panels) may still share the same TSDClient. Sending
        # SERVER_STOP_RENDERING + closing the socket would freeze the server for
        # everyone. Local thread shutdown is enough for daemon workers.
        try:
            self._stop_local_playback()
        except Exception:
            pass


class OrbitTSDViewer(TSDViewer):
    """Viewer that maps mouse orbit controls to the server's ANARI camera.

    Each update sends a batched ``SERVER_SET_OBJECT_PARAMETER`` message
    (VisRTX ``ParameterChange``) with ``position``, ``direction``, and ``up``.
    The camera pool index is taken from ``CLIENT_RECEIVE_CURRENT_CAMERA`` when
    the server sends it; otherwise the first camera in the cached scene graph.
    If neither is available yet, pool index **0** is assumed (typical for a
    single-camera scene). A one-shot background :meth:`~TSDClient.refresh_scene`
    warms the cache for ``wire_size_t`` without blocking the widget constructor.
    """

    def _setup_rendering(self) -> None:
        super()._setup_rendering()

        def _warm_scene() -> None:
            time.sleep(0.2)
            if not self.client.connected:
                return
            try:
                self.client.refresh_scene()
                self.fit_to_scene()
            except Exception:
                logger.debug("OrbitTSDViewer scene warm-up failed", exc_info=True)

        threading.Thread(
            target=_warm_scene,
            daemon=True,
            name="tsd-orbit-scene-warm",
        ).start()

    def fit_to_scene(self) -> None:
        """Set orbit camera parameters from the scene bounding box.

        Mirrors the server's ``resetView()`` logic: the lookat is placed
        at the world center and the distance is ``1.25 * diagonal``.
        """
        sg = self.client.cached_scene_graph()
        if sg is None:
            self._camera_fitted = True
            return
        bounds = sg.compute_bounds()
        if bounds is not None:
            lo, hi = bounds
            cx = 0.5 * (lo[0] + hi[0])
            cy = 0.5 * (lo[1] + hi[1])
            cz = 0.5 * (lo[2] + hi[2])
            dx = hi[0] - lo[0]
            dy = hi[1] - lo[1]
            dz = hi[2] - lo[2]
            diag = math.sqrt(dx * dx + dy * dy + dz * dz)
            dist = max(1.25 * diag, 0.01)
            self._lookat = [cx, cy, cz]
            self._distance = dist
            self._azimuth = 0.0
            self._elevation = 20.0
            logger.info(
                "fit_to_scene: center=(%.2f,%.2f,%.2f) diag=%.2f dist=%.2f",
                cx, cy, cz, diag, dist,
            )
        self._camera_fitted = True
        if self.client.connected:
            self._send_view()

    def _on_camera_pool_index_changed(self) -> None:
        if self._camera_fitted and self.client.connected:
            try:
                self._send_view()
            except Exception:
                logger.debug("Initial camera push failed", exc_info=True)

    def _send_view(self) -> None:
        from .anari_types import ANARI_CAMERA

        if not self.client.connected:
            return

        wire = 8
        sg = self.client.cached_scene_graph()
        if sg is not None:
            try:
                wire = int(getattr(sg, "_wire_size_t", 8) or 8)
            except Exception:
                logger.debug("Could not read wire_size_t from cache", exc_info=True)

        idx = self._camera_pool_index
        if not is_valid_object_pool_index(idx):
            if sg is not None and sg.cameras:
                idx = sg.cameras[0].object_index
            else:
                # Single-camera servers (e.g. TCV) almost always use pool 0.
                # We avoid calling scene_graph here (that would pull the full
                # scene on every drag); warm-up + CLIENT_RECEIVE_CURRENT_CAMERA
                # populate the real index when available.
                idx = 0
                if not getattr(self, "_logged_fallback_cam0", False):
                    logger.debug(
                        "OrbitTSDViewer: using default camera pool index 0 until "
                        "scene cache or CLIENT_RECEIVE_CURRENT_CAMERA is available"
                    )
                    self._logged_fallback_cam0 = True

        pos, direc, up = orbit_camera_pose(
            self._azimuth, self._elevation, self._distance, self._lookat
        )
        payload = build_parameter_change_payload(
            ANARI_CAMERA,
            int(idx),
            [
                ("position", "float32_vec3", pos),
                ("direction", "float32_vec3", direc),
                ("up", "float32_vec3", up),
            ],
            wire_size_t=wire,
        )
        self.client.send(MessageType.SERVER_SET_OBJECT_PARAMETER, payload)
        # Re-affirm current camera so the render pass stays bound (matches stock client).
        if wire == 4:
            cam_pl = struct.pack("<I", int(idx))
        else:
            cam_pl = struct.pack("<Q", int(idx))
        self.client.send(MessageType.SERVER_SET_CURRENT_CAMERA, cam_pl)


# ---------------------------------------------------------------------------
# Camera path helpers (position, direction, up)
# ---------------------------------------------------------------------------

CameraPose = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def interpolate_camera_path(
    control_points: list[CameraPose],
    num_steps: int = 300,
    smooth: bool = True,
) -> list[CameraPose]:
    """Build a smooth camera path from keyframes (position, direction, up).

    Parameters
    ----------
    control_points : list of (position, direction, up)
        At least 2 points required.
    num_steps : int
        Total number of interpolated poses along the path.
    smooth : bool
        If True, apply smooth-step (ease in/out) per segment.

    Returns
    -------
    list of (position, direction, up)
        Interpolated camera poses.
    """
    from .utils import normalize3 as _normalize3, slerp3_exact as _slerp3_exact

    if len(control_points) < 2:
        if not control_points:
            return []
        return [control_points[0][:3]]

    n_segments = len(control_points) - 1
    out: list[CameraPose] = []
    for step in range(num_steps):
        global_t = step / (num_steps - 1) if num_steps > 1 else 1.0
        if smooth:
            global_t = global_t * global_t * (3.0 - 2.0 * global_t)
        seg_float = global_t * n_segments
        seg = min(int(seg_float), n_segments - 1)
        local_t = seg_float - seg
        pos0, dir0, up0 = control_points[seg][:3]
        pos1, dir1, up1 = control_points[seg + 1][:3]
        r0 = math.sqrt(pos0[0] ** 2 + pos0[1] ** 2 + pos0[2] ** 2)
        r1 = math.sqrt(pos1[0] ** 2 + pos1[1] ** 2 + pos1[2] ** 2)
        r = r0 + local_t * (r1 - r0)
        pos_dir = _slerp3_exact(local_t, pos0, pos1)
        pos = (pos_dir[0] * r, pos_dir[1] * r, pos_dir[2] * r)
        direction = _slerp3_exact(local_t, dir0, dir1)
        up = _slerp3_exact(local_t, up0, up1)
        out.append((pos, direction, up))
    return out


def animate_camera_path(
    client: TSDClient,
    control_points: list[CameraPose],
    duration_sec: float | None = None,
    num_steps: int = 300,
    fps: float = 30.0,
    set_camera_fn=None,
) -> None:
    """Animate the camera along a path defined by keyframes.

    Parameters
    ----------
    client : TSDClient
        Connected client instance.
    control_points : list of (position, direction, up)
    duration_sec : float, optional
        Total duration in seconds.
    num_steps : int
        Number of interpolated frames.
    fps : float
        Frames per second (used when duration_sec is not set).
    set_camera_fn : callable, optional
        ``fn(position, direction, up)`` to apply each pose.  If None,
        calls ``client.set_camera_pose(...)`` (the downstream client
        must implement this method).
    """
    path = interpolate_camera_path(control_points, num_steps=num_steps, smooth=True)
    if not path:
        return
    if duration_sec is not None:
        step_dt = duration_sec / len(path)
    else:
        step_dt = 1.0 / fps

    if set_camera_fn is None:
        set_camera_fn = getattr(client, "set_camera_pose", None)
        if set_camera_fn is None:
            raise AttributeError(
                "Client has no set_camera_pose() method; pass set_camera_fn explicitly"
            )

    for position, direction, up in path:
        set_camera_fn(position, direction, up)
        time.sleep(step_dt)
