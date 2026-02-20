# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
TSD Jupyter — base Jupyter notebook widget for interactive remote rendering
via a TSD server.

Provides canvas display, camera control (orbit/dolly/pan), camera animations,
and a detachable popup viewer. Application-specific features (scene-time
animation, denoiser control, etc.) should be added by subclasses.

Usage::

    from tsdjupyter import TSDJupyter
    viewer = TSDJupyter("hostname", 12345)
    viewer          # displays the interactive viewer in the notebook

Mouse controls:
    Left drag       → Orbit (azimuth / elevation)
    Right drag      → Dolly (distance)
    Shift+Left drag → Dolly
    Middle drag     → Pan (lookat)
    Alt+Left drag   → Pan
    Scroll wheel    → Dolly

Camera animation API::

    viewer.turntable(speed=30)
    viewer.rock(speed=20, amplitude=60)
    viewer.animate_to({'azimuth': 90, 'elevation': 10}, duration=2.0)
    viewer.record_keyframe()
    viewer.play_keyframes(duration=10)
    viewer.stop_animation()
"""

import asyncio
import io
import math
import struct
import time
import threading
import logging
from typing import Any

import anywidget
import traitlets
from PIL import Image

from .tsd_client import (
    TSDClient,
    MessageType,
    VIEW_FORMAT,
)

logger = logging.getLogger("tsdjupyter.viewer")

# ---------------------------------------------------------------------------
# JavaScript front-end  (embedded ESM for anywidget)
# ---------------------------------------------------------------------------

_WIDGET_ESM = """
export function render({ model, el }) {
  const width  = model.get('_width');
  const height = model.get('_height');
  let activeWidth = width;
  let activeHeight = height;

  // -- Styles -------------------------------------------------------------

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

  // -- DOM setup ----------------------------------------------------------

  const wrapper = document.createElement('div');
  wrapper.style.display = 'inline-block';
  wrapper.style.fontFamily = "'SF Mono', 'Fira Code', 'Consolas', monospace";

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

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#666';
  ctx.font = '14px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Connecting...', width / 2, height / 2);

  // -- Toolbar ------------------------------------------------------------

  const toolbar = document.createElement('div');
  toolbar.style.cssText = `
    display: flex; align-items: center; gap: 6px;
    padding: 6px 8px;
    background: ${COLORS.toolbar};
    border: 1px solid ${COLORS.border}; border-top: none;
    border-radius: 0 0 6px 6px;
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

  // -- Status bar ---------------------------------------------------------

  const status = document.createElement('div');
  status.style.cssText = `
    font-size: 11px; color: ${COLORS.textDim};
    padding: 3px 8px; background: ${COLORS.toolbar};
    border: 1px solid ${COLORS.border}; border-top: none;
    border-radius: 0 0 6px 6px;
  `;
  status.textContent = model.get('_status');

  toolbar.style.borderRadius = '0';

  wrapper.append(canvas, toolbar, status);
  el.appendChild(wrapper);

  // -- Toolbar logic ------------------------------------------------------

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

  // -- Detach state -------------------------------------------------------

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

  // -- Frame display ------------------------------------------------------

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

  // -- Mouse interaction --------------------------------------------------

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
      const dx = (e.offsetX - lastX) / activeWidth;
      const dy = (e.offsetY - lastY) / activeHeight;
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

  // -- Detach / Reattach --------------------------------------------------

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
    popupWin = window.open('', 'TSD_Jupyter_' + Date.now(),
      'width=' + popW + ',height=' + popH + ',resizable=yes,scrollbars=no');
    if (!popupWin) {
      alert('Popup blocked \\u2014 please allow popups for this site.');
      return;
    }

    const doc = popupWin.document;
    doc.title = 'TSD Jupyter';
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


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(t: float) -> float:
    """Hermite smoothstep for ease-in-out motion."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp_camera(c0: dict, c1: dict, t: float) -> dict:
    """Linearly interpolate between two camera dicts."""
    return {
        "azimuth": _lerp(c0["azimuth"], c1["azimuth"], t),
        "elevation": _lerp(c0["elevation"], c1["elevation"], t),
        "distance": _lerp(c0["distance"], c1["distance"], t),
        "lookat": [_lerp(a, b, t) for a, b in zip(c0["lookat"], c1["lookat"])],
    }


# ---------------------------------------------------------------------------
# TSDJupyter widget
# ---------------------------------------------------------------------------


class TSDJupyter(anywidget.AnyWidget):
    """Base interactive Jupyter widget that connects to a TSD server and
    displays streamed rendering with orbit/dolly/pan mouse controls and
    built-in camera animation support.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port (default 12345).
    width, height : int
        Viewport resolution in pixels.
    jpeg_quality : int
        JPEG encoding quality for frame transfer (1-100).
    detachable : bool
        If True (default), show a button to pop the viewer out into a
        separate browser window for multi-monitor workflows.
    client : TSDClient or None
        Optional pre-created client instance. If None, a new TSDClient is
        created. Subclasses can pass specialised client subclasses here.
    auto_connect : bool
        If True (default), connect to the server immediately.
    """

    _esm = traitlets.Unicode(_WIDGET_ESM).tag(sync=True)

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

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12345,
        width: int = 800,
        height: int = 600,
        jpeg_quality: int = 85,
        detachable: bool = True,
        client: TSDClient | None = None,
        auto_connect: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._width = width
        self._height = height
        self._host = host
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._detachable = detachable

        # Camera state
        self._azimuth = 0.0
        self._elevation = 20.0
        self._distance = 5.0
        self._lookat = [0.0, 0.0, 0.0]
        self._view_initialized = False

        # TSD client
        self._client = client or TSDClient()
        self._frame_count = 0
        self._prev_width = width
        self._prev_height = height

        # Camera animation state
        self._anim_thread: threading.Thread | None = None
        self._anim_stop = threading.Event()
        self._anim_mode = "turntable"
        self._anim_speed = 30.0
        self._anim_rock_amplitude = 60.0
        self._anim_rock_origin = 0.0
        self._keyframes: list[dict] = []

        # React to events from JS
        self.observe(self._handle_mouse_event, names=["_mouse_event"])
        self.observe(self._handle_anim_cmd, names=["_anim_cmd"])
        self.observe(self._handle_resize_event, names=["_resize_event"])

        if auto_connect:
            self.connect()

    # -- Connection ----------------------------------------------------------

    def connect(self):
        """Connect to the TSD server and start rendering."""
        try:
            self._status = f"Connecting to {self._host}:{self._port}..."

            self._client.on_disconnect = self._handle_server_disconnect
            self._client.register_handler(
                MessageType.CLIENT_RECEIVE_FRAME_BUFFER_COLOR, self._on_frame
            )
            self._client.register_handler(
                MessageType.CLIENT_RECEIVE_VIEW, self._on_view
            )
            self._client.register_handler(
                MessageType.CLIENT_RECEIVE_FRAME_CONFIG, self._on_frame_config
            )
            self._client.register_handler(
                MessageType.CLIENT_RECEIVE_SCENE,
                lambda _t, _p: logger.debug(
                    "Received scene (%d bytes)", len(_p)
                ),
            )
            self._client.register_handler(
                MessageType.CLIENT_SCENE_TRANSFER_BEGIN, lambda _t, _p: None
            )
            self._client.register_handler(
                MessageType.PING, lambda _t, _p: None
            )
            self._client.register_handler(
                MessageType.ERROR,
                lambda _t, p: logger.error(
                    "Server error: %s", p.decode("utf-8", errors="replace")
                ),
            )

            self._client.connect(self._host, self._port)
            self._client.request_view()
            self._client.send_frame_config(self._width, self._height)
            self._client.start_rendering()
            self._status = f"Connected to {self._host}:{self._port}"

        except Exception as exc:
            self._status = f"Connection failed: {exc}"
            logger.error("Connection failed: %s", exc)

    def disconnect(self):
        """Stop rendering and disconnect."""
        self.stop_animation()
        try:
            self._client.stop_rendering()
        except Exception:
            pass
        self._client.disconnect()
        self._status = "Disconnected"

    def reconnect(self):
        """Disconnect and reconnect."""
        self.disconnect()
        self.connect()

    # -- Camera API ----------------------------------------------------------

    @property
    def camera(self) -> dict:
        """Current camera state as a dict."""
        return {
            "azimuth": self._azimuth,
            "elevation": self._elevation,
            "distance": self._distance,
            "lookat": list(self._lookat),
        }

    @camera.setter
    def camera(self, value: dict):
        """Set camera state and send to server."""
        if "azimuth" in value:
            self._azimuth = float(value["azimuth"])
        if "elevation" in value:
            self._elevation = float(value["elevation"])
        if "distance" in value:
            self._distance = float(value["distance"])
        if "lookat" in value:
            self._lookat = [float(v) for v in value["lookat"]]
        self._send_view()

    # -- Scene request -------------------------------------------------------

    def request_scene(self, timeout: float = 30.0) -> bytes:
        """Request the current scene from the server."""
        if not self._client.connected:
            return b""
        return self._client.request_scene(timeout=timeout)

    # -- Volume management ---------------------------------------------------

    def request_volume_list(self, timeout: float = 5.0) -> list:
        """Request the list of volumes (lightweight: index + name only).

        Returns list of ``{index, name}`` dicts.
        """
        if not self._client.connected:
            return []
        return self._client.request_volume_list(timeout=timeout)

    def request_volume_info(
        self, volume_index: int, timeout: float = 5.0
    ) -> dict:
        """Request full attributes for a single volume.

        Returns dict with all parameters, metadata, colors, and ``field``
        sub-object.
        """
        if not self._client.connected:
            return {}
        return self._client.request_volume_info(volume_index, timeout=timeout)

    def set_volume_attribute(
        self, volume_index: int, name: str, param_type: str, value
    ):
        """Set an attribute on a volume (or its spatial field).

        Parameters
        ----------
        volume_index : int
            Index of the volume.
        name : str
            Attribute name (e.g. ``"elevationScale"``).
        param_type : str
            One of ``"bool"``, ``"int32"``, ``"float32"``, ``"string"``.
        value
            Value matching the type.
        """
        self._client.set_volume_attribute(volume_index, name, param_type, value)

    # ========================================================================
    # Animation API
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
            Target camera state. Any of 'azimuth', 'elevation', 'distance',
            'lookat' may be provided; missing keys keep their current value.
        duration : float
            Transition duration in seconds.
        """
        full_target = self.camera
        full_target.update(
            {k: v for k, v in target.items() if k in full_target}
        )
        self._start_animation(
            "transition", target=full_target, duration=duration
        )

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

    # -- Camera animation internals ------------------------------------------

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
                    cam = self._catmull_rom_interp(
                        keyframes, seg, _smoothstep(seg_t)
                    )
                else:
                    cam = _lerp_camera(
                        keyframes[seg], keyframes[seg + 1], seg_t
                    )

                self._apply_camera(cam)
                self._send_view()

            sleep_time = interval - (time.monotonic() - now)
            if sleep_time > 0:
                self._anim_stop.wait(timeout=sleep_time)

        self._anim_playing = False

    @staticmethod
    def _catmull_rom_interp(
        keyframes: list[dict], seg: int, t: float
    ) -> dict:
        """Catmull-Rom spline interpolation for smooth keyframe paths."""
        n = len(keyframes)
        p0 = keyframes[max(seg - 1, 0)]
        p1 = keyframes[seg]
        p2 = keyframes[min(seg + 1, n - 1)]
        p3 = keyframes[min(seg + 2, n - 1)]

        def cr(a, b, c, d, t):
            return b + 0.5 * t * (
                (c - a)
                + t * (2.0 * a - 5.0 * b + 4.0 * c - d
                       + t * (3.0 * (b - c) + d - a))
            )

        return {
            "azimuth": cr(
                p0["azimuth"], p1["azimuth"], p2["azimuth"], p3["azimuth"], t
            ),
            "elevation": cr(
                p0["elevation"],
                p1["elevation"],
                p2["elevation"],
                p3["elevation"],
                t,
            ),
            "distance": cr(
                p0["distance"],
                p1["distance"],
                p2["distance"],
                p3["distance"],
                t,
            ),
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

    # -- JS toolbar command handler ------------------------------------------

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

    # -- Thread-safe trait helpers --------------------------------------------

    def _set_traits_threadsafe(self, **traits):
        """Set synced traitlets from any thread, marshalling onto the IO loop
        when called from a background thread (e.g. the recv thread)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _apply():
            for name, value in traits.items():
                setattr(self, name, value)

        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(_apply)
        else:
            _apply()

    # -- Message handlers (called from recv thread) --------------------------

    def _on_frame(self, _msg_type: int, payload: bytes):
        n = len(payload)
        if n == 0:
            return

        w, h = self._width, self._height
        if n != w * h * 4:
            pw, ph = self._prev_width, self._prev_height
            if n == pw * ph * 4:
                w, h = pw, ph
            else:
                logger.debug(
                    "Frame size mismatch: got %d bytes, expected %d (%dx%d). "
                    "Dropping transient frame.",
                    n, w * h * 4, w, h,
                )
                return

        self._frame_count += 1
        try:
            img = Image.frombytes("RGBA", (w, h), payload)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._jpeg_quality)
            jpeg_data = buf.getvalue()

            tags = self._build_status_tags()
            status = (
                f"Connected | Frame #{self._frame_count}"
                f" | {self._width}\u00d7{self._height}"
                f"{tags}"
            )
            self._set_traits_threadsafe(
                _frame_jpeg=jpeg_data, _status=status
            )
        except Exception as exc:
            logger.error("Frame encode error: %s", exc)

    def _build_status_tags(self) -> str:
        """Build extra status-bar tags. Override in subclasses to add more."""
        tags = ""
        if self._anim_playing:
            tags += f" | \u25B6 {self._anim_mode} {self._anim_speed:.0f}\u00b0/s"
        return tags

    def _on_view(self, _msg_type: int, payload: bytes):
        if len(payload) < 24:
            return
        az, el, dist, lx, ly, lz = struct.unpack(VIEW_FORMAT, payload[:24])
        self._azimuth = az
        self._elevation = el
        self._distance = dist
        self._lookat = [lx, ly, lz]
        self._view_initialized = True
        logger.info(
            "Server view: azel=(%.1f, %.1f) dist=%.1f lookat=(%.1f,%.1f,%.1f)",
            az, el, dist, lx, ly, lz,
        )

    def _on_frame_config(self, _msg_type: int, payload: bytes):
        if len(payload) >= 8:
            w, h = struct.unpack("<2I", payload[:8])
            logger.info("Server frame config: %dx%d", w, h)

    def _handle_server_disconnect(self):
        self.stop_animation()
        self._set_traits_threadsafe(_status="Server disconnected")

    # -- Viewport resize -----------------------------------------------------

    def _handle_resize_event(self, change):
        """Handle viewport resize from the detached popup window."""
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
        if self._client.connected:
            self._client.send_frame_config(new_w, new_h)
        logger.info("Viewport resized to %dx%d", new_w, new_h)

    # -- Mouse handling (called from main thread via traitlet observe) -------

    def _handle_mouse_event(self, change):
        event = change.get("new", {})
        if not event or not self._client.connected:
            return

        if self._anim_playing:
            self.stop_animation()

        action = event.get("action", "none")
        dx = event.get("dx", 0.0)
        dy = event.get("dy", 0.0)

        if action == "orbit":
            self._azimuth += dx * 180.0
            self._elevation += dy * 180.0
            self._elevation = max(-89.0, min(89.0, self._elevation))
        elif action == "dolly":
            self._distance *= 1.0 + dy * 2.0
            self._distance = max(0.001, self._distance)
        elif action == "pan":
            self._do_pan(dx, dy)

        self._send_view()

    def _do_pan(self, dx: float, dy: float):
        """Translate the lookat point in screen-space directions."""
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)

        right = [math.cos(az), 0.0, -math.sin(az)]
        fwd = [
            -math.cos(el) * math.sin(az),
            -math.sin(el),
            -math.cos(el) * math.cos(az),
        ]
        up = [
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        ]

        scale = self._distance * 0.5
        for i in range(3):
            self._lookat[i] -= (right[i] * dx + up[i] * dy) * scale

    def _send_view(self):
        self._client.send_view(
            self._azimuth, self._elevation, self._distance, self._lookat
        )

    def _status_suffix(self, msg: str):
        """Flash a message on the status bar."""
        base = f"Connected to {self._host}:{self._port}"
        self._status = f"{base} | {msg}"

    # -- Lifecycle -----------------------------------------------------------

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
