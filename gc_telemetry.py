# -*- coding: utf-8 -*-
"""Потік телеметрії: сховище LiveStore (вбудований симулятор або MQTT-підписка). Без Streamlit."""
import json
import threading
import time
from collections import deque
from datetime import datetime

import pandas as pd

from gc_simulator import Simulator

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except Exception:                                   # noqa: BLE001
    HAS_MQTT = False

# ════════════════════════════════════════════════════════════════════════════
# 7d. ПОТІК ТЕЛЕМЕТРІЇ: вбудований генератор або MQTT (telemetry_generator.py)
# ════════════════════════════════════════════════════════════════════════════
_REGISTRY = {}


def _store_registry():
    return _REGISTRY


class LiveStore:
    """Кільцеві буфери телеметрії по активах. Джерело: потік-симулятор (builtin) або MQTT-підписка."""

    def __init__(self, mode, host, port, prefix, user, pwd, interval, vstep, ambient, maxlen=300):
        self.key = (mode, host, int(port), prefix, user, pwd, float(interval), float(vstep))
        self.mode, self.host, self.port = mode, host, int(port)
        self.prefix = prefix.strip("/")
        self.user, self.pwd, self.interval, self.vstep = user, pwd, float(interval), float(vstep)
        self.ambient, self.maxlen = float(ambient), maxlen
        self.buf, self.lock, self.stop_ev = {}, threading.Lock(), threading.Event()
        self.arrivals = deque(maxlen=3000)
        self.n_msgs, self.error, self.connected = 0, "", False
        self.client = self.sim = self.thread = None

    # ---- запуск / зупинка ----
    def start(self):
        if self.mode == "mqtt":
            self._start_mqtt()
        else:
            self._start_builtin()

    def stop(self):
        self.stop_ev.set()
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:                        # noqa: BLE001
                pass
        self.connected = False

    def _start_builtin(self):
        self.sim = Simulator(self.ambient, 1.2, None, float(datetime.now().hour), self.vstep)
        for _ in range(40):                          # прогрів теплової моделі
            self.sim.tick()
        self.sim.vt = datetime.now().hour * 60.0
        self.connected = True

        def loop():
            while not self.stop_ev.is_set():
                self.sim.ambient = self.ambient
                for m in self.sim.tick():
                    self.ingest(m)
                self.stop_ev.wait(self.interval)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def _start_mqtt(self):
        try:
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:                       # paho-mqtt 1.x
            c = mqtt.Client()
        if self.user:
            c.username_pw_set(self.user, self.pwd)
        topic = f"{self.prefix}/telemetry/#"

        def on_connect(cl, ud, flags, rc, *a):
            fail = getattr(rc, "is_failure", None)
            ok = (not fail) if fail is not None else (rc == 0)
            self.connected = bool(ok)
            if ok:
                cl.subscribe(topic, qos=0)
                self.error = ""
            else:
                self.error = f"MQTT: брокер відхилив підключення ({rc})"

        def on_disconnect(cl, ud, *a):
            self.connected = False

        def on_message(cl, ud, msg):
            try:
                self.ingest(json.loads(msg.payload.decode("utf-8")))
            except Exception:                        # noqa: BLE001
                pass
        c.on_connect, c.on_disconnect, c.on_message = on_connect, on_disconnect, on_message
        c.reconnect_delay_set(1, 10)
        try:
            c.connect_async(self.host, self.port, keepalive=30)
            c.loop_start()
        except Exception as ex:                      # noqa: BLE001
            self.error = f"MQTT: {type(ex).__name__}: {ex}"
        self.client = c

    # ---- дані ----
    def ingest(self, m):
        a = m.get("asset")
        if not a:
            return
        m = dict(m)
        m["rx"] = time.time()
        with self.lock:
            self.buf.setdefault(a, deque(maxlen=self.maxlen)).append(m)
            self.n_msgs += 1
            self.arrivals.append(m["rx"])

    def latest(self):
        with self.lock:
            rows = [dict(b[-1]) for b in self.buf.values() if b]
        return pd.DataFrame(rows)

    def frame(self, assets):
        with self.lock:
            rows = [m for a in assets for m in self.buf.get(a, [])]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["t"] = pd.to_datetime(df["rx"], unit="s")
        return df

    def status(self):
        now = time.time()
        with self.lock:
            rate = sum(1 for t in self.arrivals if now - t <= 10.0) / 10.0
            n = self.n_msgs
        return dict(connected=self.connected, n=n, rate=rate)

    def summary(self):
        """Узагальнення для моделі відмов: {актив: peak, mean, frac_over, n} за вікном буфера."""
        out = {}
        with self.lock:
            for a, b in self.buf.items():
                v = [float(m.get("loading", 0.0)) for m in b]
                if v:
                    out[a] = dict(peak=max(v), mean=sum(v) / len(v),
                                  frac_over=sum(1 for x in v if x > 1.0) / len(v), n=len(v))
        return out

    # ---- керування аномаліями ----
    def inject(self, asset, kind, ticks=12):
        if self.mode == "builtin" and self.sim is not None:
            self.sim.events[asset] = (kind, int(ticks))
        elif self.client is not None:
            self.client.publish(f"{self.prefix}/control", json.dumps(
                dict(cmd="inject", asset=asset, kind=kind, ticks=int(ticks)), ensure_ascii=False))

    def clear(self):
        if self.mode == "builtin" and self.sim is not None:
            self.sim.events.clear()
        elif self.client is not None:
            self.client.publish(f"{self.prefix}/control", json.dumps(dict(cmd="clear")))


def get_store(mode, host, port, prefix, user, pwd, interval, vstep, ambient):
    reg = _store_registry()
    key = (mode, host, int(port), prefix.strip("/"), user, pwd, float(interval), float(vstep))
    cur = reg.get("store")
    if cur is not None and cur.key == key:
        return cur
    if cur is not None:
        cur.stop()
    cur = LiveStore(mode, host, port, prefix, user, pwd, interval, vstep, ambient)
    cur.start()
    reg["store"] = cur
    return cur


def stop_store():
    cur = _store_registry().pop("store", None)
    if cur is not None:
        cur.stop()


def classify_state(loading, oil_c, winding_c):
    """Стан активу за телеметрією: 🔴 тривога / 🟡 увага / 🟢 норма."""
    if loading > 1.0 or oil_c > 95 or winding_c > 110:
        return "🔴 Тривога"
    if loading > 0.9 or oil_c > 85:
        return "🟡 Увага"
    return "🟢 Норма"


def current_store():
    """Поточне активне сховище потоку (або None)."""
    return _store_registry().get("store")
