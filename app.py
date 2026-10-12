# -*- coding: utf-8 -*-
"""
Цифровий двійник розподільчої мережі 10 кВ - Вінницька область (ВСЕ В ОДНОМУ ФАЙЛІ).

Умовна (синтетична) мережа: 2 ПС 110/10 кВ, 4 фідери, 20 ТП, перемички, PV та акумулятори, розміщена в обраному
районі Вінницької області. Це НЕ реальні об'єкти й не дані обленерго.

Що всередині (розділи файлу):
    модель мережі · симулятор телеметрії (+ CLI-генератор) · режими pandapower · КЗ і РЗА · фігури ·
    прогноз відмов + SHAP + ризик · ML-прогноз навантаження (Open-Meteo) · потік MQTT · PDF-звіт ·
    сервісний шар · FastAPI · Streamlit-інтерфейс

Запуск:
    streamlit run app.py                                   # інтерфейс
    uvicorn app:api --port 8000                            # API (потрібні fastapi, uvicorn); /docs
    python app.py --generator --host broker.hivemq.com --prefix vinnytsia-twin-<суфікс>   # MQTT-генератор
    python app.py --generator --dry-run --ticks 3          # перевірка генератора без брокера

Залежності - див. requirements.txt (pandas<3 обов'язково для pandapower < 3.2).
Якщо задано змінну GRID_API_KEY, API вимагає заголовок X-API-Key.
"""
import sys

import streamlit as st


try:
    import argparse
    import copy
    import functools
    import io
    import json
    import logging
    import math
    import matplotlib
    import networkx as nx
    import numpy as np
    import os
    import pandapower as pp
    import pandapower.shortcircuit as sc
    import pandas as pd
    import plotly.graph_objects as go
    import random
    import threading
    import time
    import urllib.parse
    import urllib.request
    import warnings
    from collections import defaultdict, deque, namedtuple
    from datetime import datetime, timedelta, timezone
    from functools import lru_cache
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from typing import List, Literal, Optional
    from zoneinfo import ZoneInfo
except ImportError as _err:                          # noqa: BLE001  (на Streamlit Cloud текст помилок приховується)
    st.set_page_config(page_title="Помилка завантаження", page_icon="⚠️", layout="wide")
    st.error(f"**Не встановлено бібліотеку:** `{_err}`. Перевірте requirements.txt і натисніть Manage app → Reboot app.")
    st.stop()

try:                                                 # fpdf2 - лише для PDF-звітів
    from fpdf import FPDF
    from fpdf.fonts import FontFace
    HAS_FPDF = True
except Exception:                                    # noqa: BLE001
    FPDF, FontFace, HAS_FPDF = object, None, False



# ════════════════════════════════════════════════════════════════════════════
# Допоміжні кеші
# ════════════════════════════════════════════════════════════════════════════
def ttl_cache(ttl, maxsize=32):
    """Кеш із часом життя ttl (с). Аргументи мають бути хешованими; винятки не кешуються."""
    def deco(fn):
        store, lock = {}, threading.Lock()

        @functools.wraps(fn)
        def wrapper(*a, **k):
            key = (a, tuple(sorted(k.items())))
            now = time.time()
            with lock:
                hit = store.get(key)
                if hit is not None and now - hit[0] < ttl:
                    return hit[1]
            val = fn(*a, **k)
            with lock:
                store[key] = (now, val)
                while len(store) > maxsize:
                    store.pop(next(iter(store)))
            return val
        wrapper.cache_clear = store.clear
        return wrapper
    return deco


def df_key(df):
    """Хешований ключ вмісту DataFrame (для кешів моделей)."""
    return (df.shape, int(pd.util.hash_pandas_object(df, index=True).sum()) % (2 ** 63))


# ════════════════════════════════════════════════════════════════════════════
# Модель мережі та профілі
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 1. МОДЕЛЬ МЕРЕЖІ ТА ПРОФІЛІ
# ════════════════════════════════════════════════════════════════════════════
HOURLY = [0.52, 0.48, 0.46, 0.45, 0.46, 0.52, 0.65, 0.80, 0.85, 0.80, 0.76, 0.75,
          0.74, 0.73, 0.74, 0.78, 0.90, 0.98, 1.00, 0.97, 0.92, 0.85, 0.72, 0.60]
PRIO_W = {1: 10.0, 2: 3.0, 3: 1.0}
PRIO_TXT = {1: "1 (критичний)", 2: "2 (звичайний)", 3: "3 (обмежуваний)"}
SOC_FLOOR, BESS_EFF = 0.10, 0.92

Params = namedtuple(
    "Params",
    "temp hour sens src_cap tol t_detect t_switch repair_h vmin vmax u_ps cosphi "
    "der pv_k ins bess_mode soc island prof pvp")

PV_DEF = {"ТП-102": 0.5, "ТП-105": 0.4, "ТП-203": 0.6, "ТП-302": 0.5, "ТП-404": 0.5, "ТП-405": 0.7}
BESS_DEF = {"ТП-104": (1.0, 2.0), "ТП-304": (1.2, 2.4), "ТП-404": (1.5, 3.0)}   # МВт, МВт·год


def build_grid():
    """Дві ПС 110/10 кВ, чотири фідери по 5 ТП, перемички; кабелі з r, x, довжинами; ВДЕ/АБ."""
    rng = np.random.default_rng(42)
    nodes, edges = {}, {}
    for s, x in (("ПС-1", -4.0), ("ПС-2", 4.0)):
        nodes[s] = dict(type="src", x=x, y=0.0, load=0.0, cons=0, prio=0, kva=0, feeder="-",
                        pv=0.0, bess_mw=0.0, bess_mwh=0.0)
    feeders = {
        "Ф-1": ("ПС-1", -1, 0.9, 0.28, 101),
        "Ф-2": ("ПС-1", -1, -0.9, -0.28, 201),
        "Ф-3": ("ПС-2", +1, 0.9, 0.28, 301),
        "Ф-4": ("ПС-2", +1, -0.9, -0.28, 401),
    }
    seg_caps = [8.0, 7.0, 6.0, 5.5, 5.0]            # МВт при cosφ=0.95 -> термічний ліміт
    seg_sect = [240, 185, 150, 120, 120]            # мм² (Al)
    std_kva = [250, 400, 630, 1000, 1250, 1600]

    def mk_edge(u, v, cap, sect, tie):
        return dict(u=u, v=v, cap=cap, tie=tie, len=round(float(rng.uniform(0.7, 1.9)), 2),
                    r=round(28.7 * 1.1 / sect, 4), x=0.09, c=280.0,
                    imax=cap / (math.sqrt(3) * 10 * 0.95))

    for fname, (src, side, y0, dy, base) in feeders.items():
        prev = src
        for i in range(5):
            name = f"ТП-{base + i}"
            load = round(float(rng.uniform(0.35, 0.85)), 2)
            cands = [k for k in std_kva if 0.35 <= load / (0.95 * k / 1000) <= 0.85]
            kva = int(rng.choice(cands)) if cands else 1000
            bm, bh = BESS_DEF.get(name, (0.0, 0.0))
            nodes[name] = dict(
                type="tp", x=nodes[src]["x"] - side * 0.75 * (i + 1), y=y0 + dy * i, load=load,
                cons=int(rng.integers(90, 450)),
                prio=int(rng.choice([1, 2, 3], p=[0.15, 0.55, 0.30])), kva=kva, feeder=fname,
                pv=PV_DEF.get(name, 0.0), bess_mw=bm, bess_mwh=bh)
            edges[f"{prev}–{name}"] = mk_edge(prev, name, seg_caps[i], seg_sect[i], False)
            prev = name
    for u, v in [("ТП-105", "ТП-305"), ("ТП-205", "ТП-405"), ("ТП-105", "ТП-205"),
                 ("ТП-305", "ТП-405"), ("ТП-103", "ТП-203"), ("ТП-303", "ТП-403")]:
        edges[f"{u}–{v}"] = mk_edge(u, v, 5.0, 120, True)
    return nodes, edges


NODES, EDGES = build_grid()
SOURCES = [n for n, d in NODES.items() if d["type"] == "src"]
TPS = [n for n, d in NODES.items() if d["type"] == "tp"]
CLOSED0 = frozenset(e for e, d in EDGES.items() if not d["tie"])
ALL_ELEMS = SOURCES + TPS + [e for e, d in EDGES.items() if not d["tie"]]
E0 = frozenset()

# ── Вінницька область: райони (центри умовної типової мережі) та географія ──
# Координати - наближені центри міст. Мережа в застосунку УМОВНА (синтетична), а не реальна схема обленерго.
DISTRICTS = {
    "Вінниця": dict(lat=49.2331, lon=28.4682),
    "Жмеринка": dict(lat=49.0354, lon=28.1147),
    "Хмільник": dict(lat=49.5560, lon=27.9710),
    "Козятин": dict(lat=49.7150, lon=28.8340),
    "Гайсин": dict(lat=48.8097, lon=29.3897),
    "Тульчин": dict(lat=48.6720, lon=28.8480),
    "Могилів-Подільський": dict(lat=48.4455, lon=27.7870),
    "Бершадь": dict(lat=48.3650, lon=29.5230),
}
OBLAST_CENTER = (49.0, 28.6)
GEO_KM = 1.4                                         # км на одну умовну одиницю схеми

# Типи споживачів (умовні) за категорією надійності: 1 - критичні, 2 - звичайні, 3 - обмежувані
CTYPES = {1: ["Лікарня", "Водозабір/КНС", "Котельня", "Пункт незламності"],
          2: ["Житловий сектор", "Школа/ліцей", "Адмінбудівля/ЦНАП"],
          3: ["Агропідприємство", "Переробне підприємство", "Торговий центр", "Склад/логістика"]}
for _n in TPS:                                       # окремий генератор: не зміщує основний rng мережі
    _r = np.random.default_rng(1000 + int(_n.split("-")[1]))
    NODES[_n]["ctype"] = str(_r.choice(CTYPES[NODES[_n]["prio"]]))
for _n in SOURCES:
    NODES[_n]["ctype"] = "Підстанція 110/10 кВ"


def node_lonlat(n, center):
    lat0, lon0 = center
    d = NODES[n]
    return (lon0 + d["x"] * GEO_KM / (111.32 * math.cos(math.radians(lat0))),
            lat0 + d["y"] * GEO_KM / 110.57)


def elem_type(e):
    if e in NODES:
        return "ПС" if NODES[e]["type"] == "src" else "ТП"
    return "Лінія"


def fmt_elem(e):
    return {"ПС": "⚡ ", "ТП": "🏠 ", "Лінія": "〰️ Лінія "}[elem_type(e)] + e


def temp_factor(temp, sens):
    return 1.0 + sens / 100.0 * max(0.0, 5.0 - temp)


def load_scale(p):
    """Коефіцієнт навантаження: ML-профіль на завтра (p.prof, 24 значення) або статичний HOURLY × T."""
    if p.prof:
        return p.prof[int(p.hour)]
    return HOURLY[int(p.hour)] * temp_factor(p.temp, p.sens)


def pv_profile(h):
    """Зимова добова крива PV (частка від встановленої потужності, ясно)."""
    return max(0.0, math.sin(math.pi * (h - 7.5) / 9.5)) * 0.7 if 7.5 < h < 17 else 0.0


def pv_frac(p):
    """Частка встановленої потужності PV: з прогнозу радіації (p.pvp) або типова крива × інсоляція."""
    if p.pvp:
        return p.pvp[int(p.hour)]
    return pv_profile(p.hour) * p.ins / 100.0


def bess_frac(h, mode, soc):
    """Режим АБ у нормі: 0 - резерв; 1 - пікове голення (розряд 17-21 год, заряд 10-14 год)."""
    if mode == 0:
        return 0.0
    if 17 <= h <= 21 and soc > 0.2:
        return 0.8
    if 10 <= h <= 14 and soc < 0.95:
        return -0.5
    return 0.0


def make_params(temp=-10.0, hour=18, sens=1.2, src_cap=12.0, tol=0.0, t_detect=5.0, t_switch=4.0,
                repair_h=8.0, vmin=0.90, vmax=1.10, u_ps=10.5, cosphi=0.95, der=True, pv_k=1.0, ins=80.0,
                bess_mode=0, soc=0.80, island=True, prof=(), pvp=()):
    """Єдине місце побудови кортежу параметрів режиму P (хешований - придатний для кешів).

    tol і soc - частки (0.05 = 5 %); ins - відсотки; bess_mode: 0 - резерв, 1 - пікове голення;
    prof/pvp - 24 значення ML-профілю навантаження / частки PV (порожньо = статичні профілі)."""
    return (float(temp), int(hour), float(sens), float(src_cap), float(tol), float(t_detect), float(t_switch),
            float(repair_h), float(vmin), float(vmax), float(u_ps), float(cosphi), int(bool(der)), float(pv_k),
            float(ins), int(bess_mode), float(soc), int(bool(island)),
            tuple(float(x) for x in prof), tuple(float(x) for x in pvp))


# ════════════════════════════════════════════════════════════════════════════
# Симулятор телеметрії
# ════════════════════════════════════════════════════════════════════════════
FEEDER_SRC = {"Ф-1": "ПС-1", "Ф-2": "ПС-1", "Ф-3": "ПС-2", "Ф-4": "ПС-2"}


def build_assets():
    nodes, _ = build_grid()
    assets = {n: dict(kind="ps", kva=12600, load=0.0, feeder_of=n) for n, d in nodes.items() if d["type"] == "src"}
    for n, d in nodes.items():
        if d["type"] == "tp":
            assets[n] = dict(kind="tp", kva=d["kva"], load=d["load"], feeder_of=FEEDER_SRC[d["feeder"]])
    for s in [a for a, d in assets.items() if d["kind"] == "ps"]:
        assets[s]["load"] = sum(a["load"] for a in assets.values() if a["kind"] == "tp" and a["feeder_of"] == s)
    return assets


class Simulator:
    def __init__(self, ambient=-5.0, sens=1.2, seed=None, start_hour=6.0, vstep=15.0):
        self.assets = build_assets()
        self.rng = random.Random(seed)
        self.ambient, self.sens = ambient, sens
        self.vt = start_hour * 60.0
        self.vstep = vstep
        self.oil = {a: ambient + 15.0 for a in self.assets}
        self.events = {}          # asset -> (kind, ticks_left)

    def rated_a(self, a):
        d = self.assets[a]
        return d["kva"] / (math.sqrt(3) * (10.5 if d["kind"] == "ps" else 10.0))

    def tick(self):
        self.vt += self.vstep
        h = (self.vt / 60.0) % 24.0
        amb = self.ambient + 4.0 * math.sin(2 * math.pi * (h - 9.0) / 24.0)
        tf = 1.0 + self.sens / 100.0 * max(0.0, 5.0 - amb)
        scale = HOURLY[int(h)] * tf
        tps = [a for a, d in self.assets.items() if d["kind"] == "tp"]
        if self.rng.random() < 0.04:
            self.events[self.rng.choice(tps)] = ("overload", 8)
        if self.rng.random() < 0.02:
            self.events[self.rng.choice(tps)] = ("cooling", 20)
        msgs = []
        for a, d in self.assets.items():
            cap_mw = d["kva"] * 0.95 / 1000.0
            loading = d["load"] * scale / cap_mw
            kind, left = self.events.get(a, (None, 0))
            if kind == "overload":
                loading *= 1.5
            loading = max(0.0, loading * (1 + self.rng.gauss(0, 0.03)))
            gain = 1.35 if kind == "cooling" else 1.0
            target = amb + 52.0 * gain * loading ** 1.8
            self.oil[a] += 0.15 * (target - self.oil[a])
            oil = self.oil[a] + self.rng.gauss(0, 0.4)
            wind = oil + 23.0 * loading ** 2 + self.rng.gauss(0, 0.5)
            u = 10.5 - 0.35 * min(loading, 1.3) + self.rng.gauss(0, 0.03)
            i_a = loading * self.rated_a(a)
            alarm = "OVERLOAD" if loading > 1.0 else ("OIL_HIGH" if oil > 95 else "")
            msgs.append(dict(asset=a, kind=d["kind"], ts=time.time(), vt=f"{int(h):02d}:{int((h % 1) * 60):02d}",
                             hour=round(h, 2), i_a=round(i_a, 1), i_rated_a=round(self.rated_a(a), 1),
                             loading=round(loading, 4), oil_c=round(oil, 1), winding_c=round(wind, 1),
                             u_kv=round(u, 3), p_mw=round(loading * cap_mw, 3), ambient_c=round(amb, 1),
                             alarm=alarm))
            if left > 0:
                self.events[a] = (kind, left - 1) if left > 1 else (None, 0)
        return msgs


# ════════════════════════════════════════════════════════════════════════════
# Генератор телеметрії для MQTT (CLI: python app.py --generator ...)
# ════════════════════════════════════════════════════════════════════════════
def run_generator_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", action="store_true", help="режим генератора (python app.py --generator ...)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--prefix", default="vinnytsia-twin")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--interval", type=float, default=2.0, help="пауза між тіками, с")
    ap.add_argument("--vstep", type=float, default=15.0, help="віртуальних хвилин за тік")
    ap.add_argument("--ambient", type=float, default=-5.0, help="середня температура повітря, °C")
    ap.add_argument("--sens", type=float, default=1.2, help="чутливість навантаження, %%/°C")
    ap.add_argument("--start-hour", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=0, help="0 = нескінченно")
    ap.add_argument("--dry-run", action="store_true", help="не підключатись до брокера, друкувати JSON")
    a = ap.parse_args()

    sim = Simulator(a.ambient, a.sens, a.seed, a.start_hour, a.vstep)

    client = None
    if not a.dry_run:
        import paho.mqtt.client as mqtt
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:                       # paho-mqtt 1.x
            client = mqtt.Client()
        if a.user:
            client.username_pw_set(a.user, a.password)
        ctrl_topic = f"{a.prefix}/control"

        def on_connect(cl, ud, flags, rc, *args):
            cl.subscribe(ctrl_topic)                 # команди з застосунку: інжекція/скидання аномалій

        def on_message(cl, ud, msg):
            try:
                c = json.loads(msg.payload.decode("utf-8"))
                if c.get("cmd") == "clear":
                    sim.events.clear()
                    print("команда: скинути аномалії")
                elif c.get("cmd") == "inject" and c.get("asset") in sim.assets:
                    sim.events[c["asset"]] = (c.get("kind", "overload"), int(c.get("ticks", 12)))
                    print(f"команда: аномалія {c.get('kind')} на {c['asset']}")
            except Exception:                        # noqa: BLE001
                pass
        client.on_connect, client.on_message = on_connect, on_message
        client.reconnect_delay_set(1, 10)
        client.connect(a.host, a.port, 30)
        client.loop_start()
        print(f"Підключено до {a.host}:{a.port}, топік {a.prefix}/telemetry/<актив>, керування {ctrl_topic}")

    n = 0
    try:
        while a.ticks == 0 or n < a.ticks:
            msgs = sim.tick()
            for m in msgs:
                payload = json.dumps(m, ensure_ascii=False)
                if client is None:
                    print(payload)
                else:
                    client.publish(f"{a.prefix}/telemetry/{m['asset']}", payload, qos=0)
            n += 1
            if client is not None:
                alarms = [m["asset"] + ":" + m["alarm"] for m in msgs if m["alarm"]]
                print(f"тік {n}: віртуальний час {msgs[0]['vt']}, повідомлень {len(msgs)}"
                      + (f", тривоги: {', '.join(alarms)}" if alarms else ""))
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__" and "--generator" in sys.argv:
    run_generator_cli()
    sys.exit(0)


# ════════════════════════════════════════════════════════════════════════════
# Розрахунок режимів (pandapower), відновлення живлення, N-1/N-k
# ════════════════════════════════════════════════════════════════════════════
warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

# ════════════════════════════════════════════════════════════════════════════
# 2. PANDAPOWER: БАЗОВА МОДЕЛЬ ТА СТАН
# ════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def get_base():
    net = pp.create_empty_network(f_hz=50)
    M = dict(bus={}, hv={}, line={}, load={}, pv={}, bess={}, slack={}, trafo={}, extg={})
    for n, d in NODES.items():
        b = pp.create_bus(net, vn_kv=10.0, name=n)
        M["bus"][n] = b
        if d["type"] == "src":
            hv = pp.create_bus(net, vn_kv=110.0, name=f"{n} 110кВ")
            M["hv"][n] = hv
            M["extg"][n] = pp.create_ext_grid(net, hv, vm_pu=1.0, s_sc_max_mva=2500, s_sc_min_mva=1200,
                                              rx_max=0.1, rx_min=0.1, name=f"Система {n}")
            M["trafo"][n] = pp.create_transformer_from_parameters(
                net, hv, b, sn_mva=12.6, vn_hv_kv=110.0, vn_lv_kv=10.5, vkr_percent=0.5,
                vk_percent=10.5, pfe_kw=12.0, i0_percent=0.2, name=f"Т-{n}")
        else:
            M["load"][n] = pp.create_load(net, b, p_mw=d["load"], q_mvar=0.0, name=f"Навантаження {n}")
            if d["pv"] > 0:
                M["pv"][n] = pp.create_sgen(net, b, p_mw=0.0, q_mvar=0.0, sn_mva=d["pv"] * 1.05, k=1.2,
                                            type="pv", name=f"PV {n}")
            if d["bess_mw"] > 0:
                M["bess"][n] = pp.create_sgen(net, b, p_mw=0.0, q_mvar=0.0, sn_mva=d["bess_mw"], k=1.2,
                                              type="pv", name=f"АБ {n}")
                M["slack"][n] = pp.create_ext_grid(net, b, vm_pu=1.0, in_service=False, s_sc_max_mva=50,
                                                   s_sc_min_mva=50, rx_max=0.1, rx_min=0.1,
                                                   name=f"АБ-формувач {n}")
    for eid, e in EDGES.items():
        M["line"][eid] = pp.create_line_from_parameters(
            net, M["bus"][e["u"]], M["bus"][e["v"]], length_km=e["len"], r_ohm_per_km=e["r"],
            x_ohm_per_km=e["x"], c_nf_per_km=e["c"], max_i_ka=e["imax"], name=eid, endtemp_degree=80.0)
    return net, M


BASE, MAPS = get_base()


def new_net(p):
    net = copy.deepcopy(BASE)
    net.trafo["sn_mva"] = p.src_cap / 0.95
    net.ext_grid.loc[list(MAPS["extg"].values()), "vm_pu"] = p.u_ps / 10.5
    return net


def pv_mw_node(n, p, topo, shed_pv, f_nodes):
    d = NODES[n]
    if (not p.der) or d["pv"] <= 0 or n in shed_pv or n in f_nodes or n not in topo["energized"]:
        return 0.0
    if topo["root"][n] not in SOURCES and not p.island:      # антиострівний захист інверторів
        return 0.0
    return d["pv"] * p.pv_k * pv_frac(p)


def bess_mw_node(n, p, topo, f_nodes):
    d = NODES[n]
    if (not p.der) or d["bess_mw"] <= 0 or n in f_nodes or n not in topo["energized"]:
        return 0.0
    if topo["root"][n] not in SOURCES:
        return 0.0
    return d["bess_mw"] * bess_frac(p.hour, p.bess_mode, p.soc)


def topology(closed, f_nodes, f_edges, gf=()):
    """Граф живлення: BFS від ПС і від АБ, що формують острів (gf). Радіальна структура."""
    adj = defaultdict(list)
    for eid in closed:
        if eid in f_edges:
            continue
        e = EDGES[eid]
        if e["u"] in f_nodes or e["v"] in f_nodes:
            continue
        adj[e["u"]].append((e["v"], eid))
        adj[e["v"]].append((e["u"], eid))
    parent, order, root, dist, edge_pc = {}, [], {}, {}, {}
    roots = [s for s in SOURCES if s not in f_nodes] + [g for g in gf if g not in f_nodes]
    for s in roots:
        if s in parent:
            continue
        parent[s], root[s], dist[s] = (None, None), s, 0.0
        q = deque([s])
        while q:
            n = q.popleft()
            order.append(n)
            for m, eid in adj[n]:
                if m not in parent:
                    parent[m], root[m] = (n, eid), s
                    dist[m] = dist[n] + EDGES[eid]["len"]
                    edge_pc[eid] = (n, m)
                    q.append(m)
    children = defaultdict(list)
    for n, (par, _) in parent.items():
        if par is not None:
            children[par].append(n)
    return dict(parent=parent, order=order, root=root, dist=dist, edge_pc=edge_pc,
                children=dict(children), energized=set(order))


def downstream(topo, node):
    out, stack = [], [node]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(topo["children"].get(n, []))
    return out


def gf_cap(topo, g):
    return sum(NODES[n]["bess_mw"] for n in topo["order"] if topo["root"][n] == g)


def apply_state(net, p, topo, closed, f_nodes, f_edges, scale, shed, shed_pv, gf):
    tanp = math.tan(math.acos(p.cosphi))
    for eid, idx in MAPS["line"].items():
        e = EDGES[eid]
        net.line.at[idx, "in_service"] = bool(eid in closed and eid not in f_edges
                                              and e["u"] not in f_nodes and e["v"] not in f_nodes)
    for n, b in MAPS["bus"].items():
        net.bus.at[b, "in_service"] = n not in f_nodes
    for s in SOURCES:
        ok = s not in f_nodes
        net.bus.at[MAPS["hv"][s], "in_service"] = ok
        net.trafo.at[MAPS["trafo"][s], "in_service"] = ok
        net.ext_grid.at[MAPS["extg"][s], "in_service"] = ok
    for n, idx in MAPS["load"].items():
        pl = 0.0 if (n in shed or n in f_nodes) else NODES[n]["load"] * scale
        net.load.at[idx, "in_service"] = n not in f_nodes
        net.load.at[idx, "p_mw"] = pl
        net.load.at[idx, "q_mvar"] = pl * tanp
    for n, idx in MAPS["pv"].items():
        net.sgen.at[idx, "in_service"] = n not in f_nodes
        net.sgen.at[idx, "p_mw"] = pv_mw_node(n, p, topo, shed_pv, f_nodes)
    for n, idx in MAPS["bess"].items():
        net.sgen.at[idx, "in_service"] = n not in f_nodes
        net.sgen.at[idx, "p_mw"] = bess_mw_node(n, p, topo, f_nodes)
    for n, idx in MAPS["slack"].items():
        net.ext_grid.at[idx, "in_service"] = bool(n in gf and n not in f_nodes)


def pp_state(net, p, topo, closed, f_nodes, f_edges, scale, shed, shed_pv, gf):
    """Справжній розрахунок режиму (pandapower, Newton-Raphson) -> узагальнений стан."""
    apply_state(net, p, topo, closed, f_nodes, f_edges, scale, shed, shed_pv, gf)
    out = dict(conv=True, err="", vm={}, ratio={}, i_a={}, p_from={}, q_from={}, pl_kw={}, pflow_down={},
               src={}, p_src={}, q_src={}, i_src_a={}, losses=0.0, pv_mw=0.0, bess_mw=0.0)
    if not topo["order"]:
        return out
    try:
        pp.runpp(net, numba=False, init="flat", calculate_voltage_angles=False, max_iteration=40)
    except Exception as ex:                         # noqa: BLE001
        out["conv"] = False
        out["err"] = f"{type(ex).__name__}: {ex}"
        return out
    rb, rl = net.res_bus, net.res_line
    for n, b in MAPS["bus"].items():
        if n in topo["energized"] and not np.isnan(rb.at[b, "vm_pu"]):
            out["vm"][n] = float(rb.at[b, "vm_pu"])
    edge_par = {eid: pc[0] for eid, pc in topo["edge_pc"].items()}
    for eid, idx in MAPS["line"].items():
        if eid in edge_par and not np.isnan(rl.at[idx, "loading_percent"]):
            pf_, e = float(rl.at[idx, "p_from_mw"]), EDGES[eid]
            out["ratio"][eid] = float(rl.at[idx, "loading_percent"]) / 100.0
            out["i_a"][eid] = float(rl.at[idx, "i_ka"]) * 1000.0
            out["p_from"][eid] = pf_
            out["q_from"][eid] = float(rl.at[idx, "q_from_mvar"])
            out["pl_kw"][eid] = float(rl.at[idx, "pl_mw"]) * 1000.0
            out["pflow_down"][eid] = pf_ if e["u"] == edge_par[eid] else -pf_
    out["losses"] = float(np.nansum(rl["pl_mw"].values))
    for s in SOURCES:
        ti = MAPS["trafo"][s]
        if s in topo["energized"] and not np.isnan(net.res_trafo.at[ti, "loading_percent"]):
            out["src"][s] = float(net.res_trafo.at[ti, "loading_percent"]) / 100.0
            out["i_src_a"][s] = float(net.res_trafo.at[ti, "i_lv_ka"]) * 1000.0
            out["p_src"][s] = float(net.res_ext_grid.at[MAPS["extg"][s], "p_mw"])
            out["q_src"][s] = float(net.res_ext_grid.at[MAPS["extg"][s], "q_mvar"])
            out["losses"] += float(net.res_trafo.at[ti, "pl_mw"])
    for g in gf:
        if g in topo["energized"] and g in MAPS["slack"]:
            idx = MAPS["slack"][g]
            pg, qg = float(net.res_ext_grid.at[idx, "p_mw"]), float(net.res_ext_grid.at[idx, "q_mvar"])
            out["p_src"][g], out["q_src"][g] = pg, qg
            out["src"][g] = math.hypot(pg, qg) / max(gf_cap(topo, g), 1e-6)
    for n in MAPS["pv"]:
        out["pv_mw"] += float(net.sgen.at[MAPS["pv"][n], "p_mw"])
    for n in MAPS["bess"]:
        out["bess_mw"] += float(net.sgen.at[MAPS["bess"][n], "p_mw"])
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. ШВИДКИЙ ПРЕСКРИНІНГ (лише для підбору кандидатів) ТА ПОРУШЕННЯ
# ════════════════════════════════════════════════════════════════════════════
def fast_state(topo, p, scale, shed, shed_pv):
    tanp = math.tan(math.acos(p.cosphi))
    order, par = topo["order"], topo["parent"]
    P, Q = {}, {}
    for n in order:
        if NODES[n]["type"] == "tp":
            pl = 0.0 if n in shed else NODES[n]["load"] * scale
            P[n] = pl - pv_mw_node(n, p, topo, shed_pv, frozenset()) - bess_mw_node(n, p, topo, frozenset())
            Q[n] = pl * tanp
        else:
            P[n] = Q[n] = 0.0
    sP, sQ = dict(P), dict(Q)
    for n in reversed(order):
        pn, _ = par[n]
        if pn is not None:
            sP[pn] += sP[n]
            sQ[pn] += sQ[n]
    sn = p.src_cap / 0.95
    rt, xt = 0.005 * 10.5 ** 2 / sn, 0.105 * 10.5 ** 2 / sn
    vm, ratio, pdown, src, psrc = {}, {}, {}, {}, {}
    for n in order:
        pn, eid = par[n]
        if pn is None:
            if n in SOURCES:
                v0 = p.u_ps / 10.0
                vm[n] = v0 - (sP[n] * rt + sQ[n] * xt) / (100 * v0)
                src[n] = math.hypot(sP[n], sQ[n]) / sn
            else:
                vm[n] = 1.0
                src[n] = math.hypot(sP[n], sQ[n]) / max(gf_cap(topo, n), 1e-6)
            psrc[n] = sP[n]
        else:
            e, vp = EDGES[eid], max(vm[pn], 0.5)
            vm[n] = vm[pn] - (sP[n] * e["r"] * e["len"] + sQ[n] * e["x"] * e["len"]) / (100 * vp)
            ratio[eid] = math.hypot(sP[n], sQ[n]) / (math.sqrt(3) * 10 * vp) / e["imax"]
            pdown[eid] = sP[n]
    return dict(conv=True, vm=vm, ratio=ratio, pflow_down=pdown, src=src, p_src=psrc)


def violations(stt, p):
    if not stt["conv"]:
        return [(10.0, "conv", "")]
    t, out = 1 + p.tol, []
    for eid, r in stt["ratio"].items():
        if r > t + 1e-9:
            out.append((r / t, "line", eid))
    for s, r in stt["src"].items():
        if r > t + 1e-9:
            out.append((r / t, "src", s))
    for n, v in stt["vm"].items():
        if NODES[n]["type"] == "tp":
            if v < p.vmin - 1e-9:
                out.append((p.vmin / v, "vmin", n))
            elif v > p.vmax + 1e-9:
                out.append((v / p.vmax, "vmax", n))
    return out


def max_ratio(stt):
    vals = list(stt["ratio"].values()) + list(stt["src"].values())
    return max(vals) if vals else 0.0


def pick_fix(stt, topo, p, shed, shed_pv, scale):
    """Що відключити/обмежити, щоб усунути найгірше порушення: ('load', n) | ('pv', n) | None."""
    viol = violations(stt, p)
    if not viol:
        return None
    _, kind, ident = max(viol)

    def pv_cands(nodes):
        return [n for n in nodes if NODES[n]["type"] == "tp"
                and pv_mw_node(n, p, topo, shed_pv, frozenset()) > 1e-9]

    def load_cands(nodes):
        return [n for n in nodes if NODES[n]["type"] == "tp" and n not in shed
                and NODES[n]["load"] * scale > 0]

    if kind == "conv":
        c = load_cands(topo["order"])
        return ("load", max(c, key=lambda n: (NODES[n]["prio"], NODES[n]["load"]))) if c else None
    if kind == "line":
        nodes = downstream(topo, topo["edge_pc"][ident][1])
        export = stt["pflow_down"].get(ident, 0.0) < 0
    elif kind == "src":
        nodes = [n for n in topo["order"] if topo["root"][n] == ident]
        export = stt["p_src"].get(ident, 0.0) < 0
    elif kind == "vmax":
        c = pv_cands([n for n in topo["order"] if topo["root"][n] == topo["root"][ident]])
        return ("pv", max(c, key=lambda n: NODES[n]["pv"])) if c else None
    else:                                           # vmin
        n = ident
        while topo["parent"][n][0] is not None and topo["parent"][topo["parent"][n][0]][0] is not None:
            n = topo["parent"][n][0]
        nodes = downstream(topo, n)
        export = False
        c = load_cands(nodes)
        return ("load", max(c, key=lambda m: (NODES[m]["prio"], topo["dist"][m], NODES[m]["load"]))) if c else None
    if export:
        c = pv_cands(nodes)
        if c:
            return ("pv", max(c, key=lambda n: NODES[n]["pv"]))
    c = load_cands(nodes)
    return ("load", max(c, key=lambda n: (NODES[n]["prio"], NODES[n]["load"]))) if c else None


def balance_fast(topo, p, scale, shed, shed_pv):
    shed, shed_pv = set(shed), set(shed_pv)
    stt = fast_state(topo, p, scale, shed, shed_pv)
    for _ in range(80):
        fix = pick_fix(stt, topo, p, shed, shed_pv, scale)
        if fix is None:
            break
        (shed if fix[0] == "load" else shed_pv).add(fix[1])
        stt = fast_state(topo, p, scale, shed, shed_pv)
    return shed, shed_pv, stt


def gf_autonomy(topo, stt, gf, p):
    """Автономність острова (год): корисна енергія АБ / потужність розряду."""
    out = {}
    for g in gf:
        if g not in topo["energized"]:
            continue
        nodes = [n for n in topo["order"] if topo["root"][n] == g and NODES[n]["bess_mwh"] > 0]
        e = sum(NODES[n]["bess_mwh"] for n in nodes) * max(0.0, p.soc - SOC_FLOOR) * BESS_EFF
        pg = stt["p_src"].get(g, 0.0)
        out[g] = e / pg if pg > 1e-6 else 1e9
    return out


def total_cost(topo, stt, shed, shed_pv, gf, p, scale, f_nodes):
    aut = gf_autonomy(topo, stt, gf, p)
    c = 0.0
    for n in TPS:
        if n in f_nodes:
            continue
        w = PRIO_W[NODES[n]["prio"]] * NODES[n]["load"] * scale
        if n in shed or n not in topo["energized"]:
            c += w
        elif topo["root"][n] in aut:
            c += w * max(0.0, 1.0 - aut[topo["root"][n]] / p.repair_h)
    return c + 0.02 * sum(NODES[n]["pv"] for n in shed_pv)


# ════════════════════════════════════════════════════════════════════════════
# 4. СЦЕНАРІЙ: АВАРІЯ -> ІЗОЛЯЦІЯ -> ВІДНОВЛЕННЯ (ПЕРЕМИЧКИ / ОСТРІВ) -> ГОП
# ════════════════════════════════════════════════════════════════════════════
def solve(P, elements):
    p = Params(*P)
    f_nodes = frozenset(e for e in elements if e in NODES)
    f_edges = frozenset(e for e in elements if e in EDGES)
    scale = load_scale(p)
    net = new_net(p)
    closed0 = set(CLOSED0)
    topo_n = topology(closed0, E0, E0)
    normal = pp_state(net, p, topo_n, closed0, E0, E0, scale, set(), set(), ())
    closed_f = closed0 - f_edges
    topo_p = topology(closed_f, f_nodes, f_edges)
    post = pp_state(net, p, topo_p, closed_f, f_nodes, f_edges, scale, set(), set(), ())

    steps, ops = [], 0

    def add(typ, action, obj, why, inc=1):
        nonlocal ops
        ops += inc
        d = {"type": typ, "№": len(steps) + 1, "Час, хв": round(p.t_detect + ops * p.t_switch, 1),
             "Дія": action, "Об'єкт": obj, "Обґрунтування": why}
        steps.append(d)
        return d

    steps.append({"type": "detect", "№": 1, "Час, хв": p.t_detect,
                  "Дія": "Фіксація відмови та локалізація пошкодження",
                  "Об'єкт": ", ".join(sorted(elements)),
                  "Обґрунтування": "Сигнали РЗА/ТС, телеметрія SCADA"})
    iso = {}
    for n in sorted(f_nodes):
        for eid in sorted(closed_f):
            if n in (EDGES[eid]["u"], EDGES[eid]["v"]):
                iso[eid] = (1, f"ізоляція {n}")
    for eid in sorted(f_edges & closed0):
        iso[eid] = (2, "ізоляція пошкодженого сегмента (обидва кінці)")
    for eid, (inc, why) in iso.items():
        add("isolate", "Відкрити комутаційний апарат", eid, why, inc)

    closed, shed, shed_pv, gf = set(closed_f), set(), set(), []
    restore_min, island_steps = {}, {}
    for _ in range(25):
        topo_c = topology(closed, f_nodes, f_edges, gf)
        unener = set(TPS) - topo_c["energized"] - f_nodes
        H = nx.Graph()
        H.add_nodes_from(unener)
        for eid in closed:
            e = EDGES[eid]
            if eid not in f_edges and e["u"] in unener and e["v"] in unener:
                H.add_edge(e["u"], e["v"])
        comps = sorted(nx.connected_components(H),
                       key=lambda c: (min(NODES[n]["prio"] for n in c), -sum(NODES[n]["load"] for n in c)))
        base_stt = fast_state(topo_c, p, scale, shed, shed_pv)
        base_cost = total_cost(topo_c, base_stt, shed, shed_pv, gf, p, scale, f_nodes)
        progressed = False
        for comp in comps:
            en, options = topo_c["energized"], []
            for tie, e in EDGES.items():
                if tie in closed or tie in f_edges:
                    continue
                if (e["u"] in comp and e["v"] in en) or (e["v"] in comp and e["u"] in en):
                    t2 = topology(closed | {tie}, f_nodes, f_edges, gf)
                    sh2, pv2, s2 = balance_fast(t2, p, scale, shed, shed_pv)
                    options.append((total_cost(t2, s2, sh2, pv2, gf, p, scale, f_nodes), 0, max_ratio(s2),
                                    tie, "tie", closed | {tie}, list(gf), sh2, pv2, s2))
            bnodes = [n for n in comp if NODES[n]["bess_mw"] > 0] if (p.der and p.island) else []
            if bnodes:
                g = max(bnodes, key=lambda n: NODES[n]["bess_mw"])
                gf2 = list(gf) + [g]
                t2 = topology(closed, f_nodes, f_edges, gf2)
                sh2, pv2, s2 = balance_fast(t2, p, scale, shed, shed_pv)
                options.append((total_cost(t2, s2, sh2, pv2, gf2, p, scale, f_nodes), 1, max_ratio(s2),
                                g, "island", set(closed), gf2, sh2, pv2, s2))
            if not options:
                continue
            best = min(options, key=lambda o: o[:3])
            if best[0] >= base_cost - 1e-9:
                continue
            _, _, _, ident, kind, closed_n, gf_n, sh2, pv2, s2 = best
            for n in sorted(sh2 - shed):
                add("shed", "Відключити ТП (графік обмежень, ГОП)", n,
                    f"Усунення порушень режиму; {NODES[n]['ctype']}, пріоритет {PRIO_TXT[NODES[n]['prio']]}, "
                    f"{NODES[n]['load'] * scale:.2f} МВт")
            for n in sorted(pv2 - shed_pv):
                add("pv", "Обмежити генерацію PV", n, "Усунення перенапруги/зворотного перевантаження")
            consum = sum(NODES[n]["cons"] for n in comp)
            mw = sum(NODES[n]["load"] for n in comp) * scale
            if kind == "tie":
                e = EDGES[ident]
                entry = e["v"] if e["u"] in comp else e["u"]
                t_chk = topology(closed_n, f_nodes, f_edges, gf_n)
                add("close", "Закрити секційний вимикач (замкнути перемичку)", ident,
                    f"Відновлення {len(comp)} ТП ({consum} спож., {mw:.2f} МВт) від {t_chk['root'][entry]}")
            else:
                island_steps[ident] = add(
                    "island", "Перевести АБ у режим формування мережі (острівний режим)", f"АБ {ident}",
                    f"Живлення {len(comp)} ТП ({consum} спож., {mw:.2f} МВт) від АБ і ВДЕ")
            for n in comp:
                restore_min[n] = p.t_detect + ops * p.t_switch
            closed, gf, shed, shed_pv = set(closed_n), list(gf_n), set(sh2), set(pv2)
            progressed = True
            break
        if not progressed:
            break

    # --- доведення режиму справжнім Power Flow (pandapower) ---
    topo_f = topology(closed, f_nodes, f_edges, gf)
    fin = None
    for _ in range(30):
        fin = pp_state(net, p, topo_f, closed, f_nodes, f_edges, scale, shed, shed_pv, gf)
        fix = pick_fix(fin, topo_f, p, shed, shed_pv, scale)
        if fix is None:
            break
        if fix[0] == "load":
            shed.add(fix[1])
            add("shed", "Відключити ТП (ГОП за результатом Power Flow)", fix[1],
                f"Порушення за розрахунком pandapower; {NODES[fix[1]]['ctype']}, пріоритет {PRIO_TXT[NODES[fix[1]]['prio']]}, "
                f"{NODES[fix[1]]['load'] * scale:.2f} МВт")
        else:
            shed_pv.add(fix[1])
            add("pv", "Обмежити генерацію PV", fix[1], "Перенапруга/зворотний потік за Power Flow")
    else:
        fin = pp_state(net, p, topo_f, closed, f_nodes, f_edges, scale, shed, shed_pv, gf)
    viol = violations(fin, p)

    aut = gf_autonomy(topo_f, fin, gf, p)
    for g, stp in island_steps.items():
        nodes = [n for n in TPS if topo_f["root"].get(n) == g and n not in shed]
        a = aut.get(g, 1e9)
        stp["Обґрунтування"] = (
            f"Живлення {len(nodes)} ТП ({sum(NODES[n]['cons'] for n in nodes)} спож.) від АБ "
            f"{fin['p_src'].get(g, 0.0):.2f} МВт + ВДЕ; автономність "
            + ("більша за тривалість ремонту" if a >= p.repair_h else f"≈ {a:.1f} год (SoC {p.soc:.0%})"))

    status = {}
    for n in TPS:
        if n in f_nodes:
            status[n] = "failed"
        elif n in shed:
            status[n] = "shed"
        elif n not in topo_f["energized"]:
            status[n] = "lost"
        elif topo_f["root"][n] in gf:
            status[n] = "island"
        elif n not in topo_p["energized"]:
            status[n] = "restored"
        else:
            status[n] = "ok"
    lost = [n for n in TPS if status[n] == "lost"]
    if lost:
        steps.append({"type": "info", "№": len(steps) + 1, "Час, хв": round(p.repair_h * 60, 1),
                      "Дія": "Живлення відсутнє до завершення ремонту", "Об'єкт": ", ".join(lost),
                      "Обґрунтування": "Немає резервного шляху від ПС чи автономного джерела"})
    steps.append({"type": "repair", "№": len(steps) + 1, "Час, хв": round(p.repair_h * 60, 1),
                  "Дія": "Ремонт/заміна обладнання, повернення до нормальної схеми",
                  "Об'єкт": ", ".join(sorted(elements)),
                  "Обґрунтування": "Виїзд ОВБ, заява на вивід, синхронізація та розмикання перемичок"})

    tot_mw = sum(NODES[n]["load"] for n in TPS) * scale
    tot_cons = sum(NODES[n]["cons"] for n in TPS)
    unsup = [n for n in TPS if status[n] in ("failed", "lost", "shed")]
    ens = cmi = 0.0
    for n in TPS:
        stn = status[n]
        if stn == "ok":
            continue
        if stn == "restored":
            dur = restore_min[n] / 60.0
        elif stn == "island":
            dur = restore_min[n] / 60.0 + max(0.0, p.repair_h - aut.get(topo_f["root"][n], 1e9))
        else:
            dur = p.repair_h
        ens += NODES[n]["load"] * scale * dur
        cmi += NODES[n]["cons"] * dur * 60.0
    vms = [v for n, v in fin["vm"].items() if NODES[n]["type"] == "tp"]
    return dict(
        scale=scale, total_mw=tot_mw, unsup_mw=sum(NODES[n]["load"] for n in unsup) * scale,
        ens=ens, cust_cut=sum(NODES[n]["cons"] for n in TPS if status[n] != "ok"),
        cust_unrestored=sum(NODES[n]["cons"] for n in unsup), saidi=cmi / tot_cons,
        steps=steps, status=status, normal=normal, post=post, final=fin,
        topo_n=topo_n, topo_p=topo_p, topo_f=topo_f, closed0=closed0, closed_f=closed_f,
        closed_fin=closed, shed=sorted(shed), shed_pv=sorted(shed_pv), gf=gf, aut=aut,
        f_nodes=f_nodes, f_edges=f_edges, restore_min=restore_min, viol=len(viol),
        max_line=max_ratio(fin), vmin_fin=min(vms) if vms else float("nan"),
        losses=fin["losses"], scale_=scale)


solve_c = lru_cache(maxsize=256)(solve)


@lru_cache(maxsize=16)
def scan_all(P):
    rows, prots = [], {}
    for el in ALL_ELEMS:
        r = solve(P, (el,))
        prots[el] = r["steps"]
        acts = [s["Дія"].split(" (")[0] + " " + s["Об'єкт"]
                for s in r["steps"] if s["type"] in ("close", "shed", "island", "pv")]
        rows.append(dict(
            id=el, type=elem_type(el), unsup_mw=r["unsup_mw"],
            unsup_pct=100 * r["unsup_mw"] / r["total_mw"], ens_mwh=r["ens"],
            cust=r["cust_unrestored"], saidi=r["saidi"], max_line=100 * r["max_line"],
            vmin=r["vmin_fin"], n_ops=len(acts), plan="; ".join(acts) if acts else "перемикання не потрібні"))
    return pd.DataFrame(rows), prots


@lru_cache(maxsize=16)
def sens_grid(P, elements, hours, temps):
    return [[solve((float(t), int(h)) + tuple(P[2:-2]) + ((), ()), elements)["unsup_mw"] for h in hours]
            for t in temps]


@lru_cache(maxsize=16)
def daily_profile(P):
    p0 = Params(*P)
    rows = []
    for h in range(24):
        for tag, der in (("ВДЕ", p0.der), ("без ВДЕ", 0)):
            p = p0._replace(hour=h, der=der)
            net = new_net(p)
            topo = topology(set(CLOSED0), E0, E0)
            stt = pp_state(net, p, topo, set(CLOSED0), E0, E0, load_scale(p), set(), set(), ())
            vms = [v for n, v in stt["vm"].items() if NODES[n]["type"] == "tp"]
            rows.append(dict(h=h, case=tag, load=sum(NODES[n]["load"] for n in TPS) * load_scale(p),
                             pv=stt["pv_mw"], bess=stt["bess_mw"], imp=sum(stt["p_src"].get(s, 0) for s in SOURCES),
                             losses=stt["losses"], vmin=min(vms) if vms else float("nan"),
                             vmax=max(vms) if vms else float("nan"),
                             max_line=max(stt["ratio"].values()) if stt["ratio"] else 0.0))
    return pd.DataFrame(rows)


def health_check(p):
    """Швидка перевірка, що Power Flow рахується в нормальній схемі: (ok, текст помилки)."""
    stt = pp_state(new_net(p), p, topology(set(CLOSED0), E0, E0), set(CLOSED0), E0, E0,
                   load_scale(p), set(), set(), ())
    return stt["conv"], stt.get("err", "")


# ════════════════════════════════════════════════════════════════════════════
# Струми КЗ та релейний захист
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 5. СТРУМИ КЗ ТА РЕЛЕЙНИЙ ЗАХИСТ
# ════════════════════════════════════════════════════════════════════════════
K_N, K_ZAP, K_V = 1.2, 1.3, 0.95
T_MSZ, T_IN, SWITCH_BREAK_KA = 0.7, 1.2, 20.0


def sc_currents(net, buses, fault, case):
    buses = sorted(set(MAPS["bus"][b] if isinstance(b, str) else b for b in buses))
    sc.calc_sc(net, bus=buses, fault=fault, case=case, ip=False, ith=False)
    return {b: float(net.res_bus_sc.at[b, "ikss_ka"]) for b in buses}


def head_of(topo, node):
    if node in SOURCES:
        return None
    n = node
    while True:
        par, eid = topo["parent"][n]
        if par in SOURCES:
            return eid
        n = par


@lru_cache(maxsize=16)
def protection_settings(P):
    p = Params(*P)._replace(temp=-20.0, hour=18, der=0, prof=(), pvp=())
    net = new_net(p)
    topo = topology(set(CLOSED0), E0, E0)
    stt = pp_state(net, p, topo, set(CLOSED0), E0, E0, load_scale(p), set(), set(), ())
    if not stt["conv"]:
        raise RuntimeError("Power Flow не зійшовся при розрахунку уставок: "
                           + (stt.get("err") or "невідома помилка")
                           + ". Перевірте, що встановлено pandas<3 (див. requirements.txt).")
    net.sgen["in_service"] = False
    net.ext_grid.loc[list(MAPS["slack"].values()), "in_service"] = False
    heads = [eid for eid in CLOSED0 if EDGES[eid]["u"] in SOURCES]
    ik3 = sc_currents(net, [EDGES[h]["v"] for h in heads], "3ph", "max")
    feeders = {}
    for h in heads:
        iw = stt["i_a"][h]
        feeders[h] = dict(src=EDGES[h]["u"], iw=iw, i_to=1.25 * ik3[MAPS["bus"][EDGES[h]["v"]]] * 1000,
                          i_msz=K_N * K_ZAP / K_V * iw)
    inp = {s: K_N * K_ZAP / K_V * stt["i_src_a"][s] for s in SOURCES}
    return feeders, inp


def verdict(i3, i2, st_f, i_in):
    """Реакція захисту фідера на КЗ: (текст, рівень 0=ок,1=увага,2=відмова, t, kч)."""
    kch = i2 / st_f["i_msz"]
    if i2 >= st_f["i_to"]:
        return "Відсічка (ТВ) спрацює миттєво", 0, 0.0, kch
    if kch >= 1.5:
        return f"МСЗ фідера спрацює з витримкою {T_MSZ} с", 0, T_MSZ, kch
    if kch >= 1.2:
        return "МСЗ чутливий межово (kч < 1.5)", 1, T_MSZ, kch
    if i_in and i2 / i_in >= 1.2:
        return f"Лише резервний захист вводу ПС ({T_IN} с)", 1, T_IN, kch
    return "ЗАХИСТ НЕ СПРАЦЮЄ (немає чутливості)", 2, None, kch


@lru_cache(maxsize=16)
def fault_report(P, elements):
    p = Params(*P)
    res = solve_c(P, elements)
    feeders, inp = protection_settings(P)
    scale = load_scale(p)
    topo_n = res["topo_n"]
    net = new_net(p)
    apply_state(net, p, topo_n, set(CLOSED0), E0, E0, scale, set(), set(), ())
    net.ext_grid.loc[list(MAPS["slack"].values()), "in_service"] = False
    pts = []
    for el in elements:
        if el in NODES:
            pts.append((f"Шини {el}", el))
        else:
            e = EDGES[el]
            pts += [(f"Лінія {el}: початок ({e['u']})", e["u"]), (f"Лінія {el}: кінець ({e['v']})", e["v"])]
    buses = [b for _, b in pts]
    net.sgen["in_service"] = bool(p.der)
    i3 = sc_currents(net, buses, "3ph", "max")
    net.sgen["in_service"] = False
    i2 = sc_currents(net, buses, "2ph", "min")
    rows = []
    for label, node in pts:
        b = MAPS["bus"][node]
        k3, k2 = i3[b], i2[b]
        h = head_of(topo_n, node)
        if h is None:
            i_in = inp[node]
            kch = k2 * 1000 / i_in
            txt, lvl, t = (f"Вводний захист ПС спрацює ({T_IN} с)", 0, T_IN) if kch >= 1.5 else \
                ("Вводний захист чутливий межово", 1, T_IN) if kch >= 1.2 else ("ЗАХИСТ НЕ СПРАЦЮЄ", 2, None)
            rows.append({"Місце КЗ": label, "Ik3 max, кА": k3, "Ik2 min, кА": k2, "Захист": f"Ввід {node}",
                         "I відсічки, А": np.nan, "I МСЗ, А": i_in, "kч (МСЗ)": kch, "Реакція": txt,
                         "t, с": t if t is not None else np.nan, "lvl": lvl,
                         "Вимикач ≥ Ik3": "✅" if k3 <= SWITCH_BREAK_KA else "❌"})
        else:
            stg = feeders[h]
            txt, lvl, t, kch = verdict(k3 * 1000, k2 * 1000, stg, inp[stg["src"]])
            rows.append({"Місце КЗ": label, "Ik3 max, кА": k3, "Ik2 min, кА": k2, "Захист": f"Фідер {h}",
                         "I відсічки, А": stg["i_to"], "I МСЗ, А": stg["i_msz"], "kч (МСЗ)": kch, "Реакція": txt,
                         "t, с": t if t is not None else np.nan, "lvl": lvl,
                         "Вимикач ≥ Ik3": "✅" if k3 <= SWITCH_BREAK_KA else "❌"})
    # ----- перевірка резервних фідерів після перемикань -----
    topo_f, fin = res["topo_f"], res["final"]
    net2 = new_net(p)
    apply_state(net2, p, topo_f, res["closed_fin"], res["f_nodes"], res["f_edges"], scale,
                set(res["shed"]), set(res["shed_pv"]), res["gf"])
    net2.sgen["in_service"] = False
    net2.ext_grid.loc[list(MAPS["slack"].values()), "in_service"] = False
    far = {}
    for h, stg in feeders.items():
        if h in fin["i_a"]:
            child = topo_f["edge_pc"][h][1]
            far[h] = max(downstream(topo_f, child), key=lambda n: topo_f["dist"][n])
    i2far = sc_currents(net2, list(far.values()), "2ph", "min") if far else {}
    post_rows = []
    for h, stg in feeders.items():
        if h not in fin["i_a"]:
            post_rows.append({"Фідер": h, "Стан": "відключений/без напруги", "I роб. після, А": np.nan,
                              "I МСЗ, А": stg["i_msz"], "Запас МСЗ": np.nan, "Найдальший вузол": "-",
                              "Ik2 min у кінці, А": np.nan, "kч у кінці": np.nan, "Оцінка": "—", "lvl": 0})
            continue
        iload = fin["i_a"][h]
        margin = stg["i_msz"] / iload
        ik = i2far[MAPS["bus"][far[h]]] * 1000
        kch = ik / stg["i_msz"]
        notes, lvl = [], 0
        if margin < 1.0:
            notes.append("ХИБНЕ СПРАЦЮВАННЯ МСЗ по струму навантаження")
            lvl = 2
        elif margin < 1.15:
            notes.append("запас МСЗ межовий")
            lvl = max(lvl, 1)
        if kch < 1.2:
            notes.append("немає чутливості в кінці фідера")
            lvl = 2
        elif kch < 1.5:
            notes.append("чутливість у кінці межова")
            lvl = max(lvl, 1)
        post_rows.append({"Фідер": h, "Стан": "живить споживачів", "I роб. після, А": iload,
                          "I МСЗ, А": stg["i_msz"], "Запас МСЗ": margin, "Найдальший вузол": far[h],
                          "Ik2 min у кінці, А": ik, "kч у кінці": kch,
                          "Оцінка": "; ".join(notes) if notes else "ок", "lvl": lvl})
    in_rows = []
    for s in SOURCES:
        if s in fin["i_src_a"]:
            m = inp[s] / fin["i_src_a"][s]
            in_rows.append({"Ввід ПС": s, "I роб. після, А": fin["i_src_a"][s], "I МСЗ вводу, А": inp[s],
                            "Запас": m, "Оцінка": "ХИБНЕ СПРАЦЮВАННЯ вводу" if m < 1.0 else
                            "запас межовий" if m < 1.15 else "ок"})
    isl = []
    for g in res["gf"]:
        cap = gf_cap(topo_f, g)
        i_sc = 1.2 * cap / (math.sqrt(3) * 10.0) * 1000
        pg = fin.get("p_src", {}).get(g, 0.0) if fin else 0.0
        qg = fin.get("q_src", {}).get(g, 0.0) if fin else 0.0
        i_load = math.hypot(pg, qg) / (math.sqrt(3) * 10.0) * 1000
        min_msz = min(s["i_msz"] for s in feeders.values())
        isl.append(dict(g=g, i_sc=i_sc, i_load=i_load, min_msz=min_msz))
    return pd.DataFrame(rows), pd.DataFrame(post_rows), pd.DataFrame(in_rows), isl


# ════════════════════════════════════════════════════════════════════════════
# Фігури: схема, карта OSM, SHAP
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 6. ВІЗУАЛІЗАЦІЯ
# ════════════════════════════════════════════════════════════════════════════
PHASES = ["1. Норма", "2. Одразу після аварії", "3. Після перемикань"]
STATUS_STYLE = {
    "ok": ("Живиться нормально", "#2e7d32"),
    "restored": ("Відновлено перемиканням", "#1e88e5"),
    "island": ("Острівний режим (АБ+ВДЕ)", "#00acc1"),
    "shed": ("Відключено (ГОП)", "#fb8c00"),
    "lost": ("Без живлення", "#757575"),
    "failed": ("Пошкоджено", "#d32f2f"),
}
VOLT_SCALE = [[0, "#d32f2f"], [0.25, "#fb8c00"], [0.40, "#43a047"], [0.65, "#43a047"], [1, "#d32f2f"]]


def ratio_color(r):
    return "#d32f2f" if r > 1.0 else "#fb8c00" if r > 0.8 else "#fbc02d" if r > 0.6 else "#43a047"


def phase_data(res, phase):
    fn = res["f_nodes"]
    if phase == PHASES[0]:
        return res["normal"], res["closed0"], {n: "ok" for n in TPS}, res["topo_n"]
    if phase == PHASES[1]:
        stt, topo = res["post"], res["topo_p"]
        return stt, res["closed_f"], {n: ("failed" if n in fn else "lost" if n not in topo["energized"] else "ok")
                                      for n in TPS}, topo
    return res["final"], res["closed_fin"], res["status"], res["topo_f"]


def draw_map(res, phase, color_by):
    stt, closed, status, topo = phase_data(res, phase)
    fig = go.Figure()
    mx, my, mt = [], [], []
    for eid, e in EDGES.items():
        a, b = NODES[e["u"]], NODES[e["v"]]
        r = stt["ratio"].get(eid)
        if eid in res["f_edges"]:
            color, dash, w, txt = "#000000", "dot", 4, f"{eid}<br>ПОШКОДЖЕНО"
        elif eid not in closed:
            color, dash, w, txt = "#9aa0a6", "dash", 1.5, f"{eid}<br>Розімкнена перемичка (ліміт {e['imax'] * 1000:.0f} А)"
        elif r is None:
            color, dash, w, txt = "#cfcfcf", "solid", 2, f"{eid}<br>Без напруги"
        else:
            color, dash, w = ratio_color(r), "solid", 2 + 4 * min(r, 1.2)
            txt = (f"{eid}<br>P={stt['p_from'][eid]:+.2f} МВт, Q={stt['q_from'][eid]:+.2f} МВАр<br>"
                   f"I={stt['i_a'][eid]:.0f} А / {e['imax'] * 1000:.0f} А ({r:.0%})<br>"
                   f"Втрати {stt['pl_kw'][eid]:.1f} кВт")
        fig.add_trace(go.Scatter(x=[a["x"], b["x"]], y=[a["y"], b["y"]], mode="lines",
                                 line=dict(color=color, width=w, dash=dash), hoverinfo="skip", showlegend=False))
        mx.append((a["x"] + b["x"]) / 2)
        my.append((a["y"] + b["y"]) / 2)
        mt.append(txt)
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", hovertext=mt, hoverinfo="text",
                             marker=dict(size=12, color="rgba(0,0,0,0.01)"), showlegend=False))

    def hover(n, label):
        v = stt["vm"].get(n)
        return (f"{n}<br>{label}<br>Навантаження: {NODES[n]['load'] * res['scale']:.2f} МВт"
                f"<br>Напруга: {('%.3f о.е. (%.2f кВ)' % (v, v * 10)) if v else '—'}"
                f"<br>Споживач: {NODES[n]['ctype']}<br>Споживачів: {NODES[n]['cons']}<br>Пріоритет: {PRIO_TXT[NODES[n]['prio']]}")

    if color_by == "Напруга":
        en = [n for n in TPS if status[n] in ("ok", "restored", "island") and n in stt["vm"]]
        if en:
            fig.add_trace(go.Scatter(
                x=[NODES[n]["x"] for n in en], y=[NODES[n]["y"] for n in en], mode="markers+text", name="Напруга, о.е.",
                text=[n.replace("ТП-", "") for n in en], textposition="top center", textfont=dict(size=10),
                marker=dict(size=17, color=[stt["vm"][n] for n in en], colorscale=VOLT_SCALE, cmin=0.90, cmax=1.10,
                            colorbar=dict(title="U, о.е.", thickness=12, len=0.6),
                            line=dict(color="white", width=1.5)),
                hovertext=[hover(n, STATUS_STYLE[status[n]][0]) for n in en], hoverinfo="text", showlegend=False))
        groups = ("shed", "lost", "failed")
    else:
        groups = tuple(STATUS_STYLE)
    for key in groups:
        names = [n for n in TPS if status[n] == key]
        if not names:
            continue
        label, color = STATUS_STYLE[key]
        fig.add_trace(go.Scatter(
            x=[NODES[n]["x"] for n in names], y=[NODES[n]["y"] for n in names], mode="markers+text", name=label,
            text=[n.replace("ТП-", "") for n in names], textposition="top center", textfont=dict(size=10),
            marker=dict(size=17, color=color, line=dict(color="white", width=1.5)),
            hovertext=[hover(n, label) for n in names], hoverinfo="text"))
    fig.add_trace(go.Scatter(
        x=[NODES[s]["x"] for s in SOURCES], y=[NODES[s]["y"] for s in SOURCES], mode="markers+text",
        name="Джерело (ПС)", text=SOURCES, textposition="bottom center",
        marker=dict(size=26, symbol="square", line=dict(color="white", width=2),
                    color=["#d32f2f" if s in res["f_nodes"] else "#6a1b9a" for s in SOURCES]),
        hovertext=[f"{s}<br>Завантаження трансформатора: {stt['src'].get(s, 0):.0%}<br>"
                   f"P={stt['p_src'].get(s, 0):+.2f} МВт, Q={stt['q_src'].get(s, 0):+.2f} МВАр" for s in SOURCES],
        hoverinfo="text"))
    if res["pv_on"]:
        pvn = [n for n in TPS if NODES[n]["pv"] > 0]
        bn = [n for n in TPS if NODES[n]["bess_mw"] > 0]
        fig.add_trace(go.Scatter(x=[NODES[n]["x"] + 0.17 for n in pvn], y=[NODES[n]["y"] + 0.13 for n in pvn],
                                 mode="markers", name="PV (☀)", marker=dict(size=11, symbol="star", color="#ffb300",
                                 line=dict(color="#6d4c00", width=1)),
                                 hovertext=[f"PV {n}: {NODES[n]['pv']} МВт" for n in pvn], hoverinfo="text"))
        fig.add_trace(go.Scatter(x=[NODES[n]["x"] + 0.17 for n in bn], y=[NODES[n]["y"] - 0.13 for n in bn],
                                 mode="markers", name="АБ (🔋)", marker=dict(size=11, symbol="hexagon", color="#00897b",
                                 line=dict(color="white", width=1)),
                                 hovertext=[f"АБ {n}: {NODES[n]['bess_mw']} МВт / {NODES[n]['bess_mwh']} МВт·год"
                                            for n in bn], hoverinfo="text"))
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x"), legend=dict(orientation="h", y=-0.05),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _map_cls():
    """plotly >= 5.24: Scattermap (layout.map); старіші версії: Scattermapbox (layout.mapbox)."""
    if hasattr(go, "Scattermap"):
        return go.Scattermap, "map"
    return go.Scattermapbox, "mapbox"


def _dashed(lon1, lat1, lon2, lat2, n=9):
    xs, ys = [], []
    for i in range(0, n, 2):
        a, b = i / n, min((i + 1) / n, 1.0)
        xs += [lon1 + (lon2 - lon1) * a, lon1 + (lon2 - lon1) * b, None]
        ys += [lat1 + (lat2 - lat1) * a, lat1 + (lat2 - lat1) * b, None]
    return xs, ys


def hover_txt(n, label, stt, scale):
    v = stt["vm"].get(n)
    return (f"{n} · {NODES[n]['ctype']}<br>{label}<br>Навантаження: {NODES[n]['load'] * scale:.2f} МВт"
            f"<br>Напруга: {('%.3f о.е. (%.2f кВ)' % (v, v * 10)) if v else '—'}"
            f"<br>Споживачів: {NODES[n]['cons']}<br>Пріоритет: {PRIO_TXT[NODES[n]['prio']]}")


def draw_geo_map(res, phase, color_by, center, district):
    """Та сама мережа, але на карті OpenStreetMap навколо обраного району Вінницької області."""
    Sc, mkey = _map_cls()
    stt, closed, status, topo = phase_data(res, phase)
    fig = go.Figure()
    pos = {n: node_lonlat(n, center) for n in NODES}
    mlon, mlat, mt = [], [], []
    for eid, e in EDGES.items():
        (x1, y1), (x2, y2) = pos[e["u"]], pos[e["v"]]
        r = stt["ratio"].get(eid)
        dash = False
        if eid in res["f_edges"]:
            color, w, txt = "#000000", 6, f"{eid}<br>ПОШКОДЖЕНО"
        elif eid not in closed:
            color, w, dash = "#616161", 3, True
            txt = f"{eid}<br>Розімкнена перемичка (ліміт {e['imax'] * 1000:.0f} А)"
        elif r is None:
            color, w, txt = "#bdbdbd", 3, f"{eid}<br>Без напруги"
        else:
            color, w = ratio_color(r), 3 + 4 * min(r, 1.2)
            txt = (f"{eid}<br>P={stt['p_from'][eid]:+.2f} МВт, Q={stt['q_from'][eid]:+.2f} МВАр<br>"
                   f"I={stt['i_a'][eid]:.0f} А / {e['imax'] * 1000:.0f} А ({r:.0%})<br>"
                   f"Втрати {stt['pl_kw'][eid]:.1f} кВт")
        xs, ys = _dashed(x1, y1, x2, y2) if dash else ([x1, x2], [y1, y2])
        fig.add_trace(Sc(lon=xs, lat=ys, mode="lines", line=dict(color=color, width=w),
                         hoverinfo="skip", showlegend=False))
        mlon.append((x1 + x2) / 2)
        mlat.append((y1 + y2) / 2)
        mt.append(txt)
    fig.add_trace(Sc(lon=mlon, lat=mlat, mode="markers", marker=dict(size=14, opacity=0.01),
                     hovertext=mt, hoverinfo="text", showlegend=False))
    if color_by == "Напруга":
        en = [n for n in TPS if status[n] in ("ok", "restored", "island") and n in stt["vm"]]
        if en:
            fig.add_trace(Sc(
                lon=[pos[n][0] for n in en], lat=[pos[n][1] for n in en], mode="markers+text",
                text=[n.replace("ТП-", "") for n in en], textposition="top center", name="Напруга, о.е.",
                marker=dict(size=17, color=[stt["vm"][n] for n in en], colorscale=VOLT_SCALE, cmin=0.90, cmax=1.10,
                            colorbar=dict(title="U, о.е.", thickness=12, len=0.5)),
                hovertext=[hover_txt(n, STATUS_STYLE[status[n]][0], stt, res["scale"]) for n in en],
                hoverinfo="text", showlegend=False))
        groups = ("shed", "lost", "failed")
    else:
        groups = tuple(STATUS_STYLE)
    for key in groups:
        names = [n for n in TPS if status[n] == key]
        if not names:
            continue
        label, color = STATUS_STYLE[key]
        fig.add_trace(Sc(
            lon=[pos[n][0] for n in names], lat=[pos[n][1] for n in names], mode="markers+text", name=label,
            text=[n.replace("ТП-", "") for n in names], textposition="top center",
            marker=dict(size=17, color=color),
            hovertext=[hover_txt(n, label, stt, res["scale"]) for n in names], hoverinfo="text"))
    fig.add_trace(Sc(
        lon=[pos[s][0] for s in SOURCES], lat=[pos[s][1] for s in SOURCES], mode="markers+text",
        name="Джерело (ПС 110/10 кВ)", text=SOURCES, textposition="bottom center",
        marker=dict(size=24, color=["#d32f2f" if s in res["f_nodes"] else "#6a1b9a" for s in SOURCES]),
        hovertext=[f"{s} · умовна ПС 110/10 кВ, {district}<br>Завантаження трансформатора: "
                   f"{stt['src'].get(s, 0):.0%}<br>P={stt['p_src'].get(s, 0):+.2f} МВт, "
                   f"Q={stt['q_src'].get(s, 0):+.2f} МВАр" for s in SOURCES], hoverinfo="text"))
    if res["pv_on"]:
        pvn = [n for n in TPS if NODES[n]["pv"] > 0]
        bn = [n for n in TPS if NODES[n]["bess_mw"] > 0]
        fig.add_trace(Sc(lon=[pos[n][0] + 0.004 for n in pvn], lat=[pos[n][1] + 0.003 for n in pvn], mode="markers",
                         name="PV (☀)", marker=dict(size=11, color="#ffb300"),
                         hovertext=[f"PV {n}: {NODES[n]['pv']} МВт" for n in pvn], hoverinfo="text"))
        fig.add_trace(Sc(lon=[pos[n][0] + 0.004 for n in bn], lat=[pos[n][1] - 0.003 for n in bn], mode="markers",
                         name="АБ (🔋)", marker=dict(size=11, color="#00897b"),
                         hovertext=[f"АБ {n}: {NODES[n]['bess_mw']} МВт / {NODES[n]['bess_mwh']} МВт·год"
                                    for n in bn], hoverinfo="text"))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.02),
                      **{mkey: dict(style="open-street-map", center=dict(lat=center[0], lon=center[1]), zoom=10.2)})
    return fig


def voltage_profile(res, phase, p):
    stt, _, status, topo = phase_data(res, phase)
    fig = go.Figure()
    palette = {"ПС-1": "#6a1b9a", "ПС-2": "#1565c0"}
    seen = set()
    for n in topo["order"]:
        par, eid = topo["parent"][n]
        if par is None or n not in stt["vm"] or par not in stt["vm"]:
            continue
        r = topo["root"][n]
        fig.add_trace(go.Scatter(x=[topo["dist"][par], topo["dist"][n]], y=[stt["vm"][par], stt["vm"][n]],
                                 mode="lines+markers", line=dict(color=palette.get(r, "#00897b"), width=2),
                                 marker=dict(size=6), name=f"від {r}", legendgroup=r, showlegend=r not in seen,
                                 hovertext=[par, n], hoverinfo="text+y"))
        seen.add(r)
    fig.add_hline(y=p.vmin, line=dict(color="#d32f2f", dash="dash"), annotation_text="U мін")
    fig.add_hline(y=p.vmax, line=dict(color="#d32f2f", dash="dash"), annotation_text="U макс")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Відстань від джерела, км",
                      yaxis_title="U, о.е.", title="Профіль напруги вздовж фідерів")
    return fig




def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_waterfall(e):
    """SHAP waterfall для одного активу (e - shap.Explanation у відсоткових пунктах)."""
    import shap
    plt = _mpl()
    plt.close("all")
    shap.plots.waterfall(e, max_display=10, show=False)
    f = plt.gcf()
    f.set_size_inches(9, 5)
    return f


def fig_summary(sv, smp, names, bar=False):
    """SHAP summary (beeswarm або bar) за навчальною вибіркою."""
    import shap
    plt = _mpl()
    plt.close("all")
    shap.summary_plot(sv.values, smp.values, feature_names=names, plot_type="bar" if bar else "dot",
                      show=False, max_display=11)
    f = plt.gcf()
    f.set_size_inches(9, 5)
    return f


def close_fig(f):
    _mpl().close(f)


# ════════════════════════════════════════════════════════════════════════════
# Прогноз відмов, SHAP, ризик
# ════════════════════════════════════════════════════════════════════════════
try:
    import shap
    HAS_SHAP = True
except Exception:                                   # noqa: BLE001
    HAS_SHAP = False

# ════════════════════════════════════════════════════════════════════════════
# 7. PREDICTIVE MAINTENANCE
# ════════════════════════════════════════════════════════════════════════════
FEATURES = ["age", "loading_mean", "loading_peak", "overload_hours", "cold_hours",
            "prior_faults", "years_since_maint", "condition_idx", "is_tp", "is_ps", "is_line"]
FEAT_UA = {
    "age": "Вік, років", "loading_mean": "Середнє завантаження", "loading_peak": "Пікове завантаження",
    "overload_hours": "Годин перевантаження/рік", "cold_hours": "Годин холодного режиму/рік",
    "prior_faults": "Відмов за 3 роки", "years_since_maint": "Років від останнього ТО",
    "condition_idx": "Індекс стану (ХАРГ/ІЧ/ЧР)", "is_tp": "Тип: ТП", "is_ps": "Тип: ПС", "is_line": "Тип: лінія",
}
CATS = [(0.25, "Критичний"), (0.15, "Високий"), (0.08, "Середній"), (0.0, "Низький")]
CAT_COLOR = {"Критичний": "#d32f2f", "Високий": "#fb8c00", "Середній": "#fbc02d", "Низький": "#43a047"}
RECO = {
    "Критичний": "Позапланова діагностика (ХАРГ, тепловізор, ЧР) + план заміни; підготувати резервну схему",
    "Високий": "Діагностика протягом місяця; знизити навантаження, перевірити резервування",
    "Середній": "Включити в найближче ТО; контроль температури/навантаження",
    "Низький": "Стандартний моніторинг",
}


def category(p_year):
    for thr, name in CATS:
        if p_year >= thr:
            return name
    return "Низький"


@lru_cache(maxsize=4)
def make_training(n=10000, seed=7):
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, size=n, p=[0.7, 0.1, 0.2])
    age = rng.uniform(2, 50, n)
    lp = np.clip(rng.normal(0.75, 0.22, n), 0.25, 1.4)
    lm = lp * rng.uniform(0.6, 0.8, n)
    oh = np.where(lp > 0.95, (lp - 0.95) * 2500 * rng.uniform(0.5, 1.5, n), 0.0)
    cold = rng.uniform(0, 700, n)
    pf = rng.poisson(0.15 + age / 40)
    ysm = rng.uniform(0, 12, n)
    cond = np.clip(95 - 0.9 * age + rng.normal(0, 10, n), 10, 100)
    logit = (-5.4 + 0.045 * age + 2.0 * np.maximum(0, lp - 0.8) + 0.0007 * oh + 0.0011 * cold
             + 0.38 * pf + 0.10 * ysm - 0.028 * (cond - 60)
             + np.where(kind == 1, -0.4, 0) + np.where(kind == 2, 0.1, 0))
    failed = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(dict(age=age, loading_mean=lm, loading_peak=lp, overload_hours=oh,
                             cold_hours=cold, prior_faults=pf, years_since_maint=ysm,
                             condition_idx=cond, is_tp=(kind == 0).astype(int),
                             is_ps=(kind == 1).astype(int), is_line=(kind == 2).astype(int),
                             failed=failed.astype(int)))


def _fit_model(df):
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["failed"], test_size=0.25,
                                          random_state=1, stratify=df["failed"])
    mk = lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, random_state=0)
    auc = roc_auc_score(yte, mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    model = mk().fit(df[FEATURES], df["failed"])
    return model, auc, pd.Series(model.feature_importances_, index=FEATURES), df[FEATURES].median()


def static_registry():
    rng = np.random.default_rng(11)
    rows = [(n, "ps", int(rng.integers(12, 42))) for n in SOURCES]
    rows += [(n, "tp", int(rng.integers(4, 48))) for n in TPS]
    rows += [(eid, "line", int(rng.integers(3, 52))) for eid, e in EDGES.items() if not e["tie"]]
    out = []
    for id_, kind, age in rows:
        out.append(dict(id=id_, kind=kind, age=age, prior_faults=int(rng.poisson(0.1 + age / 45)),
                        years_since_maint=round(float(rng.uniform(0, 10)), 1),
                        condition_idx=float(np.clip(95 - 0.9 * age + rng.normal(0, 9), 12, 99)),
                        u=float(rng.uniform(0.2, 0.8))))
    return pd.DataFrame(out)


STATIC_DF = static_registry()


def registry_features(P, growth, winter_days, tel=None):
    """Завантаження активів беремо з pandapower (номінальний режим без ВДЕ), далі масштабуємо.
    tel - узагальнена live-телеметрія {актив: dict(peak, mean, frac_over, n)}: перекриває розрахункові значення."""
    p0 = Params(*P)._replace(der=0)
    net = new_net(p0)
    topo = topology(set(CLOSED0), E0, E0)
    stt = pp_state(net, p0, topo, set(CLOSED0), E0, E0, 1.0, set(), set(), ())
    nominal = []
    for _, r in STATIC_DF.iterrows():
        if r["kind"] == "tp":
            nominal.append(NODES[r["id"]]["load"] / (NODES[r["id"]]["kva"] * 0.95 / 1000))
        elif r["kind"] == "ps":
            nominal.append(stt["src"].get(r["id"], 0.0))
        else:
            nominal.append(stt["ratio"].get(r["id"], 0.0))
    if p0.prof:                                     # ML-профіль на завтра
        pf = np.array(p0.prof)
        peak_k, prof = float(pf.max()), pf / pf.max()
    else:
        peak_k, prof = temp_factor(p0.temp, p0.sens), np.array(HOURLY)
    peak = np.array(nominal) * peak_k * (1 + growth)
    df = STATIC_DF.copy()
    df["loading_peak"] = peak
    df["loading_mean"] = peak * prof.mean()
    df["overload_hours"] = winter_days * ((peak[:, None] * prof[None, :]) > 1.0).sum(axis=1)
    df["cold_hours"] = (winter_days * 24 * df["u"]).round()
    df["from_telemetry"] = False
    if tel:
        for i, r in df.iterrows():
            t = tel.get(r["id"])
            if t and t["n"] >= 20:
                df.at[i, "loading_peak"] = t["peak"]
                df.at[i, "loading_mean"] = t["mean"]
                df.at[i, "overload_hours"] = t["frac_over"] * 24 * winter_days
                df.at[i, "from_telemetry"] = True
    for k in ("tp", "ps", "line"):
        df[f"is_{k}"] = (df["kind"] == k).astype(int)
    df["type"] = df["id"].map(elem_type)
    return df


def explain(model, row, ref):
    x0 = row[FEATURES].astype(float).to_frame().T
    p0 = model.predict_proba(x0)[0, 1]
    contrib = {}
    for f in FEATURES[:8]:
        x1 = x0.copy()
        x1[f] = ref[f]
        contrib[FEAT_UA[f]] = p0 - model.predict_proba(x1)[0, 1]
    return pd.Series(contrib).sort_values()


# ════════════════════════════════════════════════════════════════════════════
# 7c. EXPLAINABLE AI: SHAP
# ════════════════════════════════════════════════════════════════════════════
UA_NAMES = [FEAT_UA[f] for f in FEATURES]


def _build_shap(df):
    model = get_model(df)[0]
    bg = df[FEATURES].sample(min(100, len(df)), random_state=0)
    smp = df[FEATURES].sample(min(1000, len(df)), random_state=1)
    ex_p = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="probability")
    sv = shap.TreeExplainer(model)(smp)                  # log-odds, швидко - для summary-графіків
    return ex_p, smp, sv


def shap_row(ex_p, x_row):
    """SHAP для одного активу, у відсоткових пунктах річної імовірності відмови."""
    e = ex_p(x_row)
    return shap.Explanation(values=e.values[0] * 100, base_values=float(np.ravel(e.base_values)[0]) * 100,
                            data=x_row.values[0], feature_names=UA_NAMES)


# ── кеші моделей (ключ - вміст навчального DataFrame) ──
_MODEL_CACHE, _SHAP_CACHE = {}, {}


def _remember(store, key, build):
    if key not in store:
        if len(store) >= 4:
            store.pop(next(iter(store)))
        store[key] = build()
    return store[key]


def get_model(df):
    """(модель, AUC, важливість ознак, медіани) - кешується за вмістом df."""
    return _remember(_MODEL_CACHE, df_key(df), lambda: _fit_model(df))


def get_shap(df):
    """(пояснювач у ймовірностях, вибірка, SHAP-значення log-odds) - кешується за вмістом df."""
    return _remember(_SHAP_CACHE, df_key(df), lambda: _build_shap(df))


def training_from_csv(raw):
    """Користувацька історія відмов (CSV bytes) -> DataFrame для навчання; ValueError, якщо формат не підходить."""
    import io
    udf = pd.read_csv(io.BytesIO(raw))
    if not set(FEATURES + ["failed"]).issubset(udf.columns) or udf["failed"].nunique() != 2:
        raise ValueError("у файлі бракує потрібних стовпців або немає обох класів у failed")
    return udf[FEATURES + ["failed"]].dropna()


def score_registry(model, reg, horizon):
    """Додає p_year, p_h (імовірність за horizon міс.), категорію та рекомендацію."""
    reg = reg.copy()
    reg["p_year"] = model.predict_proba(reg[FEATURES].astype(float))[:, 1]
    reg["p_h"] = 1 - (1 - reg["p_year"]) ** (horizon / 12.0)
    reg["cat"] = reg["p_year"].map(category)
    reg["reco"] = reg["cat"].map(RECO)
    return reg


def risk_register(reg, scan_df):
    """Ризик = P × наслідки: об'єднання реєстру активів з результатами N-1."""
    mg = reg.merge(scan_df[["id", "ens_mwh", "unsup_mw", "cust", "saidi"]], on="id")
    mg["exp_ens"] = mg["p_h"] * mg["ens_mwh"]
    mg["exp_saidi"] = mg["p_h"] * mg["saidi"]
    return mg.sort_values("exp_ens", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# ML-прогноз навантаження (Open-Meteo)
# ════════════════════════════════════════════════════════════════════════════
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:                                   # noqa: BLE001
    HAS_LGB = False

def _now_local():
    """Поточний час у Києві (naive Timestamp); запасний варіант без tzdata: UTC+3."""
    try:
        return pd.Timestamp(datetime.now(ZoneInfo("Europe/Kyiv")).replace(tzinfo=None))
    except Exception:                                # noqa: BLE001
        return pd.Timestamp((datetime.now(timezone.utc) + timedelta(hours=3)).replace(tzinfo=None))




# ════════════════════════════════════════════════════════════════════════════
# 7b. ML-ПРОГНОЗ НАВАНТАЖЕННЯ НА ЗАВТРА (LightGBM + прогноз погоди Open-Meteo)
# ════════════════════════════════════════════════════════════════════════════
OM_FC_URL = "https://api.open-meteo.com/v1/forecast"
OM_AR_URL = "https://archive-api.open-meteo.com/v1/archive"
FC_FEATS = ["hour", "hour_sin", "hour_cos", "dow", "is_wkend", "temp", "t3", "t24", "cold", "heat"]
TOTAL_BASE_MW = sum(NODES[n]["load"] for n in TPS)       # навантаження мережі при коефіцієнті = 1.0


def _om_get(url, params, timeout=8):
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": "grid-twin-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _om_series(js, key="temperature_2m"):
    h = js["hourly"]
    return pd.Series(h[key], index=pd.to_datetime(h["time"]), dtype=float)


def synth_weather(start, end, seed=3):
    """Синтетична температура (запасний варіант, якщо Open-Meteo недоступний)."""
    idx = pd.date_range(start, end, freq="h")
    seasonal = 7.8 - 12.4 * np.cos(2 * np.pi * (idx.dayofyear.values - 15) / 365.0)   # клімат Вінниччини: січ ≈ -4.6, лип ≈ +20 °C
    daily = 4.0 * np.sin(2 * np.pi * (idx.hour.values - 9) / 24.0)
    rng = np.random.default_rng(seed)
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.985 * noise[i - 1] + rng.normal(0, 0.6)
    return pd.Series(seasonal + daily + noise, index=idx)


@ttl_cache(3600)
def get_weather(lat, lon, day_key):
    """Годинна температура: архів ERA5 (≈14 міс.) + прогноз на 3 доби; сонячна радіація - лише прогноз.
    Повертає (ряд T, джерело, часовий пояс, ряд радіації або None)."""
    today = pd.Timestamp(day_key)
    tz = "Europe/Kyiv"
    try:
        fcj = _om_get(OM_FC_URL, dict(latitude=lat, longitude=lon, hourly="temperature_2m,shortwave_radiation",
                                      past_days=10, forecast_days=3, timezone=tz))
        s_fc = _om_series(fcj)
        rad = _om_series(fcj, "shortwave_radiation") if "shortwave_radiation" in fcj["hourly"] else None
    except Exception as ex:                          # noqa: BLE001
        s = synth_weather(today - pd.Timedelta(days=420), today + pd.Timedelta(days=3))
        return s, f"синтетична погода (Open-Meteo недоступний: {type(ex).__name__})", tz, None
    try:
        arj = _om_get(OM_AR_URL, dict(latitude=lat, longitude=lon, hourly="temperature_2m", timezone=tz,
                                      start_date=(today - pd.Timedelta(days=420)).strftime("%Y-%m-%d"),
                                      end_date=(today - pd.Timedelta(days=11)).strftime("%Y-%m-%d")), timeout=20)
        s_ar, src = _om_series(arj), "Open-Meteo: архів ERA5 + прогноз"
    except Exception:                                # noqa: BLE001
        s_ar = synth_weather(s_fc.index[0] - pd.Timedelta(days=420), s_fc.index[0] - pd.Timedelta(hours=1))
        src = "Open-Meteo: прогноз (історія температур синтетична)"
    s = pd.concat([s_ar, s_fc])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.asfreq("h").interpolate(limit=12).ffill().bfill()
    return s, src, tz, rad


def _true_load(idx, temp):
    """Синтетична «істинна» поведінка: профіль доби × вихідні × холод/спека (для демо-історії)."""
    h, dow = idx.hour.values, idx.dayofweek.values
    base = np.asarray(HOURLY)[h]
    wk = np.where(dow == 5, 0.93, np.where(dow == 6, 0.88, 1.0))
    morning = np.where((dow >= 5) & (h >= 6) & (h <= 10), 0.94, 1.0)
    return (base * wk * morning * (1 + 0.012 * np.maximum(0, 5 - temp))
            * (1 + 0.008 * np.maximum(0, temp - 24)))


def _fc_features(ser):
    d = pd.DataFrame({"temp": ser})
    h = ser.index.hour
    d["hour"] = h
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["dow"] = ser.index.dayofweek
    d["is_wkend"] = (d["dow"] >= 5).astype(int)
    d["t3"] = ser.rolling(3, min_periods=1).mean()
    d["t24"] = ser.rolling(24, min_periods=1).mean()
    d["cold"] = np.maximum(0, 5 - ser)
    d["heat"] = np.maximum(0, ser - 24)
    return d


def _make_reg():
    if HAS_LGB:
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=0,
                                 verbose=-1), "LightGBM"
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0), "HistGradientBoosting (sklearn)"


def _read_hist(raw, hist_idx):
    """CSV (ts, load_mw; крок будь-який) -> коефіцієнт навантаження на сітці годин hist_idx."""
    dfh = pd.read_csv(io.BytesIO(raw))
    cols = {c.lower().strip(): c for c in dfh.columns}
    tcol = next((cols[k] for k in ("ts", "timestamp", "time", "datetime") if k in cols), None)
    lcol = next((cols[k] for k in ("load_mw", "load", "p_mw") if k in cols), None)
    if tcol is None or lcol is None:
        raise ValueError("потрібні стовпці ts та load_mw")
    s = pd.Series(pd.to_numeric(dfh[lcol], errors="coerce").values, index=pd.to_datetime(dfh[tcol]))
    s = s.dropna().sort_index().resample("h").mean() / TOTAL_BASE_MW
    y = s.reindex(hist_idx).dropna()
    if len(y) < 24 * 21:
        raise ValueError("потрібно щонайменше 21 добу даних, що перетинаються з погодою")
    return y


@ttl_cache(3600)
def forecast_load(lat, lon, day_key, hist_bytes=None):
    ser, w_src, tz, rad = get_weather(lat, lon, day_key)
    now_loc = _now_local()
    t_idx = pd.date_range(now_loc.normalize() + pd.Timedelta(days=1), periods=24, freq="h")
    if not set(t_idx).issubset(set(ser.index)):
        raise ValueError("у погодних даних немає всіх 24 годин завтрашньої доби")
    feats = _fc_features(ser)
    hist_idx = ser.index[ser.index <= now_loc.floor("h")]
    y, d_src = None, "синтетична історія навантаження (демо)"
    if hist_bytes:
        try:
            y = _read_hist(hist_bytes, hist_idx)
            d_src = f"ваш CSV ({len(y)} год)"
        except Exception as ex:                      # noqa: BLE001
            d_src = f"синтетична історія (CSV не підійшов: {ex})"
    if y is None:
        rng = np.random.default_rng(5)
        noise = np.zeros(len(hist_idx))
        for i in range(1, len(noise)):
            noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.012)
        y = pd.Series(_true_load(hist_idx, ser.loc[hist_idx].values) * (1 + noise), index=hist_idx)
    X = feats.loc[y.index, FC_FEATS]
    cut = y.index[-1] - pd.Timedelta(days=14)
    tr, te = y.index <= cut, y.index > cut
    m_cv, name = _make_reg()
    m_cv.fit(X[tr], y[tr])
    pred_te = m_cv.predict(X[te])
    yt = y[te].values
    stat_te = (np.asarray(HOURLY)[X.loc[te, "hour"].astype(int).values]
               * (1 + 0.012 * np.maximum(0, 5 - X.loc[te, "temp"].values)))
    m, _ = _make_reg()
    m.fit(X, y)
    Xt = feats.loc[t_idx, FC_FEATS]
    prof = np.clip(m.predict(Xt), 0.2, 2.5)
    stat = np.asarray(HOURLY) * (1 + 0.012 * np.maximum(0, 5 - Xt["temp"].values))
    k = min(96, len(yt))
    if rad is not None and set(t_idx).issubset(set(rad.index)):
        pv_f = np.clip(rad.reindex(t_idx).fillna(0).values / 1000.0 * 0.85, 0, 1)     # PR ≈ 0.85
        pv_src = "Open-Meteo: сонячна радіація (shortwave_radiation)"
    else:
        pv_f = np.array([pv_profile(h) for h in range(24)])
        pv_src = "типова зимова крива (прогноз радіації недоступний)"
    return dict(pv_frac=[float(v) for v in pv_f], pv_src=pv_src,
                
        date=t_idx[0].strftime("%d.%m.%Y"), tz=tz, weather_src=w_src, data_src=d_src, model=name, n_train=int(len(y)),
        mae_ml=float(np.mean(np.abs(pred_te - yt))), mae_static=float(np.mean(np.abs(stat_te - yt))),
        mape_ml=float(100 * np.mean(np.abs(pred_te - yt) / np.maximum(yt, 1e-6))),
        profile=[float(v) for v in prof], static_profile=[float(v) for v in stat],
        temps=[float(v) for v in Xt["temp"].values],
        bt=dict(ts=[t.strftime("%Y-%m-%d %H:%M") for t in y[te].index[-k:]], actual=[float(v) for v in yt[-k:]],
                ml=[float(v) for v in pred_te[-k:]], static=[float(v) for v in stat_te[-k:]]))


@ttl_cache(3600)
def oblast_weather(day_key):
    """Умови на завтра по районах: T мін/макс, пік навантаження типової мережі, потенціал PV."""
    names = list(DISTRICTS)
    t_idx = pd.date_range(_now_local().normalize() + pd.Timedelta(days=1), periods=24, freq="h")
    series = []
    try:
        js = _om_get(OM_FC_URL, dict(latitude=",".join(str(DISTRICTS[n]["lat"]) for n in names),
                                     longitude=",".join(str(DISTRICTS[n]["lon"]) for n in names),
                                     hourly="temperature_2m,shortwave_radiation", forecast_days=3,
                                     timezone="Europe/Kyiv"), timeout=15)
        js = [js] if isinstance(js, dict) else js
        if len(js) != len(names):
            raise ValueError("несподівана відповідь API")
        for j in js:
            series.append((_om_series(j).reindex(t_idx).values,
                           _om_series(j, "shortwave_radiation").reindex(t_idx).fillna(0).values))
        src = "Open-Meteo"
    except Exception as ex:                          # noqa: BLE001
        series = []
        base = synth_weather(t_idx[0] - pd.Timedelta(days=2), t_idx[-1] + pd.Timedelta(days=1)).reindex(t_idx).values
        for i, n in enumerate(names):
            off = (49.0 - DISTRICTS[n]["lat"]) * 0.6 + 0.2 * (i % 3)
            series.append((base + off, np.array([pv_profile(h) / 0.7 * 400.0 for h in range(24)])))
        src = f"синтетичні дані (Open-Meteo недоступний: {type(ex).__name__})"
    rows = []
    for n, (T, R) in zip(names, series):
        f = np.asarray(HOURLY) * (1 + 0.012 * np.maximum(0, 5 - T))
        rows.append(dict(district=n, lat=DISTRICTS[n]["lat"], lon=DISTRICTS[n]["lon"], tmin=float(np.nanmin(T)),
                         tmax=float(np.nanmax(T)), peak_factor=float(f.max()), peak_mw=float(f.max() * TOTAL_BASE_MW),
                         pv_yield=float(np.nansum(R) / 1000.0 * 0.85)))
    return pd.DataFrame(rows), src, t_idx[0].strftime("%d.%m.%Y")


# ════════════════════════════════════════════════════════════════════════════
# Потік телеметрії (вбудований / MQTT)
# ════════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════════
# PDF-звіт
# ════════════════════════════════════════════════════════════════════════════
def _need_fpdf():
    if not HAS_FPDF:
        raise RuntimeError("Не встановлено fpdf2 (додайте fpdf2>=2.7 у requirements.txt)")


def _font_paths():
    d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    reg, bold = os.path.join(d, "DejaVuSans.ttf"), os.path.join(d, "DejaVuSans-Bold.ttf")
    if os.path.exists(reg) and os.path.exists(bold):
        return reg, bold
    raise RuntimeError("Не знайдено шрифт DejaVuSans (постачається разом з matplotlib)")


def _png(fig, dpi=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


# ───────────────────────────── діаграми (matplotlib) ─────────────────────────────
def chart_network(res):
    """Схема мережі після перемикань: колір ліній - завантаження, колір вузлів - статус."""
    fin, closed = res["final"], res["closed_fin"]
    fig = Figure(figsize=(9, 4.8))
    ax = fig.subplots()
    for eid, e in EDGES.items():
        a, b = NODES[e["u"]], NODES[e["v"]]
        r = fin["ratio"].get(eid)
        if eid in res["f_edges"]:
            color, lw, ls = "black", 3.0, ":"
        elif eid not in closed:
            color, lw, ls = "#9aa0a6", 1.2, "--"
        elif r is None:
            color, lw, ls = "#cfcfcf", 1.5, "-"
        else:
            color, lw, ls = ratio_color(r), 1.5 + 3.0 * min(r, 1.2), "-"
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="round")
    for key, (label, color) in STATUS_STYLE.items():
        names = [n for n in TPS if res["status"][n] == key]
        if not names:
            continue
        ax.scatter([NODES[n]["x"] for n in names], [NODES[n]["y"] for n in names], s=130, c=color,
                   edgecolors="white", linewidths=1.2, zorder=3, label=label)
        for n in names:
            ax.annotate(n.replace("ТП-", ""), (NODES[n]["x"], NODES[n]["y"]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6)
    ax.scatter([NODES[s]["x"] for s in SOURCES], [NODES[s]["y"] for s in SOURCES], marker="s", s=320,
               c=["#d32f2f" if s in res["f_nodes"] else "#6a1b9a" for s in SOURCES], zorder=3, label="Джерело (ПС)")
    for s in SOURCES:
        ax.annotate(s, (NODES[s]["x"], NODES[s]["y"]), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=7)
    handles, labels = ax.get_legend_handles_labels()
    for color, text in (("#43a047", "лінія <60%"), ("#fbc02d", "60–80%"), ("#fb8c00", "80–100%"), ("#d32f2f", ">100%")):
        handles.append(Line2D([0], [0], color=color, lw=3))
        labels.append(text)
    handles.append(Line2D([0], [0], color="#9aa0a6", lw=1.5, ls="--"))
    labels.append("розімкнена перемичка")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5, frameon=False)
    return fig


def chart_voltage(res, vmin, vmax):
    fin, topo = res["final"], res["topo_f"]
    fig = Figure(figsize=(9, 3.0))
    ax = fig.subplots()
    palette, seen = {"ПС-1": "#6a1b9a", "ПС-2": "#1565c0"}, set()
    for n in topo["order"]:
        par, _ = topo["parent"][n]
        if par is None or n not in fin["vm"] or par not in fin["vm"]:
            continue
        r = topo["root"][n]
        ax.plot([topo["dist"][par], topo["dist"][n]], [fin["vm"][par], fin["vm"][n]], "-o", ms=3, lw=1.6,
                color=palette.get(r, "#00897b"), label=f"від {r}" if r not in seen else None)
        seen.add(r)
    ax.axhline(vmin, color="#d32f2f", ls="--", lw=1)
    ax.axhline(vmax, color="#d32f2f", ls="--", lw=1)
    ax.set_xlabel("Відстань від джерела, км", fontsize=8)
    ax.set_ylabel("U, о.е.", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return fig


def chart_risk(mg, horizon):
    fig = Figure(figsize=(9, 4.4))
    ax = fig.subplots()
    sizes = 40 + 360 * np.sqrt(mg["cust"] / max(mg["cust"].max(), 1))
    sc = ax.scatter(100 * mg["p_h"], mg["ens_mwh"], s=sizes, c=mg["exp_ens"], cmap="YlOrRd",
                    edgecolors="#555", linewidths=0.5, alpha=0.9)
    for r in mg.head(8).itertuples():
        ax.annotate(r.id, (100 * r.p_h, r.ens_mwh), textcoords="offset points", xytext=(4, 5), fontsize=6.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Очікуваний ENS, МВт·год", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(f"Імовірність відмови за {horizon} міс., %", fontsize=8)
    ax.set_ylabel("Наслідок аварії: ENS, МВт·год", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    return fig


# ───────────────────────────── PDF ─────────────────────────────
class _PDF(FPDF):
    def __init__(self, header_text):
        super().__init__(orientation="P", unit="mm", format="A4")
        reg, bold = _font_paths()
        self.add_font("DejaVu", "", reg)
        self.add_font("DejaVu", "B", bold)
        self.header_text = header_text
        self.set_margins(12, 14, 12)
        self.set_auto_page_break(True, margin=14)
        self.alias_nb_pages()

    def header(self):
        self.set_font("DejaVu", "", 7)
        self.set_text_color(120)
        self.cell(0, 4, self.header_text, align="R")
        self.ln(7)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-11)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(120)
        self.cell(0, 5, f"Стор. {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0)

    # ---- прості блоки ----
    def h1(self, text):
        self.set_font("DejaVu", "B", 15)
        self.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text):
        self.ln(2)
        self.set_font("DejaVu", "B", 11)
        self.set_fill_color(236, 239, 241)
        self.cell(0, 7, " " + text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def para(self, text, size=9, color=0):
        self.set_font("DejaVu", "", size)
        self.set_text_color(color)
        self.multi_cell(0, size * 0.52, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(1)

    def grid(self, header, rows, widths, size=7.5, align="LEFT"):
        self.set_font("DejaVu", "", size)
        with super().table(col_widths=widths, text_align=align, line_height=size * 0.5,
                           headings_style=FontFace(emphasis="BOLD", fill_color=(224, 224, 224))) as t:
            t.row([str(h) for h in header])
            for r in rows:
                t.row([str(c) for c in r])
        self.ln(2)

    def image_buf(self, buf, w=186):
        self.image(buf, w=w)
        self.ln(2)


def _f(v, fmt="{:.2f}", dash="—"):
    return dash if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _conclusion(res):
    if res["viol"] > 0:
        return (f"Після перемикань залишилось {res['viol']} порушень режиму (завантаження/напруга): потрібні додаткові "
                "заходи (мобільна генерація, додаткові обмеження).", (183, 28, 28))
    if res["unsup_mw"] <= 1e-9:
        return "Живлення всіх споживачів вдається відновити без порушень режиму.", (46, 125, 50)
    return "Частину споживачів неможливо повернути до завершення ремонту.", (230, 81, 0)


def render_report(ctx):
    """ctx: district, elements, P(Params), res(solve), profile_label, generated_at, [faults, post_df, in_df, isl],
    [mg (risk_register), horizon]. Повертає bytes PDF."""
    _need_fpdf()
    res, p, els = ctx["res"], ctx["p"], ctx["elements"]
    gen = ctx.get("generated_at") or datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf = _PDF(f"Цифровий двійник мережі 10 кВ · Вінницька область · {ctx['district']}")
    pdf.add_page()
    pdf.h1("Звіт диспетчера: аварійний сценарій")
    pdf.para(f"Район: {ctx['district']} · елементи, що вийшли з ладу: {', '.join(els)} · сформовано {gen}", 9, 90)

    text, color = _conclusion(res)
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(1)

    pdf.h2("1. Вихідні умови")
    pdf.grid(["Параметр", "Значення", "Параметр", "Значення"], [
        ("Година доби", f"{p.hour}:00", "Температура", f"{p.temp:+.0f} °C"),
        ("Профіль навантаження", ctx.get("profile_label", "статичний"), "Коеф. навантаження", f"{res['scale']:.2f}"),
        ("Допустима потужність ПС", f"{p.src_cap:.1f} МВт", "Допуск перевантаження", f"{100 * p.tol:.0f} %"),
        ("ВДЕ / АБ", "увімкнено" if p.der else "вимкнено", "Острівний режим", "дозволено" if p.island else "ні"),
        ("Час виявлення / комутації", f"{p.t_detect:.0f} / {p.t_switch:.0f} хв", "Тривалість ремонту", f"{p.repair_h:.1f} год"),
        ("Допустима напруга", f"{p.vmin:.2f}–{p.vmax:.2f} о.е.", "cos φ", f"{p.cosphi:.2f}"),
    ], (30, 28, 30, 28))

    fin = res["final"]
    pdf.h2("2. Ключові показники")
    pdf.grid(["Показник", "Значення", "Показник", "Значення"], [
        ("Загальне навантаження", f"{res['total_mw']:.1f} МВт", "Недовідпуск (ENS)", f"{res['ens']:.1f} МВт·год"),
        ("Без живлення після перемикань", f"{res['unsup_mw']:.2f} МВт ({100 * res['unsup_mw'] / res['total_mw']:.1f} %)",
         "SAIDI сценарію", f"{res['saidi']:.0f} хв/спож."),
        ("Споживачів без живлення", f"{res['cust_unrestored']}", "Зазнали переривання", f"{res['cust_cut']}"),
        ("Втрати в мережі", f"{res['losses'] * 1000:.0f} кВт", "Мін. напруга", f"{res['vmin_fin']:.3f} о.е."),
        ("Макс. завантаження", f"{res['max_line']:.0%}", "Порушень режиму", f"{res['viol']}"),
        ("Генерація PV", f"{fin['pv_mw']:.2f} МВт", "АБ (розряд +/заряд −)", f"{fin['bess_mw']:+.2f} МВт"),
    ], (36, 32, 28, 22))
    for g in res["gf"]:
        a = res["aut"].get(g, 1e9)
        n_isl = sum(1 for n in TPS if res["topo_f"]["root"].get(n) == g and n not in res["shed"])
        pdf.para(f"Острів від АБ {g}: {n_isl} ТП; автономність "
                 + ("більша за тривалість ремонту." if a >= p.repair_h else f"≈ {a:.1f} год (SoC {p.soc:.0%})."), 8.5)

    pdf.h2("3. Схема мережі після перемикань")
    pdf.image_buf(_png(chart_network(res)), 180)

    pdf.add_page()
    pdf.h2("4. Протокол перемикань")
    steps = [s for s in res["steps"]]
    pdf.grid(["№", "Час, хв", "Дія", "Об'єкт", "Обґрунтування"],
              [(s["№"], f"{s['Час, хв']:.0f}", s["Дія"], s["Об'єкт"], s["Обґрунтування"]) for s in steps],
              (6, 11, 44, 28, 62), size=7)
    if res["shed"]:
        pdf.para("Відключено за ГОП: " + ", ".join(f"{n} ({NODES[n]['ctype']})" for n in res["shed"]), 8.5)

    pdf.h2("5. Профіль напруги вздовж фідерів")
    pdf.image_buf(_png(chart_voltage(res, p.vmin, p.vmax)), 180)

    pdf.h2("6. Найбільш завантажені лінії")
    rows = []
    for eid, e in EDGES.items():
        if eid in fin["ratio"]:
            rows.append((eid, f"{e['imax'] * 1000:.0f}", 100 * res["post"]["ratio"].get(eid, 0.0), 100 * fin["ratio"][eid],
                         fin["i_a"][eid], fin["p_from"][eid], fin["pl_kw"][eid]))
    rows.sort(key=lambda r: -r[3])
    pdf.grid(["Лінія", "Ліміт, А", "Після аварії, %", "Після перемикань, %", "I, А", "P, МВт", "Втрати, кВт"],
              [(r[0], r[1], f"{r[2]:.0f}", f"{r[3]:.0f}", f"{r[4]:.0f}", f"{r[5]:+.2f}", f"{r[6]:.1f}") for r in rows[:10]],
              (34, 16, 20, 24, 14, 16, 18))

    if ctx.get("post_df") is not None:
        pdf.h2("7. Релейний захист після перемикань")
        post = ctx["post_df"]
        pdf.grid(["Фідер", "I роб., А", "I МСЗ, А", "Запас", "kч в кінці", "Оцінка"],
                  [(r["Фідер"], _f(r["I роб. після, А"], "{:.0f}"), _f(r["I МСЗ, А"], "{:.0f}"), _f(r["Запас МСЗ"]),
                    _f(r["kч у кінці"]), r["Оцінка"]) for _, r in post.iterrows()], (34, 16, 16, 14, 18, 62))
        if (post["lvl"] == 2).any():
            pdf.para("УВАГА: ризик неселективної роботи захисту на фідерах: "
                     + ", ".join(post.loc[post["lvl"] == 2, "Фідер"]) + ".", 9, (183, 28, 28))
        faults = ctx.get("faults")
        if faults is not None and len(faults):
            pdf.para("Струми КЗ у місці пошкодження:", 9)
            pdf.grid(["Місце КЗ", "Ik3 max, кА", "Ik2 min, кА", "Захист", "Реакція"],
                      [(r["Місце КЗ"], _f(r["Ik3 max, кА"]), _f(r["Ik2 min, кА"]), r["Захист"], r["Реакція"])
                       for _, r in faults.iterrows()], (50, 18, 18, 24, 56))
        for it in ctx.get("isl", []):
            pdf.para(f"Острів від АБ {it['g']}: струм КЗ інвертора ≈ {it['i_sc']:.0f} А при робочому струмі "
                     f"{it['i_load']:.0f} А; найнижча уставка МСЗ {it['min_msz']:.0f} А - потрібні адаптивні уставки.", 8.5)

    mg = ctx.get("mg")
    pdf.h2("8. Ризик = імовірність відмови × наслідки")
    if mg is None or not len(mg):
        pdf.para("Розділ ризику не включено (потрібен розрахунок N-1 для всієї мережі).", 9, 90)
    else:
        hz = ctx.get("horizon", 12)
        pdf.para(f"Імовірність відмови - ML-модель за горизонт {hz} міс.; наслідки - з N-1 (ENS при відмові активу). "
                 "Розмір бульбашки - кількість споживачів без живлення.", 8.5, 90)
        pdf.image_buf(_png(chart_risk(mg, hz)), 175)
        top = mg.head(10)
        pdf.grid(["Актив", "Тип", f"P, % ({hz} міс.)", "ENS, МВт·год", "Спожив.", "Очікуваний ENS", "Категорія"],
                  [(r.id, r.type, f"{100 * r.p_h:.1f}", f"{r.ens_mwh:.1f}", r.cust, f"{r.exp_ens:.2f}", r.cat)
                   for r in top.itertuples()], (24, 14, 24, 24, 18, 28, 24))
        w = mg.iloc[0]
        pdf.para(f"Пріоритет №1: {w['id']} - очікуваний недовідпуск {w['exp_ens']:.2f} МВт·год за {hz} міс. "
                 f"({w['cat']} ризик).", 9)

    pdf.h2("Примітки")
    pdf.para("Електрична мережа умовна (синтетична): типова схема 2 ПС 110/10 кВ, 4 фідери, 20 ТП; це не реальні об'єкти "
             "й не дані обленерго. Розрахунки режимів - pandapower (Newton-Raphson). Звіт має демонстраційний характер "
             "і не замінює оперативних розрахунків та погоджених бланків перемикань.", 8, 90)
    return bytes(pdf.output())


# ════════════════════════════════════════════════════════════════════════════
# Сервісний шар (фасад)
# ════════════════════════════════════════════════════════════════════════════
def day_key():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def validate_elements(elements):
    els = tuple(elements)
    bad = [e for e in els if e not in ALL_ELEMS]
    if not els or bad:
        raise ValueError(f"невідомі або порожні елементи: {bad or 'порожній список'}; допустимі: див. ALL_ELEMS")
    return els


def resolve_params(district="Вінниця", use_forecast=False, hist_bytes=None, **kw):
    """Параметри режиму P; з use_forecast=True підставляє ML-профіль на завтра (температура - з прогнозу).
    Повертає (P, FC або None, повідомлення про помилку прогнозу)."""
    if district not in DISTRICTS:
        raise ValueError(f"невідомий район: {district}")
    fc, err = None, ""
    if use_forecast:
        try:
            fc = forecast_load(DISTRICTS[district]["lat"], DISTRICTS[district]["lon"], day_key(), hist_bytes)
            kw = dict(kw, temp=float(fc["temps"][int(kw.get("hour", 18))]), prof=fc["profile"], pvp=fc["pv_frac"])
        except Exception as ex:                      # noqa: BLE001
            err = f"{type(ex).__name__}: {ex}"
    return make_params(**kw), fc, err


def profile_label(P, fc=None):
    p = Params(*P)
    if p.prof:
        return f"ML-прогноз на {fc['date']}" if fc else "ML-прогноз на завтра"
    return "статичний (HOURLY × T)"


def risk_table(P, horizon=12, growth=0.0, winter_days=60, tel=None, train_df=None):
    """Реєстр ризиків: ML-ймовірності відмов × наслідки N-1. Повертає (mg, scan_df)."""
    scan_df, _ = scan_all(P)
    model = get_model(make_training() if train_df is None else train_df)[0]
    reg = score_registry(model, registry_features(P, growth, winter_days, tel), horizon)
    return risk_register(reg, scan_df), scan_df


def build_pdf(P, elements, district, include_risk=True, horizon=12, growth=0.0, winter_days=60,
              fc=None, tel=None):
    """PDF-звіт: умови, показники, схема, протокол, напруги, лінії, РЗА і (опційно) матриця ризиків."""
    els = validate_elements(elements)
    ctx = dict(district=district, elements=els, p=Params(*P), res=solve_c(P, els),
               profile_label=profile_label(P, fc), horizon=horizon)
    try:
        faults, post_df, in_df, isl = fault_report(P, els)
        ctx.update(faults=faults, post_df=post_df, in_df=in_df, isl=isl)
    except Exception:                                # noqa: BLE001  (РЗА - необов'язковий розділ)
        pass
    if include_risk:
        ctx["mg"], _ = risk_table(P, horizon, growth, winter_days, tel)
    return render_report(ctx)


# ── JSON-сумісні підсумки (для API) ──
def df_records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def summarize_scenario(res):
    fin = res["final"]
    steps = [{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in s.items()} for s in res["steps"]]
    return dict(
        metrics=dict(total_mw=float(res["total_mw"]), unsupplied_mw=float(res["unsup_mw"]), ens_mwh=float(res["ens"]),
                     customers_interrupted=int(res["cust_cut"]), customers_unrestored=int(res["cust_unrestored"]),
                     saidi_min=float(res["saidi"]), losses_kw=float(res["losses"] * 1000),
                     vmin_pu=float(res["vmin_fin"]), max_loading=float(res["max_line"]),
                     violations=int(res["viol"]), pv_mw=float(fin["pv_mw"]), bess_mw=float(fin["bess_mw"])),
        status=dict(res["status"]), shed=list(res["shed"]), shed_pv=list(res["shed_pv"]),
        closed_ties=sorted(e for e in res["closed_fin"] if EDGES[e]["tie"]),
        islands=[dict(source=g, autonomy_h=(None if res["aut"].get(g, 1e9) > 1e8 else float(res["aut"][g])))
                 for g in res["gf"]],
        steps=steps,
        lines=[dict(line=eid, loading=float(fin["ratio"][eid]), i_a=float(fin["i_a"][eid]),
                    p_mw=float(fin["p_from"][eid]), q_mvar=float(fin["q_from"][eid]), loss_kw=float(fin["pl_kw"][eid]))
               for eid in fin["ratio"]],
        voltages={n: round(float(v), 4) for n, v in fin["vm"].items() if NODES[n]["type"] == "tp"},
    )


def grid_description():
    return dict(
        nodes=[dict(id=n, type=d["type"], kva=d["kva"], load_mw=d["load"], consumers=d["cons"], priority=d["prio"],
                    consumer_type=d["ctype"], pv_mw=d["pv"], bess_mw=d["bess_mw"], bess_mwh=d["bess_mwh"])
               for n, d in NODES.items()],
        edges=[dict(id=eid, from_=e["u"], to=e["v"], length_km=e["len"], limit_a=round(e["imax"] * 1000, 1),
                    normally_open=e["tie"]) for eid, e in EDGES.items()])


import types as _types
service = _types.SimpleNamespace(**{n: globals()[n] for n in (
    "day_key", "validate_elements", "resolve_params", "profile_label", "risk_table", "build_pdf", "df_records",
    "summarize_scenario", "grid_description", "forecast_load", "scan_all", "solve_c", "fault_report")})


# ════════════════════════════════════════════════════════════════════════════
# FastAPI (uvicorn app:api): ті самі розрахунки через HTTP/JSON
# ════════════════════════════════════════════════════════════════════════════
try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Query
    from fastapi.responses import Response
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except Exception:                                    # noqa: BLE001
    HAS_FASTAPI = False

api = None
if HAS_FASTAPI:
    def _auth(x_api_key: Optional[str] = Header(default=None)):
        key = os.environ.get("GRID_API_KEY")
        if key and x_api_key != key:
            raise HTTPException(status_code=401, detail="Невірний або відсутній X-API-Key")


    api = FastAPI(title="Grid Twin API - Вінницька область", version="3.0",
                  description="Сценарії N-1/N-k, режими pandapower, РЗА, прогноз навантаження, ризики, PDF-звіти. "
                              "Мережа УМОВНА (синтетична).",
                  dependencies=[Depends(_auth)])


    class ParamsIn(BaseModel):
        """Параметри режиму (одиниці як у бічній панелі застосунку: відсотки - у відсотках)."""
        district: str = "Вінниця"
        temp: float = Field(-10.0, ge=-40, le=45, description="°C; з use_forecast=true береться з прогнозу")
        hour: int = Field(18, ge=0, le=23)
        sens: float = Field(1.2, ge=0, le=5, description="%/°C")
        src_cap: float = Field(12.0, gt=0, le=100, description="допустима потужність кожної ПС, МВт")
        tol_pct: float = Field(0.0, ge=0, le=50)
        t_detect: float = Field(5.0, ge=0)
        t_switch: float = Field(4.0, ge=0)
        repair_h: float = Field(8.0, gt=0)
        vmin: float = Field(0.90, ge=0.5, le=1.0)
        vmax: float = Field(1.10, ge=1.0, le=1.5)
        u_ps: float = Field(10.5, ge=9.5, le=11.5)
        cosphi: float = Field(0.95, ge=0.5, le=1.0)
        der: bool = True
        pv_k: float = Field(1.0, ge=0, le=5)
        ins_pct: float = Field(80.0, ge=0, le=100)
        bess_mode: Literal["reserve", "peak_shaving"] = "reserve"
        soc_pct: float = Field(80.0, ge=0, le=100)
        island: bool = True
        use_forecast: bool = Field(False, description="ML-профіль навантаження та PV на завтра (Open-Meteo)")

        def resolve(self):
            try:
                P, fc, err = service.resolve_params(
                    district=self.district, use_forecast=self.use_forecast, temp=self.temp, hour=self.hour,
                    sens=self.sens, src_cap=self.src_cap, tol=self.tol_pct / 100, t_detect=self.t_detect,
                    t_switch=self.t_switch, repair_h=self.repair_h, vmin=self.vmin, vmax=self.vmax, u_ps=self.u_ps,
                    cosphi=self.cosphi, der=self.der, pv_k=self.pv_k, ins=self.ins_pct,
                    bess_mode=0 if self.bess_mode == "reserve" else 1, soc=self.soc_pct / 100, island=self.island)
            except ValueError as ex:
                raise HTTPException(status_code=422, detail=str(ex))
            return P, fc, err


    class ScenarioIn(BaseModel):
        params: ParamsIn = ParamsIn()
        elements: List[str] = Field(..., examples=[["ТП-103"]], description="Елементи, що виходять з ладу (N-k)")


    class RiskIn(BaseModel):
        params: ParamsIn = ParamsIn()
        horizon_months: int = Field(12, ge=1, le=24)
        growth_pct: float = Field(0.0, ge=0, le=40)
        winter_days: int = Field(60, ge=0, le=150)
        top: int = Field(15, ge=1, le=100)


    class ReportIn(ScenarioIn):
        include_risk: bool = True
        horizon_months: int = Field(12, ge=1, le=24)


    def _elements(els):
        try:
            return service.validate_elements(els)
        except ValueError as ex:
            raise HTTPException(status_code=422, detail=str(ex))


    @api.get("/health", tags=["service"])
    def api_health():
        return {"status": "ok", "elements": len(ALL_ELEMS), "districts": len(DISTRICTS)}


    @api.get("/districts", tags=["grid"])
    def api_districts():
        return [dict(name=n, **c) for n, c in DISTRICTS.items()]


    @api.get("/grid", tags=["grid"])
    def api_grid():
        """Склад умовної мережі: вузли (ПС/ТП) та лінії."""
        return service.grid_description()


    @api.post("/scenario", tags=["calc"])
    def api_scenario(body: ScenarioIn):
        """Аварія -> ізоляція -> відновлення (перемички/острів) -> ГОП; підсумки, протокол перемикань, режим ліній."""
        P, fc, err = body.params.resolve()
        out = service.summarize_scenario(service.solve_c(P, _elements(body.elements)))
        out["profile"] = service.profile_label(P, fc)
        if err:
            out["forecast_warning"] = err
        return out


    @api.post("/n1", tags=["calc"])
    def api_n1(body: ParamsIn):
        """N-1 для всіх елементів мережі (довго: ≈10–40 с першого разу, далі з кешу)."""
        P, _, _ = body.resolve()
        df, _ = service.scan_all(P)
        return service.df_records(df.sort_values("ens_mwh", ascending=False))


    @api.post("/protection", tags=["calc"])
    def api_protection(body: ScenarioIn):
        """Струми КЗ у місці пошкодження та перевірка уставок РЗА після перемикань."""
        P, _, _ = body.params.resolve()
        faults, post_df, in_df, isl = service.fault_report(P, _elements(body.elements))
        return dict(faults=service.df_records(faults.drop(columns="lvl")), feeders_after=service.df_records(post_df.drop(columns="lvl")),
                    inputs=service.df_records(in_df), islands=isl)


    @api.get("/forecast", tags=["ml"])
    def api_forecast(district: str = Query("Вінниця")):
        """ML-прогноз навантаження та PV на завтра (LightGBM + Open-Meteo) для району."""
        if district not in DISTRICTS:
            raise HTTPException(status_code=422, detail=f"невідомий район: {district}")
        try:
            fc = service.forecast_load(DISTRICTS[district]["lat"], DISTRICTS[district]["lon"], service.day_key())
        except Exception as ex:                          # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"прогноз недоступний: {type(ex).__name__}: {ex}")
        return {k: v for k, v in fc.items() if k != "bt"}


    @api.post("/risk", tags=["ml"])
    def api_risk(body: RiskIn):
        """Ризик = P(відмови, ML) × наслідки (N-1): топ активів за очікуваним недовідпуском."""
        P, _, _ = body.params.resolve()
        mg, _ = service.risk_table(P, body.horizon_months, body.growth_pct / 100, body.winter_days)
        cols = ["id", "type", "p_h", "ens_mwh", "unsup_mw", "cust", "exp_ens", "exp_saidi", "cat", "reco"]
        return service.df_records(mg[cols].head(body.top))


    @api.post("/report/pdf", tags=["report"], response_class=Response)
    def api_report_pdf(body: ReportIn):
        """PDF-звіт для диспетчера/керівника (application/pdf)."""
        P, fc, _ = body.params.resolve()
        pdf = service.build_pdf(P, _elements(body.elements), body.params.district, body.include_risk,
                                body.horizon_months, fc=fc)
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="grid_twin_report.pdf"'})


# ════════════════════════════════════════════════════════════════════════════
# Streamlit-інтерфейс
# ════════════════════════════════════════════════════════════════════════════
def _in_streamlit():
    """True лише коли код виконує Streamlit (а не uvicorn/pytest/імпорт з іншого скрипта)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:
            return get_script_run_ctx() is not None
    except Exception:                                # noqa: BLE001
        return False


if _in_streamlit():
    st.set_page_config(page_title="Цифровий двійник мережі 10 кВ: Вінницька область", page_icon="⚡", layout="wide")


    def spin(msg, fn, *a, **k):
        """Виклик обчислювального шару зі спінером."""
        with st.spinner(msg):
            return fn(*a, **k)


    @st.fragment(run_every=2)
    def live_panel(store, assets):
        s = store.status()
        lat_df = store.latest()
        if lat_df.empty:
            if store.error:
                st.error(store.error)
            elif store.mode == "mqtt":
                st.info(f"Очікуємо дані: {store.host}:{store.port}, топік `{store.prefix}/telemetry/#`. "
                        f"Запустіть генератор: `python telemetry_generator.py --host {store.host} "
                        f"--port {store.port} --prefix {store.prefix}`")
            else:
                st.info("Очікуємо перші повідомлення…")
            return
        for col, dflt in (("vt", "-"), ("kind", "-"), ("i_rated_a", np.nan), ("p_mw", np.nan), ("u_kv", np.nan),
                          ("winding_c", np.nan), ("oil_c", np.nan), ("loading", 0.0), ("i_a", np.nan)):
            if col not in lat_df:
                lat_df[col] = dflt
        lat_df["Стан"] = [classify_state(r.loading, r.oil_c, r.winding_c) for r in lat_df.itertuples()]
        bad = lat_df[lat_df["Стан"].str.startswith("🔴")]
        c = st.columns(5)
        c[0].metric("Зв'язок", "є" if s["connected"] else "немає")
        c[1].metric("Повідомлень", f"{s['n']:,}".replace(",", " "), f"{s['rate']:.1f} повід./с", delta_color="off")
        c[2].metric("Віртуальний час", str(lat_df["vt"].iloc[0]))
        imax = lat_df["oil_c"].idxmax()
        c[3].metric("Макс. T масла", f"{lat_df.at[imax, 'oil_c']:.0f} °C", str(lat_df.at[imax, "asset"]), delta_color="off")
        c[4].metric("Тривог", len(bad))
        if len(bad):
            st.error("🔴 Тривога: " + ", ".join(
                f"{r.asset} (навантаження {r.loading:.0%}, масло {r.oil_c:.0f} °C)" for r in bad.itertuples()))
        df = store.frame(assets)
        if not df.empty:
            f1 = go.Figure()
            f2 = go.Figure()
            for a in assets:
                d = df[df["asset"] == a]
                f1.add_trace(go.Scatter(x=d["t"], y=d["oil_c"], name=a, mode="lines"))
                f2.add_trace(go.Scatter(x=d["t"], y=d["i_a"], name=a, mode="lines"))
            f1.add_hline(y=95, line=dict(color="#d32f2f", dash="dash"), annotation_text="T масла 95 °C")
            f1.add_hline(y=85, line=dict(color="#fb8c00", dash="dot"))
            f1.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="°C",
                             title="Температура верхніх шарів масла")
            f2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="А",
                             title="Струм 10 кВ, А")
            g1, g2 = st.columns(2)
            g1.plotly_chart(f1, key="live_t")
            g2.plotly_chart(f2, key="live_i")
        show = lat_df.sort_values("loading", ascending=False)[
            ["asset", "kind", "loading", "i_a", "i_rated_a", "oil_c", "winding_c", "u_kv", "p_mw", "Стан"]].copy()
        show["loading"] *= 100
        show.columns = ["Актив", "Тип", "Завантаження, %", "I, А", "I ном, А", "T масла, °C", "T обмотки, °C",
                        "U, кВ", "P, МВт", "Стан"]
        st.dataframe(show, hide_index=True, column_config={
            "Завантаження, %": st.column_config.ProgressColumn(min_value=0, max_value=150, format="%.0f%%"),
            "I, А": st.column_config.NumberColumn(format="%.0f"),
            "I ном, А": st.column_config.NumberColumn(format="%.0f"),
            "T масла, °C": st.column_config.NumberColumn(format="%.1f"),
            "T обмотки, °C": st.column_config.NumberColumn(format="%.1f"),
            "U, кВ": st.column_config.NumberColumn(format="%.2f"),
            "P, МВт": st.column_config.NumberColumn(format="%.2f")})


    # ════════════════════════════════════════════════════════════════════════════
    # 8. ІНТЕРФЕЙС
    # ════════════════════════════════════════════════════════════════════════════
    st.title("⚡ Цифровий двійник мережі 10 кВ · Вінницька область")
    st.caption("Умовна (синтетична) типова мережа, розміщена в обраному районі Вінницької області - не реальні об'єкти обленерго. "
               "2 ПС 110/10 кВ, 4 фідери, 20 ТП, 6 нормально-розімкнених перемичок, "
               "6 PV-станцій та 3 акумуляторні батареї. Режими рахує pandapower (Newton-Raphson).")

    with st.sidebar:
        st.header("🗺️ Вінницька область")
        district = st.selectbox("Район (центр умовної мережі)", list(DISTRICTS), index=0,
                                help="Визначає розміщення мережі на карті та координати для погоди й сонячної радіації Open-Meteo")
        lat, lon = DISTRICTS[district]["lat"], DISTRICTS[district]["lon"]
        st.header("📈 Профіль навантаження")
        prof_mode = st.radio("Джерело добового профілю", ["Статичний (HOURLY × T)", "ML-прогноз на завтра"],
                             help="ML-режим: LightGBM + прогноз температури та сонячної радіації Open-Meteo для обраного району; "
                                  "година доби = година завтрашнього дня; повзунок «Інсоляція» ігнорується")
        use_fc = prof_mode.startswith("ML")
        with st.expander("Історія навантаження (CSV зі SCADA)"):
            hist_up = st.file_uploader("CSV історії навантаження (ts, load_mw)", type="csv", key="hist_csv")
            hist_bytes = hist_up.getvalue() if hist_up is not None else None
        st.header("⚙️ Умови режиму")
        temp = st.slider("Температура повітря, °C", -30, 15, -10,
                         help="У режимі «ML-прогноз» температуру береться з прогнозу погоди для обраної години")
        hour = st.slider("Година доби", 0, 23, 18)
        sens = st.slider("Температурна чутливість навантаження, %/°C", 0.0, 3.0, 1.2, 0.1)
        src_cap = st.number_input("Допустима потужність кожної ПС, МВт", 5.0, 40.0, 12.0, 0.5)
        tol = st.slider("Допустиме перевантаження ліній/ПС, %", 0, 20, 0)
        st.header("🔌 Електричний режим")
        u_ps = st.slider("Напруга на шинах 10 кВ ПС, кВ", 10.0, 10.8, 10.5, 0.1)
        cosphi = st.slider("cos φ навантаження", 0.80, 1.00, 0.95, 0.01)
        vmin = st.slider("Мін. допустима напруга, о.е.", 0.85, 0.98, 0.90, 0.01)
        vmax = st.slider("Макс. допустима напруга, о.е.", 1.02, 1.15, 1.10, 0.01)
        st.header("☀️ ВДЕ та накопичувачі")
        der = st.checkbox("Увімкнути ВДЕ (PV) та АБ", True)
        pv_k = st.slider("Потужність PV, × від встановленої", 0.0, 2.0, 1.0, 0.1)
        ins = st.slider("Інсоляція, % від зимового максимуму", 0, 100, 80)
        bess_mode_txt = st.selectbox("Режим АБ у нормі", ["Резерв (не віддає в мережу)", "Пікове голення"])
        soc = st.slider("Рівень заряду АБ (SoC), %", 10, 100, 80)
        island = st.checkbox("Дозволити острівний режим (АБ формує мережу)", True)
        st.header("⏱️ Час та ремонт")
        t_detect = st.number_input("Виявлення і локалізація, хв", 1.0, 60.0, 5.0, 1.0)
        t_switch = st.number_input("Час однієї комутації, хв", 1.0, 60.0, 4.0, 1.0)
        repair_h = st.number_input("Тривалість ремонту, год", 1.0, 72.0, 8.0, 0.5)

    DAY_KEY = service.day_key()
    FC, FC_ERR = None, ""
    try:
        FC = spin("Прогноз навантаження (Open-Meteo + LightGBM)…", forecast_load, float(lat), float(lon), DAY_KEY, hist_bytes)
    except Exception as ex:                             # noqa: BLE001
        FC_ERR = f"{type(ex).__name__}: {ex}"
    use_fc = bool(use_fc and FC is not None)
    if use_fc:
        temp = int(round(FC["temps"][hour]))

    P = make_params(temp=temp, hour=hour, sens=sens, src_cap=src_cap, tol=tol / 100.0, t_detect=t_detect,
                    t_switch=t_switch, repair_h=repair_h, vmin=vmin, vmax=vmax, u_ps=u_ps, cosphi=cosphi, der=der,
                    pv_k=pv_k, ins=ins, bess_mode=0 if bess_mode_txt.startswith("Резерв") else 1, soc=soc / 100.0,
                    island=island, prof=FC["profile"] if use_fc else (), pvp=FC["pv_frac"] if use_fc else ())
    PP = Params(*P)
    with st.sidebar:
        st.caption(f"Коефіцієнт навантаження: **{load_scale(PP):.2f}** · T = **{temp:+d} °C** · "
                   f"PV зараз: **{pv_frac(PP):.0%}** від встановленої.")
        if FC_ERR:
            st.warning(f"ML-прогноз недоступний ({FC_ERR}); використано статичний профіль.")

    # --- діагностика: якщо Power Flow не рахується (наприклад, несумісні версії pandas/pandapower) ---
    _ok, _err = health_check(PP)
    if not _ok:
        st.error("⚠️ Power Flow (pandapower) не вдалося виконати: "
                 f"`{_err or 'не зійшовся'}`. Якщо в тексті є «read-only», додайте у requirements.txt "
                 "рядок `pandas<3` (або оновіть pandapower) і перезапустіть застосунок.")
        st.stop()

    tabs = st.tabs(["🔥 Сценарний симулятор", "⚡ КЗ та релейний захист", "☀️ ВДЕ та добовий режим",
                    "🗺️ Вінницька область", "📈 Прогноз навантаження", "📚 Бібліотека протоколів N-1",
                    "📡 Телеметрія (live)", "🔮 Прогноз відмов", "⚖️ Ризик = P × наслідки"])
    tab1, tab_kz, tab_der, tab_obl, tab_fc, tab2, tab_live, tab3, tab4 = tabs

    # ─────────────────────────── TAB 1: СИМУЛЯТОР ───────────────────────────
    with tab1:
        st.subheader(f"Що станеться, якщо елемент вийде з ладу о {hour}:00 при {temp:+d} °C?")
        sel = st.multiselect("Елемент(и), що виходять з ладу (можна кілька - сценарій N-k)",
                             ALL_ELEMS, default=["ТП-103"], format_func=fmt_elem)
        res = None
        if not sel:
            st.info("Оберіть хоча б один елемент.")
        else:
            res = dict(spin("Розрахунок сценарію (pandapower)…", solve_c, P, tuple(sel)))
            res["pv_on"] = bool(der)
            fin = res["final"]
            c = st.columns(5)
            c[0].metric("Загальне навантаження", f"{res['total_mw']:.1f} МВт")
            c[1].metric("Без живлення після перемикань", f"{res['unsup_mw']:.2f} МВт",
                        f"{100 * res['unsup_mw'] / res['total_mw']:.1f}% навантаження", delta_color="inverse")
            c[2].metric("Споживачів без живлення", f"{res['cust_unrestored']:,}".replace(",", " "),
                        f"{res['cust_cut']} зазнали переривання", delta_color="off")
            c[3].metric("Недовідпуск (ENS)", f"{res['ens']:.1f} МВт·год")
            c[4].metric("SAIDI сценарію", f"{res['saidi']:.0f} хв/спож.")
            c = st.columns(5)
            c[0].metric("Втрати в мережі", f"{res['losses'] * 1000:.0f} кВт",
                        f"{100 * res['losses'] / max(res['total_mw'], 1e-9):.1f}% навантаження", delta_color="off")
            c[1].metric("Мін. напруга", f"{res['vmin_fin']:.3f} о.е.", f"{res['vmin_fin'] * 10:.2f} кВ", delta_color="off")
            c[2].metric("Макс. завантаження", f"{res['max_line']:.0%}")
            c[3].metric("Генерація PV", f"{fin['pv_mw']:.2f} МВт")
            c[4].metric("АБ (розряд +/заряд −)", f"{fin['bess_mw']:+.2f} МВт")

            if res["viol"] > 0:
                st.error(f"Після перемикань залишилось {res['viol']} порушень режиму (завантаження/напруга) - "
                         "потрібні додаткові заходи (мобільна генерація, додаткові обмеження).")
            elif res["unsup_mw"] <= 1e-9:
                st.success("Живлення всіх споживачів вдається відновити без порушень режиму.")
            else:
                st.warning("Частину споживачів неможливо повернути до завершення ремонту.")
            for g in res["gf"]:
                a = res["aut"].get(g, 1e9)
                n_isl = sum(1 for n in TPS if res["topo_f"]["root"].get(n) == g and n not in res["shed"])
                if a < repair_h:
                    st.warning(f"🔋 Острів від АБ {g}: {n_isl} ТП живляться ≈ {a:.1f} год (SoC {soc}%), "
                               f"потім живлення зникає до завершення ремонту ({repair_h:.1f} год).")
                else:
                    st.info(f"🔋 Острів від АБ {g}: {n_isl} ТП живляться автономно, енергії АБ вистачає на весь ремонт.")

            cm = st.columns([2, 1])
            phase = cm[0].radio("Стан схеми", PHASES, index=2, horizontal=True)
            color_by = cm[1].radio("Забарвлення вузлів", ["Статус", "Напруга"], horizontal=True)
            view = st.radio("Вигляд", ["Карта (OpenStreetMap)", "Схема"], horizontal=True, key="view_mode")
            if view.startswith("Карта"):
                st.plotly_chart(draw_geo_map(res, phase, color_by, (lat, lon), district), key="geomap")
            else:
                st.plotly_chart(draw_map(res, phase, color_by), key="map")
            st.caption("Лінії: 🟢 <60% · 🟡 60–80% · 🟠 80–100% · 🔴 >100% від допустимого струму; пунктир сірий - "
                       "розімкнена перемичка; чорний пунктир - пошкоджено. ★ - PV, ⬢ - АБ.")
            st.plotly_chart(voltage_profile(res, phase, PP), key="vprof")

            st.markdown("#### 📋 Протокол перемикань (готовий для диспетчера)")
            proto = pd.DataFrame(res["steps"]).drop(columns="type")
            st.dataframe(proto, hide_index=True)
            st.download_button("⬇️ Завантажити протокол (CSV)", proto.to_csv(index=False).encode("utf-8-sig"),
                               file_name="protocol_" + "_".join(sel) + ".csv", mime="text/csv")

            st.markdown("#### 📄 PDF-звіт для диспетчера / керівника")
            pc = st.columns([3, 1, 2])
            rep_risk = pc[0].checkbox("Додати матрицю ризиків (потрібен N-1 для всієї мережі, до ~40 с)", value=False)
            rep_h = pc[1].selectbox("Горизонт, міс.", [3, 6, 12, 24], index=2)
            rep_key = (P, tuple(sel), district, rep_risk, rep_h)
            pc[2].write("")
            if pc[2].button("Сформувати PDF-звіт"):
                with st.spinner("Формування звіту…"):
                    st.session_state["pdf_report"] = (rep_key, service.build_pdf(
                        P, tuple(sel), district, include_risk=rep_risk, horizon=rep_h, fc=FC if use_fc else None))
            stored = st.session_state.get("pdf_report")
            if stored and stored[0] == rep_key:
                st.download_button("⬇️ Завантажити PDF-звіт", stored[1], file_name="grid_twin_report.pdf",
                                   mime="application/pdf")
            elif stored:
                st.info("Параметри або вибір елементів змінилися - сформуйте звіт заново.")

            st.markdown("#### Режим ліній за етапами (pandapower)")
            rows = []
            for eid, e in EDGES.items():
                if e["tie"] and eid not in res["closed_fin"]:
                    continue
                g = lambda s: 100 * s["ratio"].get(eid, 0.0)
                rows.append({"Лінія": eid, "Довжина, км": e["len"], "Ліміт, А": e["imax"] * 1000,
                             "Норма, %": g(res["normal"]), "Після аварії, %": g(res["post"]),
                             "Після перемикань, %": g(fin), "P, МВт": fin["p_from"].get(eid, 0.0),
                             "Q, МВАр": fin["q_from"].get(eid, 0.0), "I, А": fin["i_a"].get(eid, 0.0),
                             "Втрати, кВт": fin["pl_kw"].get(eid, 0.0)})
            ldf = pd.DataFrame(rows).sort_values("Після перемикань, %", ascending=False)
            pc = lambda: st.column_config.ProgressColumn(min_value=0, max_value=150, format="%.0f%%")
            st.dataframe(ldf, hide_index=True, column_config={
                "Норма, %": pc(), "Після аварії, %": pc(), "Після перемикань, %": pc(),
                "Ліміт, А": st.column_config.NumberColumn(format="%.0f"),
                "P, МВт": st.column_config.NumberColumn(format="%+.2f"),
                "Q, МВАр": st.column_config.NumberColumn(format="%+.2f"),
                "I, А": st.column_config.NumberColumn(format="%.0f"),
                "Втрати, кВт": st.column_config.NumberColumn(format="%.1f")})
            sl = pd.DataFrame([{"Джерело": s, "Завантаження, %": 100 * fin["src"].get(s, 0.0),
                                "P, МВт": fin["p_src"].get(s, 0.0), "Q, МВАр": fin["q_src"].get(s, 0.0)}
                               for s in list(SOURCES) + list(res["gf"])])
            st.dataframe(sl, hide_index=True, column_config={
                "Завантаження, %": pc(), "P, МВт": st.column_config.NumberColumn(format="%+.2f"),
                "Q, МВАр": st.column_config.NumberColumn(format="%+.2f")})
            with st.expander("Напруги у вузлах після перемикань"):
                vdf = pd.DataFrame([{"Вузол": n, "U, о.е.": fin["vm"].get(n, np.nan),
                                     "U, кВ": fin["vm"].get(n, np.nan) * 10,
                                     "Стан": STATUS_STYLE[res["status"][n]][0]} for n in TPS])
                st.dataframe(vdf.sort_values("U, о.е."), hide_index=True, column_config={
                    "U, о.е.": st.column_config.NumberColumn(format="%.3f"),
                    "U, кВ": st.column_config.NumberColumn(format="%.2f")})

            with st.expander("🌡️ Чутливість: залежність від температури та години (довго: ~30 розрахунків)"):
                if st.button("Розрахувати карту чутливості"):
                    hours_t, temps_t = (6, 12, 15, 18, 21), (-25, -20, -15, -10, -5, 0, 10)
                    z = spin("Побудова карти чутливості…", sens_grid, P, tuple(sel), hours_t, temps_t)
                    fig = go.Figure(go.Heatmap(z=z, x=[f"{h}:00" for h in hours_t], y=[f"{t:+d} °C" for t in temps_t],
                                               colorscale="YlOrRd", texttemplate="%{z:.1f}",
                                               colorbar=dict(title="МВт без<br>живлення")))
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                                      title="Невідновлене навантаження після перемикань, МВт")
                    st.plotly_chart(fig, key="sens")

    # ─────────────────────────── TAB: КЗ ТА ЗАХИСТ ───────────────────────────
    with tab_kz:
        st.subheader("Струми короткого замикання та перевірка уставок релейного захисту")
        if not sel:
            st.info("Оберіть пошкоджений елемент на вкладці «Сценарний симулятор».")
        else:
            feeders, inp = spin("Розрахунок уставок захисту…", protection_settings, P)
            faults, post_df, in_df, isl = spin("Розрахунок струмів КЗ…", fault_report, P, tuple(sel))
            st.markdown("**Уставки (розрахунок за проєктним режимом: пік, −20 °C, без ВДЕ):** "
                        f"ТВ = 1.25·Ik3 у кінці першого сегмента; МСЗ = {K_N}·{K_ZAP}/{K_V}·Iроб.max, "
                        f"t={T_MSZ} с; вводи ПС: t={T_IN} с.")
            st.dataframe(pd.DataFrame([{"Фідер (головна ділянка)": h, "ПС": s["src"], "I роб. max, А": s["iw"],
                                        "I відсічки ТВ, А": s["i_to"], "I МСЗ, А": s["i_msz"]}
                                       for h, s in feeders.items()]), hide_index=True,
                         column_config={k: st.column_config.NumberColumn(format="%.0f")
                                        for k in ("I роб. max, А", "I відсічки ТВ, А", "I МСЗ, А")})

            st.markdown(f"#### КЗ у місці пошкодження ({', '.join(sel)})")
            show = faults.drop(columns="lvl")
            st.dataframe(show, hide_index=True, column_config={
                "Ik3 max, кА": st.column_config.NumberColumn(format="%.2f"),
                "Ik2 min, кА": st.column_config.NumberColumn(format="%.2f"),
                "I відсічки, А": st.column_config.NumberColumn(format="%.0f"),
                "I МСЗ, А": st.column_config.NumberColumn(format="%.0f"),
                "kч (МСЗ)": st.column_config.NumberColumn(format="%.2f"),
                "t, с": st.column_config.NumberColumn(format="%.1f")})
            if (faults["lvl"] == 2).any():
                st.error("Для частини точок КЗ основний захист не має чутливості - потрібен перегляд уставок "
                         "або додаткові пристрої (реклоузери, направлений/дистанційний захист).")
            elif (faults["lvl"] == 1).any():
                st.warning("Є точки з межовою чутливістю захисту (kч < 1.5).")
            else:
                st.success("Для обраних точок КЗ захист діє селективно та з достатньою чутливістю.")
            st.caption("Ik3 max враховує підживлення від ВДЕ (k=1.2·Iном). Ik2 min - без ВДЕ (консервативно). "
                       f"Комутаційна здатність вимикачів прийнята {SWITCH_BREAK_KA:.0f} кА.")

            st.markdown("#### Перевірка захисту після перемикань (резервна схема)")
            st.dataframe(post_df.drop(columns="lvl"), hide_index=True, column_config={
                "I роб. після, А": st.column_config.NumberColumn(format="%.0f"),
                "I МСЗ, А": st.column_config.NumberColumn(format="%.0f"),
                "Запас МСЗ": st.column_config.NumberColumn(format="%.2f"),
                "Ik2 min у кінці, А": st.column_config.NumberColumn(format="%.0f"),
                "kч у кінці": st.column_config.NumberColumn(format="%.2f")})
            if (post_df["lvl"] == 2).any():
                bad = post_df.loc[post_df["lvl"] == 2, "Фідер"].tolist()
                st.error(f"Ризик неселективної роботи захисту на фідерах: {', '.join(bad)} - резервна схема може "
                         "відключитись повторно (хибне спрацювання МСЗ або втрата чутливості в кінці подовженого фідера).")
            elif (post_df["lvl"] == 1).any():
                st.warning("Деякі фідери працюють з межовим запасом захисту.")
            if len(in_df):
                st.dataframe(in_df, hide_index=True, column_config={
                    "I роб. після, А": st.column_config.NumberColumn(format="%.0f"),
                    "I МСЗ вводу, А": st.column_config.NumberColumn(format="%.0f"),
                    "Запас": st.column_config.NumberColumn(format="%.2f")})
            for it in isl:
                st.warning(f"🔋 Острів від АБ {it['g']}: струм КЗ інвертора ≈ {it['i_sc']:.0f} А "
                           f"(≈1.2·Iном) при робочому струмі острова {it['i_load']:.0f} А, а найнижча уставка МСЗ "
                           f"фідерів {it['min_msz']:.0f} А: звичайний МСЗ в острові не відчує КЗ. Потрібні адаптивні "
                           "уставки або захист за напругою/частотою.")

    # ─────────────────────────── TAB: ВДЕ ТА ДОБОВИЙ РЕЖИМ ───────────────────────────
    with tab_der:
        st.subheader("Вплив ВДЕ та накопичувачів на нормальний режим (24 години)")
        dp = spin("Розрахунок доби (24 режими pandapower)…", daily_profile, P)
        a, b = dp[dp["case"] == "ВДЕ"], dp[dp["case"] == "без ВДЕ"]
        m = st.columns(4)
        m[0].metric("Пік імпорту з ПС", f"{a['imp'].max():.2f} МВт", f"{a['imp'].max() - b['imp'].max():+.2f} vs без ВДЕ",
                    delta_color="inverse")
        m[1].metric("Виробіток PV за добу", f"{a['pv'].sum():.1f} МВт·год")
        m[2].metric("Втрати за добу", f"{a['losses'].sum() * 1000:.0f} кВт·год",
                    f"{(a['losses'].sum() - b['losses'].sum()) * 1000:+.0f} vs без ВДЕ", delta_color="inverse")
        m[3].metric("Макс. напруга за добу", f"{a['vmax'].max():.3f} о.е.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a["h"], y=a["load"], name="Навантаження", line=dict(color="#455a64", width=3)))
        fig.add_trace(go.Scatter(x=a["h"], y=a["pv"], name="PV", fill="tozeroy", line=dict(color="#ffb300")))
        fig.add_trace(go.Scatter(x=a["h"], y=a["bess"], name="АБ (+розряд/−заряд)", line=dict(color="#00897b")))
        fig.add_trace(go.Scatter(x=a["h"], y=a["imp"], name="Імпорт з ПС (з ВДЕ)", line=dict(color="#6a1b9a", dash="dot", width=3)))
        fig.add_trace(go.Scatter(x=b["h"], y=b["imp"], name="Імпорт з ПС (без ВДЕ)", line=dict(color="#9e9e9e", dash="dash")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Година", yaxis_title="МВт",
                          title="Добовий баланс потужності (форма «качки» при ВДЕ)")
        st.plotly_chart(fig, key="daily1")
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=a["h"], y=a["vmin"], name="U мін (з ВДЕ)", line=dict(color="#d32f2f")))
        f2.add_trace(go.Scatter(x=a["h"], y=a["vmax"], name="U макс (з ВДЕ)", line=dict(color="#1565c0")))
        f2.add_trace(go.Scatter(x=b["h"], y=b["vmin"], name="U мін (без ВДЕ)", line=dict(color="#d32f2f", dash="dash")))
        f2.add_trace(go.Scatter(x=b["h"], y=b["vmax"], name="U макс (без ВДЕ)", line=dict(color="#1565c0", dash="dash")))
        f2.add_hline(y=PP.vmax, line=dict(color="gray", dash="dot"))
        f2.add_hline(y=PP.vmin, line=dict(color="gray", dash="dot"))
        f2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Година", yaxis_title="U, о.е.",
                         title="Діапазон напруг у вузлах ТП")
        st.plotly_chart(f2, key="daily2")
        st.caption("О 18:00 (вечірній пік) PV не виробляє, тому саме АБ визначають, чи зможе частина мережі "
                   "працювати в острівному режимі. Налаштування ВДЕ - у бічній панелі.")

    # ─────────────────────────── TAB: ВІННИЦЬКА ОБЛАСТЬ ───────────────────────────
    with tab_obl:
        st.subheader("Вінницька область: умови на завтра по районах")
        ow, o_src, o_date = spin("Погода по районах області…", oblast_weather, DAY_KEY)
        cold = ow.loc[ow["tmin"].idxmin()]
        peak = ow.loc[ow["peak_mw"].idxmax()]
        m = st.columns(4)
        m[0].metric("Дата", o_date)
        m[1].metric("Найхолодніший район", str(cold["district"]), f"{cold['tmin']:.1f} °C", delta_color="off")
        m[2].metric("Макс. пік типової мережі", f"{peak['peak_mw']:.1f} МВт", str(peak["district"]), delta_color="off")
        m[3].metric("PV: середній потенціал", f"{ow['pv_yield'].mean():.2f} кВт·год/кВт")
        Sc, mkey = _map_cls()
        fo = go.Figure()
        sel_r = ow[ow["district"] == district]
        fo.add_trace(Sc(lon=sel_r["lon"], lat=sel_r["lat"], mode="markers", hoverinfo="skip", showlegend=False,
                        marker=dict(size=38, color="#6a1b9a", opacity=0.35)))
        span = max(ow["peak_mw"].max() - ow["peak_mw"].min(), 1e-6)
        fo.add_trace(Sc(
            lon=ow["lon"], lat=ow["lat"], mode="markers+text", text=ow["district"], textposition="top center",
            showlegend=False, hoverinfo="text",
            hovertext=[f"{r.district}<br>T: {r.tmin:.1f} … {r.tmax:.1f} °C<br>Пік типової мережі: {r.peak_mw:.1f} МВт<br>"
                       f"PV: {r.pv_yield:.2f} кВт·год/кВт" for r in ow.itertuples()],
            marker=dict(size=[16 + 14 * (v - ow["peak_mw"].min()) / span for v in ow["peak_mw"]],
                        color=ow["tmin"], colorscale="RdYlBu", reversescale=True,
                        colorbar=dict(title="T мін, °C", thickness=12, len=0.6))))
        fo.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0),
                         **{mkey: dict(style="open-street-map", center=dict(lat=OBLAST_CENTER[0], lon=OBLAST_CENTER[1]),
                                       zoom=6.9)})
        st.plotly_chart(fo, key="oblast_map")
        st.caption("Колір - мінімальна температура завтра, розмір - пік навантаження типової мережі (за температурою, "
                   f"статична модель), фіолетове кільце - обраний район ({district}). Джерело: {o_src}.")
        otbl = ow[["district", "tmin", "tmax", "peak_factor", "peak_mw", "pv_yield"]].copy()
        otbl.insert(0, "Обрано", ["◀" if d == district else "" for d in otbl["district"]])
        otbl.columns = ["Обрано", "Район", "T мін, °C", "T макс, °C", "Коеф. піку", "Пік типової мережі, МВт",
                        "PV, кВт·год/кВт"]
        st.dataframe(otbl.sort_values("T мін, °C"), hide_index=True, column_config={
            "T мін, °C": st.column_config.NumberColumn(format="%.1f"),
            "T макс, °C": st.column_config.NumberColumn(format="%.1f"),
            "Коеф. піку": st.column_config.NumberColumn(format="%.2f"),
            "Пік типової мережі, МВт": st.column_config.NumberColumn(format="%.1f"),
            "PV, кВт·год/кВт": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Електрична мережа в застосунку умовна й однакова для всіх районів; відрізняються розташування, "
                   "погода й сонячна радіація. Для реальних розрахунків підключіть дані ГІС/SCADA вашого РЕМ.")

    # ─────────────────────────── TAB: ПРОГНОЗ НАВАНТАЖЕННЯ ───────────────────────────
    with tab_fc:
        st.subheader(f"ML-прогноз навантаження на завтра · {district} (LightGBM + Open-Meteo)")
        if FC is None:
            st.error(f"Прогноз недоступний: {FC_ERR}")
        else:
            prof_ml, prof_st, tmp = np.array(FC["profile"]), np.array(FC["static_profile"]), np.array(FC["temps"])
            pk = int(prof_ml.argmax())
            m = st.columns(5)
            m[0].metric("Дата прогнозу", FC["date"])
            m[1].metric("Пік навантаження", f"{prof_ml.max() * TOTAL_BASE_MW:.1f} МВт", f"о {pk}:00", delta_color="off")
            m[2].metric("Температура (мін / макс)", f"{tmp.min():.0f} / {tmp.max():.0f} °C")
            m[3].metric("MAE ML, коеф. (14 діб)", f"{FC['mae_ml']:.3f}",
                        f"{FC['mae_ml'] - FC['mae_static']:+.3f} vs статична", delta_color="inverse")
            m[4].metric("MAPE ML", f"{FC['mape_ml']:.1f}%")
            if use_fc:
                st.success("Симулятор і всі розрахунки використовують ML-профіль на завтра (температура береться з прогнозу).")
            else:
                st.info("Зараз активний статичний профіль. Оберіть «ML-прогноз на завтра» у бічній панелі, "
                        "щоб підставити цей графік у розрахунки.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(24)), y=prof_ml * TOTAL_BASE_MW, name="ML-прогноз",
                                     line=dict(color="#6a1b9a", width=3)))
            fig.add_trace(go.Scatter(x=list(range(24)), y=prof_st * TOTAL_BASE_MW, name="Статична модель (HOURLY × T)",
                                     line=dict(color="#9e9e9e", dash="dash")))
            fig.add_trace(go.Scatter(x=list(range(24)), y=tmp, name="Температура, °C", yaxis="y2",
                                     line=dict(color="#1e88e5", dash="dot")))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Година",
                              yaxis_title="МВт", yaxis2=dict(title="°C", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=-0.2), title=f"Навантаження мережі на {FC['date']}")
            st.plotly_chart(fig, key="fc_chart")
            st.caption(f"PV: {FC['pv_src']}; потенціал доби ≈ {sum(FC['pv_frac']):.1f} кВт·год/кВт встановленої. "
                       f"Район: {district} ({lat:.2f}° пн. ш., {lon:.2f}° сх. д.).")
            st.caption(f"Модель: **{FC['model']}**, навчальних годин: {FC['n_train']}. Погода: {FC['weather_src']}. "
                       f"Історія навантаження: {FC['data_src']}. Ознаки: година, день тижня, температура, "
                       "ковзні середні T за 3/24 год, градуси холоду/спеки.")
            if "синтетична" in FC["data_src"]:
                st.warning("Історія навантаження синтетична (демо). Для реальної точності завантажте CSV зі SCADA "
                           "(стовпці ts, load_mw) у бічній панелі: «Локація, погода та історія».")
            with st.expander("Перевірка на останніх 14 добах (бектест)"):
                bt = FC["bt"]
                fb = go.Figure()
                fb.add_trace(go.Scatter(x=bt["ts"], y=bt["actual"], name="Факт", line=dict(color="#455a64")))
                fb.add_trace(go.Scatter(x=bt["ts"], y=bt["ml"], name="ML", line=dict(color="#6a1b9a")))
                fb.add_trace(go.Scatter(x=bt["ts"], y=bt["static"], name="Статична", line=dict(color="#9e9e9e", dash="dash")))
                fb.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="коеф. навантаження",
                                 title="Останні 96 годин тестового вікна")
                st.plotly_chart(fb, key="fc_bt")
            tdf = pd.DataFrame({"Година": range(24), "T, °C": tmp, "Коеф. ML": prof_ml,
                                "Навантаження ML, МВт": prof_ml * TOTAL_BASE_MW, "Коеф. статичний": prof_st})
            st.dataframe(tdf, hide_index=True, column_config={
                "T, °C": st.column_config.NumberColumn(format="%.1f"),
                "Коеф. ML": st.column_config.NumberColumn(format="%.3f"),
                "Навантаження ML, МВт": st.column_config.NumberColumn(format="%.2f"),
                "Коеф. статичний": st.column_config.NumberColumn(format="%.3f")})
            st.download_button("⬇️ Прогноз (CSV)", tdf.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"load_forecast_{FC['date']}.csv", mime="text/csv")

    # ─────────────────────────── TAB 2: N-1 ───────────────────────────
    scan_df, prots = None, None
    with tab2:
        st.subheader("Бібліотека готових протоколів: N-1 для кожного елемента мережі")
        st.caption("Розраховує режим pandapower для всіх 42 елементів (≈10–40 с залежно від ПК). Результат кешується, "
                   "після зміни параметрів у бічній панелі розрахунок повторюється.")
        if st.button("▶️ Розрахувати N-1 для всієї мережі") or st.session_state.get("scan_req"):
            st.session_state["scan_req"] = True
            scan_df, prots = spin("Розрахунок N-1 для всіх елементів (pandapower)…", scan_all, P)
        if scan_df is None:
            st.info("Натисніть кнопку, щоб побудувати бібліотеку протоколів.")
        else:
            show = scan_df.sort_values("ens_mwh", ascending=False).rename(columns={
                "id": "Елемент", "type": "Тип", "unsup_mw": "Без живлення, МВт", "unsup_pct": "Без живлення, %",
                "ens_mwh": "ENS, МВт·год", "cust": "Споживачів без живлення", "saidi": "SAIDI, хв/спож.",
                "max_line": "Макс. завант., %", "vmin": "U мін, о.е.", "n_ops": "Операцій",
                "plan": "Ключові перемикання"})
            st.dataframe(show, hide_index=True, column_config={
                "Без живлення, МВт": st.column_config.NumberColumn(format="%.2f"),
                "Без живлення, %": st.column_config.NumberColumn(format="%.1f"),
                "ENS, МВт·год": st.column_config.NumberColumn(format="%.1f"),
                "SAIDI, хв/спож.": st.column_config.NumberColumn(format="%.0f"),
                "U мін, о.е.": st.column_config.NumberColumn(format="%.3f"),
                "Макс. завант., %": st.column_config.ProgressColumn(min_value=0, max_value=150, format="%.0f%%")})
            allp = pd.concat([pd.DataFrame(v).drop(columns="type").assign(Елемент_відмови=k) for k, v in prots.items()])
            st.download_button("⬇️ Усі протоколи (CSV)", allp.to_csv(index=False).encode("utf-8-sig"),
                               file_name="all_N1_protocols.csv", mime="text/csv")
            pick = st.selectbox("Переглянути протокол для елемента",
                                scan_df.sort_values("ens_mwh", ascending=False)["id"], format_func=fmt_elem)
            st.dataframe(pd.DataFrame(prots[pick]).drop(columns="type"), hide_index=True)

    # ─────────────────────────── TAB: ТЕЛЕМЕТРІЯ LIVE ───────────────────────────
    with tab_live:
        st.subheader("Телеметрія в реальному часі (імітація SCADA / IoT-потоку)")
        st.caption("Паралельний скрипт `telemetry_generator.py` публікує в MQTT струми, температуру масла й обмоток та "
                   "напругу кожного трансформатора; застосунок підписується й оновлює графіки кожні 2 с. "
                   "Без брокера працює вбудований генератор того самого симулятора.")
        cs = st.columns([1.4, 1, 1, 1])
        src_txt = cs[0].radio("Джерело потоку", ["Вбудований генератор", "MQTT-брокер"], horizontal=True)
        run_on = cs[1].toggle("Потік увімкнено", True)
        is_mqtt = src_txt.startswith("MQTT")
        interval, vstep = 1.0, 15.0
        if not is_mqtt:
            interval = cs[2].number_input("Пауза між тіками, с", 0.5, 10.0, 1.0, 0.5)
            vstep = cs[3].number_input("Віртуальних хв/тік", 1.0, 60.0, 15.0, 1.0)
        host, port, prefix, mq_user, mq_pwd = "", 1883, "", "", ""
        if is_mqtt:
            cm_ = st.columns([2, 1, 2, 1, 1])
            host = cm_[0].text_input("Брокер", "broker.hivemq.com")
            port = int(cm_[1].number_input("Порт", 1, 65535, 1883))
            prefix = cm_[2].text_input("Префікс топіків", "vinnytsia-twin-demo-CHANGE-ME",
                                       help="Публічні брокери бачать усі; задайте унікальний префікс і той самий у --prefix генератора")
            mq_user = cm_[3].text_input("Логін", "")
            mq_pwd = cm_[4].text_input("Пароль", "", type="password")
        if not run_on:
            stop_store()
            st.info("Потік вимкнено.")
        elif is_mqtt and not HAS_MQTT:
            st.error("Не встановлено paho-mqtt (додайте `paho-mqtt>=2.0` у requirements.txt).")
        else:
            store = get_store("mqtt" if is_mqtt else "builtin", host.strip(), port, prefix.strip(), mq_user, mq_pwd,
                              interval, vstep, float(temp))
            store.ambient = float(temp)
            sel_live = st.multiselect("Активи на графіках", ["ПС-1", "ПС-2"] + TPS,
                                      default=["ПС-1", "ТП-103", "ТП-404"], format_func=fmt_elem)
            ci = st.columns([2, 2, 1, 1])
            a_pick = ci[0].selectbox("Інжектувати аномалію на", ["ПС-1", "ПС-2"] + TPS, index=2, format_func=fmt_elem)
            k_pick = ci[1].radio("Тип аномалії", ["Перевантаження", "Відмова охолодження"], horizontal=True)
            ci[2].write("")
            if ci[2].button("⚡ Інжектувати"):
                store.inject(a_pick, "overload" if k_pick == "Перевантаження" else "cooling", 12)
            ci[3].write("")
            if ci[3].button("Скинути"):
                store.clear()
            live_panel(store, sel_live or ["ПС-1"])

    # ─────────────────────────── TAB 3: ПРОГНОЗ ВІДМОВ ───────────────────────────
    with tab3:
        st.subheader("Імовірність аварії обладнання до того, як станеться обрив")
        cc = st.columns(3)
        horizon = cc[0].slider("Горизонт прогнозу, міс.", 1, 24, 12)
        growth = cc[1].slider("Приріст навантаження, %", 0, 40, 0)
        wdays = cc[2].slider("Днів на рік з обраними умовами (температура з панелі)", 0, 150, 60)
        up = st.file_uploader("Опційно: власна історія відмов (CSV зі стовпцями "
                              + ", ".join(FEATURES) + ", failed)", type="csv")
        train_df = make_training()
        if up is not None:
            try:
                train_df = training_from_csv(up.getvalue())
                st.success(f"Модель навчена на ваших даних: {len(train_df)} записів.")
            except Exception as ex:  # noqa: BLE001
                st.warning(f"Файл не використано ({ex}); застосовано синтетичні дані.")
        model, auc, imp, ref = spin("Навчання моделі прогнозу відмов…", get_model, train_df)

        live_store = current_store()
        use_tel = st.checkbox("Підставити живу телеметрію у ознаки (пік/середнє завантаження, години перевантаження)",
                              value=False, disabled=live_store is None,
                              help="Потрібен запущений потік на вкладці «Телеметрія (live)»")
        tel = live_store.summary() if (use_tel and live_store is not None) else None
        reg = score_registry(model, registry_features(P, growth / 100.0, wdays, tel), horizon)
        if tel:
            st.caption(f"Живі дані використано для {int(reg['from_telemetry'].sum())} активів (≥ 20 вимірів у буфері).")

        m = st.columns(4)
        m[0].metric("AUC моделі (тест)", f"{auc:.3f}")
        m[1].metric("Середній ризик за горизонт", f"{100 * reg['p_h'].mean():.1f}%")
        m[2].metric(f"Очікувана к-сть відмов за {horizon} міс.", f"{reg['p_h'].sum():.1f}")
        m[3].metric("Критичних + високих", int(reg["cat"].isin(["Критичний", "Високий"]).sum()))

        top = reg.sort_values("p_h", ascending=False).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=100 * top["p_h"], y=top["id"], orientation="h",
                               marker_color=[CAT_COLOR[c] for c in top["cat"]],
                               text=[f"{100 * v:.1f}%" for v in top["p_h"]], textposition="outside"))
        fig.update_layout(height=470, margin=dict(l=10, r=40, t=30, b=10),
                          title=f"Топ-15 за імовірністю відмови протягом {horizon} міс., %")
        st.plotly_chart(fig, key="topp")

        tbl = reg.sort_values("p_h", ascending=False)[
            ["id", "type", "age", "loading_peak", "overload_hours", "prior_faults",
             "years_since_maint", "condition_idx", "p_h", "cat", "reco"]].copy()
        tbl["loading_peak"] *= 100
        tbl["p_h"] *= 100
        tbl.columns = ["Актив", "Тип", "Вік, років", "Пік. завант., %", "Год. перевант./рік",
                       "Відмов за 3 р.", "Років від ТО", "Індекс стану", f"P відмови, % ({horizon} міс.)",
                       "Категорія", "Рекомендація"]
        st.dataframe(tbl, hide_index=True, column_config={
            f"P відмови, % ({horizon} міс.)": st.column_config.ProgressColumn(min_value=0, max_value=50, format="%.1f%%"),
            "Пік. завант., %": st.column_config.NumberColumn(format="%.0f"),
            "Індекс стану": st.column_config.NumberColumn(format="%.0f")})

        st.markdown("#### 🔎 Чому ризик саме такий? (SHAP)")
        asset = st.selectbox("Актив", reg.sort_values("p_h", ascending=False)["id"], format_func=fmt_elem)
        row = reg[reg["id"] == asset].iloc[0]
        shap_done = False
        if HAS_SHAP:
            try:
                ex_p, smp, sv = spin("Побудова SHAP-пояснювача…", get_shap, train_df)
                x_row = row[FEATURES].astype(float).to_frame().T
                e = shap_row(ex_p, x_row)
                ca, cb = st.columns([3, 2])
                fw = fig_waterfall(e)
                ca.pyplot(fw, clear_figure=True)
                close_fig(fw)
                top3 = np.argsort(-np.abs(e.values))[:3]
                cb.markdown(f"**{fmt_elem(asset)}**: річна імовірність відмови **{e.base_values + e.values.sum():.1f}%** "
                            f"(у середньому по навчальній вибірці {e.base_values:.1f}%).")
                cb.markdown("Найсильніше впливають:")
                for i in top3:
                    cb.markdown(f"- {UA_NAMES[i]} = {e.data[i]:.2f}: **{e.values[i]:+.1f} п.п.**")
                cb.caption("Waterfall: від середнього ризику до ризику цього активу; червоне - підвищує, синє - знижує. "
                           "Значення SHAP у відсоткових пунктах річної імовірності.")
                with st.expander("Загальна картина: SHAP summary (навчальна вибірка)"):
                    kind = st.radio("Вигляд", ["Beeswarm (розподіл впливу)", "Bar (середній |SHAP|)"], horizontal=True)
                    fs = fig_summary(sv, smp, UA_NAMES, bar=kind.startswith("Bar"))
                    st.pyplot(fs, clear_figure=True)
                    close_fig(fs)
                    st.caption("Вісь X - вплив на log-odds відмови. Кожна точка - один актив навчальної вибірки; "
                               "колір - значення ознаки (червоне - високе).")
                shap_done = True
            except Exception as ex_:                     # noqa: BLE001
                st.warning(f"SHAP недоступний ({type(ex_).__name__}: {ex_}); показано спрощене пояснення.")
        if not shap_done:
            ex = explain(model, row, ref)
            fig2 = go.Figure(go.Bar(x=100 * ex.values, y=ex.index, orientation="h",
                                    marker_color=["#d32f2f" if v > 0 else "#43a047" for v in ex.values]))
            fig2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                               title="Внесок факторів у річну імовірність відмови (п.п. відносно «типового» активу)")
            st.plotly_chart(fig2, key="expl")
            with st.expander("Важливість ознак у моделі"):
                imp2 = imp.rename(index=FEAT_UA).sort_values()
                f3 = go.Figure(go.Bar(x=imp2.values, y=imp2.index, orientation="h"))
                f3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(f3, key="imp")
        st.caption("Модель: GradientBoostingClassifier + SHAP (TreeExplainer). Завантаження активів - з розрахунку pandapower "
                   "або з live-телеметрії. "
                   "Імовірність за горизонт: 1 − (1 − p_рік)^(міс./12). Синтетична історія слугує лише демонстрацією.")

    # ─────────────────────────── TAB 4: РИЗИК ───────────────────────────
    with tab4:
        st.subheader("Пріоритезація: ймовірність відмови × наслідки для споживачів")
        st.caption("Наслідки беруться з N-1 (вкладка «Бібліотека протоколів»), імовірності - з моделі прогнозу.")
        if scan_df is None:
            if st.button("▶️ Розрахувати N-1 і побудувати матрицю ризику", key="scan2"):
                st.session_state["scan_req"] = True
                st.rerun()
            st.info("Для матриці ризику потрібен розрахунок N-1 (кнопка вище або на вкладці «Бібліотека протоколів N-1»).")
        else:
            mg = risk_register(reg, scan_df)
            fig4 = go.Figure(go.Scatter(
                x=100 * mg["p_h"], y=mg["ens_mwh"], mode="markers+text", text=mg["id"],
                textposition="top center", textfont=dict(size=9),
                marker=dict(size=10 + 25 * np.sqrt(mg["cust"] / max(mg["cust"].max(), 1)),
                            color=mg["exp_ens"], colorscale="YlOrRd", showscale=True,
                            colorbar=dict(title="Очікуваний<br>недовідпуск,<br>МВт·год"),
                            line=dict(color="white", width=1)),
                hovertext=[f"{r.id}<br>P={100 * r.p_h:.1f}%<br>ENS={r.ens_mwh:.1f} МВт·год<br>"
                           f"Без живлення: {r.cust} спож." for r in mg.itertuples()], hoverinfo="text"))
            fig4.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                               xaxis_title=f"Імовірність відмови за {horizon} міс., %",
                               yaxis_title="Наслідок аварії: ENS, МВт·год",
                               title="Матриця ризику (розмір бульбашки = споживачі без живлення)")
            st.plotly_chart(fig4, key="riskmat")
            out = mg[["id", "type", "p_h", "ens_mwh", "unsup_mw", "cust", "exp_ens", "exp_saidi", "cat"]].copy()
            out["p_h"] *= 100
            out.columns = ["Актив", "Тип", f"P, % ({horizon} міс.)", "ENS при відмові, МВт·год", "Без живлення, МВт",
                           "Споживачів без живлення", "Очікуваний ENS, МВт·год", "Очікуваний SAIDI, хв/спож.", "Категорія"]
            st.dataframe(out, hide_index=True, column_config={
                f"P, % ({horizon} міс.)": st.column_config.NumberColumn(format="%.1f"),
                "ENS при відмові, МВт·год": st.column_config.NumberColumn(format="%.1f"),
                "Без живлення, МВт": st.column_config.NumberColumn(format="%.2f"),
                "Очікуваний ENS, МВт·год": st.column_config.NumberColumn(format="%.2f"),
                "Очікуваний SAIDI, хв/спож.": st.column_config.NumberColumn(format="%.1f")})
            st.download_button("⬇️ Ризик-реєстр (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                               file_name="risk_register.csv", mime="text/csv")
            worst = mg.iloc[0]
            st.info(f"**Пріоритет №1: {fmt_elem(worst['id'])}** - очікуваний недовідпуск "
                    f"{worst['exp_ens']:.2f} МВт·год за {horizon} міс. ({worst['cat']} ризик). "
                    f"Готовий протокол перемикань нижче.")
            st.dataframe(pd.DataFrame(prots[worst["id"]]).drop(columns="type"), hide_index=True)
