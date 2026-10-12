# -*- coding: utf-8 -*-
"""Модель мережі та профілі (чиста логіка, без pandas/Streamlit): умовна мережа 10 кВ у Вінницькій області.

build_grid() - топологія й параметри; Params/make_params() - параметри режиму; профілі навантаження, PV та АБ.
"""
import math
from collections import namedtuple

import numpy as np

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
