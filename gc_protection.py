# -*- coding: utf-8 -*-
"""Струми КЗ (pandapower.shortcircuit) та перевірка уставок релейного захисту. Без Streamlit."""
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import pandapower.shortcircuit as sc

from gc_model import *  # noqa: F401,F403
from gc_powerflow import *  # noqa: F401,F403

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
