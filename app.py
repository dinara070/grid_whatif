# -*- coding: utf-8 -*-
"""
Цифровий двійник розподільчої мережі 10 кВ (v2: pandapower + ВДЕ/АБ + КЗ)

  1) Сценарний симулятор «Що, якщо?» (N-1 / N-k) із реальним розрахунком режимів pandapower:
     P, Q, напруги у вузлах, втрати, завантаження ліній і трансформаторів ПС.
  2) ВДЕ та накопичувачі: PV на ТП, акумуляторні батареї (АБ), двосторонні потоки,
     острівний режим (АБ формує мережу) під час аварії.
  3) Струми КЗ (pandapower.shortcircuit) і перевірка уставок релейного захисту
     (ТВ, МСЗ, вводи ПС), у т. ч. хибні спрацювання на резервних фідерах після перемикань.
  4) Прогноз відмов обладнання (ML) та ризик = P × наслідки.

Запуск:  pip install -r requirements.txt  &&  streamlit run app.py

Архітектура розрахунку: швидкий лінеаризований прескринінг (радіальна модель) лише добирає
кандидатів (перемички, ГОП, острівний режим). Усі показники, які бачить користувач, та
перевірка допустимості й доведення режиму виконуються pandapower (Newton-Raphson).
Дані синтетичні (демо).

ВАЖЛИВО: у requirements.txt має бути  pandas<3  (pandapower < 3.2 несумісний з pandas 3.x,
runpp падає з "assignment destination is read-only").
"""
import copy
import logging
import math
import warnings
from collections import defaultdict, deque, namedtuple

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.shortcircuit as sc
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

st.set_page_config(
    page_title="Цифровий двійник РЕМ: Power Flow, ВДЕ, КЗ, Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
)

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
    "der pv_k ins bess_mode soc island")

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


def elem_type(e):
    if e in NODES:
        return "ПС" if NODES[e]["type"] == "src" else "ТП"
    return "Лінія"


def fmt_elem(e):
    return {"ПС": "⚡ ", "ТП": "🏠 ", "Лінія": "〰️ Лінія "}[elem_type(e)] + e


def temp_factor(temp, sens):
    return 1.0 + sens / 100.0 * max(0.0, 5.0 - temp)


def load_scale(p):
    return HOURLY[int(p.hour)] * temp_factor(p.temp, p.sens)


def pv_profile(h):
    """Зимова добова крива PV (частка від встановленої потужності, ясно)."""
    return max(0.0, math.sin(math.pi * (h - 7.5) / 9.5)) * 0.7 if 7.5 < h < 17 else 0.0


def bess_frac(h, mode, soc):
    """Режим АБ у нормі: 0 - резерв; 1 - пікове голення (розряд 17-21 год, заряд 10-14 год)."""
    if mode == 0:
        return 0.0
    if 17 <= h <= 21 and soc > 0.2:
        return 0.8
    if 10 <= h <= 14 and soc < 0.95:
        return -0.5
    return 0.0


# ════════════════════════════════════════════════════════════════════════════
# 2. PANDAPOWER: БАЗОВА МОДЕЛЬ ТА СТАН
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Побудова моделі pandapower…")
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
    return d["pv"] * p.pv_k * pv_profile(p.hour) * p.ins / 100.0


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
                    f"Усунення порушень режиму; пріоритет {PRIO_TXT[NODES[n]['prio']]}, "
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
                f"Порушення за розрахунком pandapower; пріоритет {PRIO_TXT[NODES[fix[1]]['prio']]}, "
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


solve_c = st.cache_data(show_spinner="Розрахунок сценарію (pandapower)…")(solve)


@st.cache_data(show_spinner="Розрахунок N-1 для всіх елементів мережі (pandapower)…")
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


@st.cache_data(show_spinner="Побудова карти чутливості…")
def sens_grid(P, elements, hours, temps):
    return [[solve((float(t), int(h)) + tuple(P[2:]), elements)["unsup_mw"] for h in hours] for t in temps]


@st.cache_data(show_spinner="Розрахунок доби (24 режими pandapower)…")
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


@st.cache_data(show_spinner="Розрахунок уставок захисту…")
def protection_settings(P):
    p = Params(*P)._replace(temp=-20.0, hour=18, der=0)
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


@st.cache_data(show_spinner="Розрахунок струмів КЗ (pandapower.shortcircuit)…")
def fault_report(P, elements):
    p = Params(*P)
    res = solve(P, elements)
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
                f"<br>Споживачів: {NODES[n]['cons']}<br>Пріоритет: {PRIO_TXT[NODES[n]['prio']]}")

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


@st.cache_data
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


@st.cache_resource(show_spinner="Навчання моделі прогнозу відмов…")
def get_model(df):
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


def registry_features(P, growth, winter_days):
    """Завантаження активів беремо з pandapower (номінальний режим без ВДЕ), далі масштабуємо."""
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
    peak = np.array(nominal) * temp_factor(p0.temp, p0.sens) * (1 + growth)
    prof = np.array(HOURLY)
    df = STATIC_DF.copy()
    df["loading_peak"] = peak
    df["loading_mean"] = peak * prof.mean()
    df["overload_hours"] = winter_days * ((peak[:, None] * prof[None, :]) > 1.0).sum(axis=1)
    df["cold_hours"] = (winter_days * 24 * df["u"]).round()
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
# 8. ІНТЕРФЕЙС
# ════════════════════════════════════════════════════════════════════════════
st.title("⚡ Цифровий двійник мережі 10 кВ: Power Flow, ВДЕ, КЗ та прогноз відмов")
st.caption("Демонстраційні синтетичні дані. 2 ПС 110/10 кВ, 4 фідери, 20 ТП, 6 нормально-розімкнених перемичок, "
           "6 PV-станцій та 3 акумуляторні батареї. Режими рахує pandapower (Newton-Raphson).")

with st.sidebar:
    st.header("⚙️ Умови режиму")
    temp = st.slider("Температура повітря, °C", -30, 15, -10)
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
    st.caption(f"Коефіцієнт навантаження: **{HOURLY[hour] * temp_factor(temp, sens):.2f}** · "
               f"PV зараз: **{pv_profile(hour) * ins / 100:.0%}** від встановленої.")

P = (float(temp), int(hour), float(sens), float(src_cap), tol / 100.0, float(t_detect), float(t_switch),
     float(repair_h), float(vmin), float(vmax), float(u_ps), float(cosphi), int(der), float(pv_k), float(ins),
     0 if bess_mode_txt.startswith("Резерв") else 1, soc / 100.0, int(island))
PP = Params(*P)

# --- діагностика: якщо Power Flow не рахується (наприклад, несумісні версії pandas/pandapower) ---
_chk = pp_state(new_net(PP), PP, topology(set(CLOSED0), E0, E0), set(CLOSED0), E0, E0,
                load_scale(PP), set(), set(), ())
if not _chk["conv"]:
    st.error("⚠️ Power Flow (pandapower) не вдалося виконати: "
             f"`{_chk['err'] or 'не зійшовся'}`. Якщо в тексті є «read-only», додайте у requirements.txt "
             "рядок `pandas<3` (або оновіть pandapower) і перезапустіть застосунок.")
    st.stop()

tabs = st.tabs(["🔥 Сценарний симулятор", "⚡ КЗ та релейний захист", "☀️ ВДЕ та добовий режим",
                "📚 Бібліотека протоколів N-1", "🔮 Прогноз відмов", "⚖️ Ризик = P × наслідки"])
tab1, tab_kz, tab_der, tab2, tab3, tab4 = tabs

# ─────────────────────────── TAB 1: СИМУЛЯТОР ───────────────────────────
with tab1:
    st.subheader(f"Що станеться, якщо елемент вийде з ладу о {hour}:00 при {temp:+d} °C?")
    sel = st.multiselect("Елемент(и), що виходять з ладу (можна кілька - сценарій N-k)",
                         ALL_ELEMS, default=["ТП-103"], format_func=fmt_elem)
    res = None
    if not sel:
        st.info("Оберіть хоча б один елемент.")
    else:
        res = dict(solve_c(P, tuple(sel)))
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
        st.plotly_chart(draw_map(res, phase, color_by), key="map")
        st.caption("Лінії: 🟢 <60% · 🟡 60–80% · 🟠 80–100% · 🔴 >100% від допустимого струму; пунктир сірий - "
                   "розімкнена перемичка; чорний пунктир - пошкоджено. ★ - PV, ⬢ - АБ.")
        st.plotly_chart(voltage_profile(res, phase, PP), key="vprof")

        st.markdown("#### 📋 Протокол перемикань (готовий для диспетчера)")
        proto = pd.DataFrame(res["steps"]).drop(columns="type")
        st.dataframe(proto, hide_index=True)
        st.download_button("⬇️ Завантажити протокол (CSV)", proto.to_csv(index=False).encode("utf-8-sig"),
                           file_name="protocol_" + "_".join(sel) + ".csv", mime="text/csv")

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
                z = sens_grid(P, tuple(sel), hours_t, temps_t)
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
        feeders, inp = protection_settings(P)
        faults, post_df, in_df, isl = fault_report(P, tuple(sel))
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
    dp = daily_profile(P)
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

# ─────────────────────────── TAB 2: N-1 ───────────────────────────
scan_df, prots = None, None
with tab2:
    st.subheader("Бібліотека готових протоколів: N-1 для кожного елемента мережі")
    st.caption("Розраховує режим pandapower для всіх 42 елементів (≈10–40 с залежно від ПК). Результат кешується, "
               "після зміни параметрів у бічній панелі розрахунок повторюється.")
    if st.button("▶️ Розрахувати N-1 для всієї мережі") or st.session_state.get("scan_req"):
        st.session_state["scan_req"] = True
        scan_df, prots = scan_all(P)
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
            udf = pd.read_csv(up)
            if set(FEATURES + ["failed"]).issubset(udf.columns) and udf["failed"].nunique() == 2:
                train_df = udf[FEATURES + ["failed"]].dropna()
                st.success(f"Модель навчена на ваших даних: {len(train_df)} записів.")
            else:
                st.warning("У файлі бракує потрібних стовпців - використано синтетичні дані.")
        except Exception as ex:  # noqa: BLE001
            st.warning(f"Не вдалося прочитати файл: {ex}")
    model, auc, imp, ref = get_model(train_df)

    reg = registry_features(P, growth / 100.0, wdays)
    reg["p_year"] = model.predict_proba(reg[FEATURES].astype(float))[:, 1]
    reg["p_h"] = 1 - (1 - reg["p_year"]) ** (horizon / 12.0)
    reg["cat"] = reg["p_year"].map(category)
    reg["reco"] = reg["cat"].map(RECO)

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

    st.markdown("#### 🔎 Чому ризик саме такий?")
    asset = st.selectbox("Актив", reg.sort_values("p_h", ascending=False)["id"], format_func=fmt_elem)
    row = reg[reg["id"] == asset].iloc[0]
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
    st.caption("Модель: GradientBoostingClassifier. Завантаження активів беруться з розрахунку pandapower. "
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
        mg = reg.merge(scan_df[["id", "ens_mwh", "unsup_mw", "cust", "saidi"]], on="id")
        mg["exp_ens"] = mg["p_h"] * mg["ens_mwh"]
        mg["exp_saidi"] = mg["p_h"] * mg["saidi"]
        mg = mg.sort_values("exp_ens", ascending=False)
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
