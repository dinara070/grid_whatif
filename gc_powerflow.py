# -*- coding: utf-8 -*-
"""Розрахунок режимів (pandapower), відновлення живлення, сценарії N-1/N-k. Без Streamlit.

Основні точки входу: solve(P, elements) / solve_c (з кешем), scan_all(P), sens_grid(...), daily_profile(P).
"""
import copy
import logging
import math
import warnings
from collections import defaultdict, deque
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp

from gc_model import *  # noqa: F401,F403  (константи мережі, Params, профілі)

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
