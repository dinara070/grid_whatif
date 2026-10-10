# -*- coding: utf-8 -*-
"""
Цифровий двійник розподільчої мережі 10 кВ:
  1) Сценарний симулятор «Що, якщо?» (N-1 / N-k) з автоматичним планом перемикань
  2) Прогнозування відмов обладнання (Predictive Maintenance) на основі ML
  3) Ризик = Імовірність відмови × Наслідки (ENS, SAIDI) -> пріоритезація ТО

Запуск:
    pip install streamlit numpy pandas networkx plotly scikit-learn
    streamlit run grid_whatif_app.py

Усі дані - синтетичні (демо). Для реальної експлуатації замініть build_grid()
на вивантаження з ГІС/SCADA/АСКОЕ, а make_training() - на історію аварій з CMMS/ЕАМ.
"""
from collections import defaultdict, deque, namedtuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Цифровий двійник РЕМ: What-If та Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
)

# ════════════════════════════════════════════════════════════════════════════
# 1. МОДЕЛЬ МЕРЕЖІ
# ════════════════════════════════════════════════════════════════════════════
# Добовий профіль навантаження (зимовий робочий день), пік о 18:00 = 1.0
HOURLY = [0.52, 0.48, 0.46, 0.45, 0.46, 0.52, 0.65, 0.80, 0.85, 0.80, 0.76, 0.75,
          0.74, 0.73, 0.74, 0.78, 0.90, 0.98, 1.00, 0.97, 0.92, 0.85, 0.72, 0.60]
PRIO_W = {1: 10.0, 2: 3.0, 3: 1.0}          # вага втрат: 1 - критичні споживачі
PRIO_TXT = {1: "1 (критичний)", 2: "2 (звичайний)", 3: "3 (обмежуваний)"}

Params = namedtuple("Params", "temp hour sens src_cap tol t_detect t_switch repair_h")


def build_grid():
    """Дві ПС 110/10 кВ, чотири фідери по 5 ТП, нормально-розімкнені перемички."""
    rng = np.random.default_rng(42)
    nodes, edges = {}, {}
    nodes["ПС-1"] = dict(type="src", x=-4.0, y=0.0, load=0.0, cons=0, prio=0, kva=0, feeder="-")
    nodes["ПС-2"] = dict(type="src", x=4.0, y=0.0, load=0.0, cons=0, prio=0, kva=0, feeder="-")
    feeders = {
        "Ф-1": ("ПС-1", -1, 0.9, 0.28, 101),
        "Ф-2": ("ПС-1", -1, -0.9, -0.28, 201),
        "Ф-3": ("ПС-2", +1, 0.9, 0.28, 301),
        "Ф-4": ("ПС-2", +1, -0.9, -0.28, 401),
    }
    seg_caps = [8.0, 7.0, 6.0, 5.5, 5.0]       # МВт, зменшується вздовж фідера
    std_kva = [250, 400, 630, 1000, 1250, 1600]
    for fname, (src, side, y0, dy, base) in feeders.items():
        prev = src
        for i in range(5):
            name = f"ТП-{base + i}"
            load = round(float(rng.uniform(0.35, 0.85)), 2)
            cands = [k for k in std_kva if 0.35 <= load / (0.95 * k / 1000) <= 0.85]
            kva = int(rng.choice(cands)) if cands else 1000
            nodes[name] = dict(
                type="tp",
                x=nodes[src]["x"] - side * 0.75 * (i + 1),
                y=y0 + dy * i,
                load=load,
                cons=int(rng.integers(90, 450)),
                prio=int(rng.choice([1, 2, 3], p=[0.15, 0.55, 0.30])),
                kva=kva,
                feeder=fname,
            )
            edges[f"{prev}–{name}"] = dict(u=prev, v=name, cap=seg_caps[i], tie=False)
            prev = name
    ties = [("ТП-105", "ТП-305"), ("ТП-205", "ТП-405"), ("ТП-105", "ТП-205"),
            ("ТП-305", "ТП-405"), ("ТП-103", "ТП-203"), ("ТП-303", "ТП-403")]
    for u, v in ties:
        edges[f"{u}–{v}"] = dict(u=u, v=v, cap=5.0, tie=True)
    return nodes, edges


NODES, EDGES = build_grid()
SOURCES = [n for n, d in NODES.items() if d["type"] == "src"]
TPS = [n for n, d in NODES.items() if d["type"] == "tp"]
CLOSED0 = frozenset(e for e, d in EDGES.items() if not d["tie"])
ALL_ELEMS = SOURCES + TPS + [e for e, d in EDGES.items() if not d["tie"]]


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


# ════════════════════════════════════════════════════════════════════════════
# 2. РОЗРАХУНОК РЕЖИМУ (радіальні потоки) І ВІДНОВЛЕННЯ ЖИВЛЕННЯ
# ════════════════════════════════════════════════════════════════════════════
def evaluate(closed, f_nodes, f_edges, scale, shed=frozenset()):
    """Будує ліс живлення від ПС по замкнених лініях і рахує потоки (МВт)."""
    adj = defaultdict(list)
    for eid in closed:
        if eid in f_edges:
            continue
        e = EDGES[eid]
        if e["u"] in f_nodes or e["v"] in f_nodes:
            continue
        adj[e["u"]].append((e["v"], eid))
        adj[e["v"]].append((e["u"], eid))
    parent, order, root = {}, [], {}
    for s in SOURCES:
        if s in f_nodes or s in parent:
            continue
        parent[s] = (None, None)
        root[s] = s
        q = deque([s])
        while q:
            n = q.popleft()
            order.append(n)
            for m, eid in adj[n]:
                if m not in parent:
                    parent[m] = (n, eid)
                    root[m] = s
                    q.append(m)
    load = {n: (0.0 if (n in shed or n in f_nodes) else NODES[n]["load"] * scale) for n in TPS}
    sub = {n: load.get(n, 0.0) for n in order}
    eflow = {}
    for n in reversed(order):
        par, eid = parent[n]
        if par is not None:
            sub[par] += sub[n]
            eflow[eid] = sub[n]
    children = defaultdict(list)
    for n, (par, _) in parent.items():
        if par is not None:
            children[par].append(n)
    return dict(parent=parent, order=order, root=root, load=load, sub=sub,
                eflow=eflow, children=dict(children), energized=set(order))


def downstream(st_, node):
    out, stack = [], [node]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(st_["children"].get(n, []))
    return out


def worst_violation(st_, p):
    worst = None
    for eid, f in st_["eflow"].items():
        r = f / (EDGES[eid]["cap"] * (1 + p.tol))
        if r > 1 + 1e-9 and (worst is None or r > worst[0]):
            worst = (r, "e", eid)
    for s in SOURCES:
        if s in st_["sub"]:
            r = st_["sub"][s] / (p.src_cap * (1 + p.tol))
            if r > 1 + 1e-9 and (worst is None or r > worst[0]):
                worst = (r, "s", s)
    return worst


def count_violations(st_, p):
    c = sum(1 for eid, f in st_["eflow"].items() if f > EDGES[eid]["cap"] * (1 + p.tol) + 1e-9)
    c += sum(1 for s in SOURCES if s in st_["sub"] and st_["sub"][s] > p.src_cap * (1 + p.tol) + 1e-9)
    return c


def max_loading(st_, p):
    m = 0.0
    for eid, f in st_["eflow"].items():
        m = max(m, f / EDGES[eid]["cap"])
    for s in SOURCES:
        if s in st_["sub"]:
            m = max(m, st_["sub"][s] / p.src_cap)
    return m


def balance(closed, f_nodes, f_edges, scale, p, shed):
    """Знімає перевантаження графіком обмежень: спершу найменш важливі (prio 3)."""
    shed = set(shed)
    st_ = evaluate(closed, f_nodes, f_edges, scale, shed)
    for _ in range(80):
        w = worst_violation(st_, p)
        if w is None:
            break
        if w[1] == "e":
            child = next(n for n, (_, eid) in st_["parent"].items() if eid == w[2])
            cand = downstream(st_, child)
        else:
            cand = [n for n in st_["order"] if st_["root"][n] == w[2]]
        cand = [n for n in cand if n in NODES and NODES[n]["type"] == "tp"
                and n not in shed and st_["load"][n] > 0]
        if not cand:
            break
        pick = max(cand, key=lambda n: (NODES[n]["prio"], st_["load"][n]))
        shed.add(pick)
        st_ = evaluate(closed, f_nodes, f_edges, scale, shed)
    return shed, st_


def wcost(nodes, scale):
    return sum(NODES[n]["load"] * scale * PRIO_W[NODES[n]["prio"]] for n in nodes)


def solve(P, elements):
    """Повний сценарій: аварія -> ізоляція -> відновлення перемичками -> ГОП."""
    p = Params(*P)
    f_nodes = {e for e in elements if e in NODES}
    f_edges = {e for e in elements if e in EDGES}
    scale = load_scale(p)
    closed0 = set(CLOSED0)
    normal = evaluate(closed0, frozenset(), frozenset(), scale)
    closed_f = closed0 - f_edges
    post = evaluate(closed_f, f_nodes, f_edges, scale)

    steps, ops = [], 0

    def add(typ, action, obj, why, inc=1, t=None):
        nonlocal ops
        ops += inc
        steps.append({"type": typ, "№": len(steps) + 1,
                      "Час, хв": round(t if t is not None else p.t_detect + ops * p.t_switch, 1),
                      "Дія": action, "Об'єкт": obj, "Обґрунтування": why})

    steps.append({"type": "detect", "№": 1, "Час, хв": p.t_detect,
                  "Дія": "Фіксація відмови та локалізація пошкодження",
                  "Об'єкт": ", ".join(sorted(elements)),
                  "Обґрунтування": "Сигнали РЗА/ТС, телеметрія SCADA"})
    # ізоляція пошкодженої ділянки
    iso = {}
    for n in sorted(f_nodes):
        for eid in sorted(closed_f):
            if n in (EDGES[eid]["u"], EDGES[eid]["v"]):
                iso[eid] = (1, f"ізоляція {n}")
    for eid in sorted(f_edges & closed0):
        iso[eid] = (2, "ізоляція пошкодженого сегмента (обидва кінці)")
    for eid, (inc, why) in iso.items():
        add("isolate", "Відкрити комутаційний апарат", eid, why, inc)

    closed, shed, st_cur = set(closed_f), set(), post
    restore_min = {}
    for _ in range(25):
        unener = set(TPS) - st_cur["energized"] - f_nodes
        H = nx.Graph()
        H.add_nodes_from(unener)
        for eid in closed:
            if eid in f_edges:
                continue
            e = EDGES[eid]
            if e["u"] in unener and e["v"] in unener:
                H.add_edge(e["u"], e["v"])
        comps = sorted(nx.connected_components(H),
                       key=lambda c: (min(NODES[n]["prio"] for n in c),
                                      -sum(NODES[n]["load"] for n in c)))
        progressed = False
        for comp in comps:
            en = st_cur["energized"]
            cands = []
            for eid, e in EDGES.items():
                if eid in closed or eid in f_edges:
                    continue
                if (e["u"] in comp and e["v"] in en) or (e["v"] in comp and e["u"] in en):
                    cands.append(eid)
            base_cost = wcost(shed | set(comp), scale)
            best = None
            for tie in cands:
                shed2, st2 = balance(closed | {tie}, f_nodes, f_edges, scale, p, shed)
                key = (wcost(shed2, scale), max_loading(st2, p), tie)
                if best is None or key < best[0]:
                    best = (key, tie, shed2, st2)
            if best and best[0][0] < base_cost - 1e-9:
                _, tie, shed2, st2 = best
                for n in sorted(shed2 - shed):
                    add("shed", "Відключити ТП (графік обмежень, ГОП)", n,
                        f"Усунення перевантаження; пріоритет {PRIO_TXT[NODES[n]['prio']]}, "
                        f"{NODES[n]['load'] * scale:.2f} МВт")
                e = EDGES[tie]
                entry = e["v"] if e["u"] in comp else e["u"]
                add("close", "Закрити секційний вимикач (замкнути перемичку)", tie,
                    f"Відновлення {len(comp)} ТП ({sum(NODES[n]['cons'] for n in comp)} спож., "
                    f"{sum(NODES[n]['load'] for n in comp) * scale:.2f} МВт) від {st2['root'][entry]}; "
                    f"макс. завантаження {max_loading(st2, p):.0%}")
                for n in comp:
                    restore_min[n] = p.t_detect + ops * p.t_switch
                closed.add(tie)
                shed, st_cur = shed2, st2
                progressed = True
                break
        if not progressed:
            break

    status = {}
    for n in TPS:
        if n in f_nodes:
            status[n] = "failed"
        elif n in shed:
            status[n] = "shed"
        elif n not in st_cur["energized"]:
            status[n] = "lost"
        elif n not in post["energized"]:
            status[n] = "restored"
        else:
            status[n] = "ok"
    lost = [n for n in TPS if status[n] == "lost"]
    if lost:
        steps.append({"type": "info", "№": len(steps) + 1, "Час, хв": round(p.repair_h * 60, 1),
                      "Дія": "Живлення відсутнє до завершення ремонту", "Об'єкт": ", ".join(lost),
                      "Обґрунтування": "Немає резервного шляху від живих ПС"})
    steps.append({"type": "repair", "№": len(steps) + 1, "Час, хв": round(p.repair_h * 60, 1),
                  "Дія": "Ремонт/заміна обладнання, повернення до нормальної схеми",
                  "Об'єкт": ", ".join(sorted(elements)),
                  "Обґрунтування": "Виїзд ОВБ, заява на вивід, потім розмикання перемичок"})

    tot_mw = sum(NODES[n]["load"] for n in TPS) * scale
    tot_cons = sum(NODES[n]["cons"] for n in TPS)
    unsup = [n for n in TPS if status[n] in ("failed", "lost", "shed")]
    interrupted = [n for n in TPS if status[n] != "ok"]
    ens = cmi = 0.0
    for n in interrupted:
        dur = (restore_min[n] / 60.0) if status[n] == "restored" else p.repair_h
        ens += NODES[n]["load"] * scale * dur
        cmi += NODES[n]["cons"] * dur * 60.0
    return dict(
        scale=scale, total_mw=tot_mw, unsup_mw=sum(NODES[n]["load"] for n in unsup) * scale,
        ens=ens, cust_cut=sum(NODES[n]["cons"] for n in interrupted),
        cust_unrestored=sum(NODES[n]["cons"] for n in unsup), saidi=cmi / tot_cons,
        steps=steps, status=status, normal=normal, post=post, final=st_cur,
        closed0=closed0, closed_f=closed_f, closed_fin=closed, shed=sorted(shed),
        f_nodes=f_nodes, f_edges=f_edges, restore_min=restore_min,
        max_line=max_loading(st_cur, p), viol=count_violations(st_cur, p),
        src_load={s: (st_cur["sub"].get(s, 0.0) / p.src_cap) for s in SOURCES},
    )


solve_c = st.cache_data(show_spinner="Розрахунок сценарію…")(solve)


@st.cache_data(show_spinner="Розрахунок N-1 для всіх елементів мережі…")
def scan_all(P):
    rows, prots = [], {}
    for el in ALL_ELEMS:
        r = solve(P, (el,))
        prots[el] = r["steps"]
        acts = [s["Дія"].split(" (")[0] + " " + s["Об'єкт"]
                for s in r["steps"] if s["type"] in ("close", "shed")]
        rows.append(dict(
            id=el, type=elem_type(el), unsup_mw=r["unsup_mw"],
            unsup_pct=100 * r["unsup_mw"] / r["total_mw"], ens_mwh=r["ens"],
            cust=r["cust_unrestored"], saidi=r["saidi"], max_line=100 * r["max_line"],
            n_ops=len(acts), plan="; ".join(acts) if acts else "перемикання не потрібні",
        ))
    return pd.DataFrame(rows), prots


@st.cache_data(show_spinner="Побудова карти чутливості…")
def sens_grid(P, elements, hours, temps):
    z = []
    for t in temps:
        z.append([solve((float(t), int(h)) + tuple(P[2:]), elements)["unsup_mw"] for h in hours])
    return z


# ════════════════════════════════════════════════════════════════════════════
# 3. ВІЗУАЛІЗАЦІЯ СХЕМИ
# ════════════════════════════════════════════════════════════════════════════
PHASES = ["1. Норма", "2. Одразу після аварії", "3. Після перемикань"]
STATUS_STYLE = {
    "ok": ("Живиться нормально", "#2e7d32"),
    "restored": ("Відновлено перемиканням", "#1e88e5"),
    "shed": ("Відключено (ГОП)", "#fb8c00"),
    "lost": ("Без живлення", "#757575"),
    "failed": ("Пошкоджено", "#d32f2f"),
}


def ratio_color(r):
    if r > 1.0:
        return "#d32f2f"
    if r > 0.8:
        return "#fb8c00"
    if r > 0.6:
        return "#fbc02d"
    return "#43a047"


def draw_map(res, phase):
    if phase == PHASES[0]:
        stt, closed = res["normal"], res["closed0"]
        status = {n: "ok" for n in TPS}
    elif phase == PHASES[1]:
        stt, closed = res["post"], res["closed_f"]
        status = {n: ("failed" if n in res["f_nodes"] else
                      "lost" if n not in stt["energized"] else "ok") for n in TPS}
    else:
        stt, closed, status = res["final"], res["closed_fin"], res["status"]

    fig = go.Figure()
    mx, my, mt = [], [], []
    for eid, e in EDGES.items():
        a, b = NODES[e["u"]], NODES[e["v"]]
        flow = stt["eflow"].get(eid)
        if eid in res["f_edges"]:
            color, dash, w, txt = "#000000", "dot", 4, f"{eid}<br>ПОШКОДЖЕНО"
        elif eid not in closed:
            color, dash, w, txt = "#9aa0a6", "dash", 1.5, f"{eid}<br>Розімкнена перемичка (ліміт {e['cap']} МВт)"
        elif flow is None:
            color, dash, w, txt = "#cfcfcf", "solid", 2, f"{eid}<br>Без напруги"
        else:
            r = flow / e["cap"]
            color, dash, w = ratio_color(r), "solid", 2 + 4 * min(r, 1.2)
            txt = f"{eid}<br>Потік {flow:.2f} МВт / ліміт {e['cap']} МВт<br>Завантаження {r:.0%}"
        fig.add_trace(go.Scatter(x=[a["x"], b["x"]], y=[a["y"], b["y"]], mode="lines",
                                 line=dict(color=color, width=w, dash=dash),
                                 hoverinfo="skip", showlegend=False))
        mx.append((a["x"] + b["x"]) / 2)
        my.append((a["y"] + b["y"]) / 2)
        mt.append(txt)
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", hovertext=mt, hoverinfo="text",
                             marker=dict(size=12, color="rgba(0,0,0,0.01)"), showlegend=False))

    for key, (label, color) in STATUS_STYLE.items():
        names = [n for n in TPS if status[n] == key]
        if not names:
            continue
        fig.add_trace(go.Scatter(
            x=[NODES[n]["x"] for n in names], y=[NODES[n]["y"] for n in names],
            mode="markers+text", name=label, text=[n.replace("ТП-", "") for n in names],
            textposition="top center", textfont=dict(size=10),
            marker=dict(size=16, color=color, line=dict(color="white", width=1.5)),
            hovertext=[f"{n}<br>{label}<br>Навантаження: {NODES[n]['load'] * res['scale']:.2f} МВт"
                       f"<br>Споживачів: {NODES[n]['cons']}<br>Пріоритет: {PRIO_TXT[NODES[n]['prio']]}"
                       f"<br>Трансформатор: {NODES[n]['kva']} кВА" for n in names],
            hoverinfo="text"))
    fig.add_trace(go.Scatter(
        x=[NODES[s]["x"] for s in SOURCES], y=[NODES[s]["y"] for s in SOURCES],
        mode="markers+text", name="Джерело (ПС)", text=SOURCES, textposition="bottom center",
        marker=dict(size=26, symbol="square", line=dict(color="white", width=2),
                    color=["#d32f2f" if s in res["f_nodes"] else "#6a1b9a" for s in SOURCES]),
        hovertext=[f"{s}<br>Завантаження: {res_src_load(res, phase, s):.0%}" for s in SOURCES],
        hoverinfo="text"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.05), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def res_src_load(res, phase, s):
    stt = {PHASES[0]: res["normal"], PHASES[1]: res["post"]}.get(phase, res["final"])
    return stt["sub"].get(s, 0.0) / SRC_CAP


# ════════════════════════════════════════════════════════════════════════════
# 4. PREDICTIVE MAINTENANCE
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
    """Синтетична історія «актив-рік -> відмова протягом року». Замініть реальними даними."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, size=n, p=[0.7, 0.1, 0.2])        # 0 ТП, 1 ПС, 2 лінія
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
    rows = []
    for n in SOURCES:
        rows.append((n, "ps", int(rng.integers(12, 42))))
    for n in TPS:
        rows.append((n, "tp", int(rng.integers(4, 48))))
    for eid, e in EDGES.items():
        if not e["tie"]:
            rows.append((eid, "line", int(rng.integers(3, 52))))
    out = []
    for id_, kind, age in rows:
        out.append(dict(id=id_, kind=kind, age=age, prior_faults=int(rng.poisson(0.1 + age / 45)),
                        years_since_maint=round(float(rng.uniform(0, 10)), 1),
                        condition_idx=float(np.clip(95 - 0.9 * age + rng.normal(0, 9), 12, 99)),
                        u=float(rng.uniform(0.2, 0.8))))
    return pd.DataFrame(out)


STATIC_DF = static_registry()


def registry_features(P, growth, winter_days):
    p = Params(*P)
    ev = evaluate(set(CLOSED0), frozenset(), frozenset(), 1.0)
    nominal = []
    for _, r in STATIC_DF.iterrows():
        if r["kind"] == "tp":
            nominal.append(NODES[r["id"]]["load"] / (NODES[r["id"]]["kva"] * 0.95 / 1000))
        elif r["kind"] == "ps":
            nominal.append(ev["sub"][r["id"]] / p.src_cap)
        else:
            nominal.append(ev["eflow"].get(r["id"], 0.0) / EDGES[r["id"]]["cap"])
    peak = np.array(nominal) * temp_factor(p.temp, p.sens) * (1 + growth)
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
# 5. ІНТЕРФЕЙС
# ════════════════════════════════════════════════════════════════════════════
st.title("⚡ Цифровий двійник мережі 10 кВ: «Що, якщо?» та прогноз відмов")
st.caption("Демонстраційні синтетичні дані. Радіальна мережа: 2 ПС 110/10 кВ, 4 фідери, 20 ТП, "
           "6 нормально-розімкнених перемичок.")

with st.sidebar:
    st.header("⚙️ Умови режиму")
    temp = st.slider("Температура повітря, °C", -30, 15, -10)
    hour = st.slider("Година доби", 0, 23, 18)
    sens = st.slider("Температурна чутливість навантаження, %/°C", 0.0, 3.0, 1.2, 0.1)
    src_cap = st.number_input("Допустима потужність кожної ПС, МВт", 5.0, 40.0, 12.0, 0.5)
    tol = st.slider("Допустиме перевантаження ліній/ПС, %", 0, 20, 0)
    st.header("⏱️ Час та ремонт")
    t_detect = st.number_input("Виявлення і локалізація, хв", 1.0, 60.0, 5.0, 1.0)
    t_switch = st.number_input("Час однієї комутації, хв", 1.0, 60.0, 4.0, 1.0)
    repair_h = st.number_input("Тривалість ремонту, год", 1.0, 72.0, 8.0, 0.5)
    st.caption(f"Коефіцієнт навантаження зараз: **{HOURLY[hour] * temp_factor(temp, sens):.2f}** "
               f"від номінального піку.")

P = (float(temp), int(hour), float(sens), float(src_cap), tol / 100.0,
     float(t_detect), float(t_switch), float(repair_h))
SRC_CAP = float(src_cap)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Сценарний симулятор", "📚 Бібліотека протоколів N-1",
    "🔮 Прогноз відмов (Predictive Maintenance)", "⚖️ Ризик = P × наслідки"])

# ─────────────────────────── TAB 1 ───────────────────────────
with tab1:
    st.subheader(f"Що станеться, якщо елемент вийде з ладу о {hour}:00 при {temp:+d} °C?")
    sel = st.multiselect("Елемент(и), що виходять з ладу (можна кілька - сценарій N-k)",
                         ALL_ELEMS, default=["ТП-103"], format_func=fmt_elem)
    if not sel:
        st.info("Оберіть хоча б один елемент.")
    else:
        res = solve_c(P, tuple(sel))
        c = st.columns(6)
        c[0].metric("Загальне навантаження", f"{res['total_mw']:.1f} МВт")
        c[1].metric("Без живлення після перемикань", f"{res['unsup_mw']:.2f} МВт",
                    f"{100 * res['unsup_mw'] / res['total_mw']:.1f}% навантаження", delta_color="inverse")
        c[2].metric("Споживачів без живлення", f"{res['cust_unrestored']:,}".replace(",", " "),
                    f"{res['cust_cut']} зазнали переривання", delta_color="off")
        c[3].metric("Недовідпуск (ENS)", f"{res['ens']:.1f} МВт·год")
        c[4].metric("SAIDI сценарію", f"{res['saidi']:.0f} хв/спож.")
        c[5].metric("Макс. завантаження", f"{res['max_line']:.0%}")

        if res["viol"] > 0:
            st.error(f"Після перемикань залишилось {res['viol']} перевантажених елементів - потрібні "
                     "додаткові обмеження або мобільна генерація.")
        elif res["unsup_mw"] <= 1e-9:
            st.success("Живлення всіх споживачів можна відновити перемиканнями без перевантажень.")
        else:
            st.warning("Частину споживачів неможливо повернути до завершення ремонту.")

        phase = st.radio("Стан схеми", PHASES, index=2, horizontal=True)
        st.plotly_chart(draw_map(res, phase), key="map")
        st.caption("Лінії: 🟢 <60% · 🟡 60–80% · 🟠 80–100% · 🔴 >100% від допустимого струму; "
                   "пунктир сірий - розімкнена перемичка; чорний пунктир - пошкоджено.")

        st.markdown("#### 📋 Протокол перемикань (готовий для диспетчера)")
        proto = pd.DataFrame(res["steps"]).drop(columns="type")
        st.dataframe(proto, hide_index=True)
        st.download_button("⬇️ Завантажити протокол (CSV)", proto.to_csv(index=False).encode("utf-8-sig"),
                           file_name="protocol_" + "_".join(sel) + ".csv", mime="text/csv")

        st.markdown("#### Завантаження ліній за етапами")
        rows = []
        for eid, e in EDGES.items():
            if e["tie"] and eid not in res["closed_fin"]:
                continue
            g = lambda s: 100 * s["eflow"].get(eid, 0.0) / e["cap"]
            rows.append({"Лінія": eid, "Ліміт, МВт": e["cap"], "Норма, %": g(res["normal"]),
                         "Після аварії, %": g(res["post"]), "Після перемикань, %": g(res["final"]),
                         "Потік після, МВт": res["final"]["eflow"].get(eid, 0.0)})
        ldf = pd.DataFrame(rows).sort_values("Після перемикань, %", ascending=False)
        pc = lambda: st.column_config.ProgressColumn(min_value=0, max_value=150, format="%.0f%%")
        st.dataframe(ldf, hide_index=True, column_config={
            "Норма, %": pc(), "Після аварії, %": pc(), "Після перемикань, %": pc(),
            "Потік після, МВт": st.column_config.NumberColumn(format="%.2f")})
        sl = pd.DataFrame({"ПС": SOURCES, "Завантаження після, %": [100 * res["src_load"][s] for s in SOURCES]})
        st.dataframe(sl, hide_index=True,
                     column_config={"Завантаження після, %": pc()})

        with st.expander("🌡️ Чутливість: як результат залежить від температури та години"):
            hours_t, temps_t = (6, 9, 12, 15, 18, 21), (-25, -20, -15, -10, -5, 0, 5, 10)
            z = sens_grid(P, tuple(sel), hours_t, temps_t)
            fig = go.Figure(go.Heatmap(z=z, x=[f"{h}:00" for h in hours_t], y=[f"{t:+d} °C" for t in temps_t],
                                       colorscale="YlOrRd", texttemplate="%{z:.1f}",
                                       colorbar=dict(title="МВт без<br>живлення")))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              title="Невідновлене навантаження після перемикань, МВт")
            st.plotly_chart(fig, key="sens")

# ─────────────────────────── TAB 2 ───────────────────────────
with tab2:
    st.subheader("Бібліотека готових протоколів: N-1 для кожного елемента мережі")
    st.caption("Розраховано для поточних умов (бічна панель). Диспетчер заздалегідь бачить, "
               "які елементи найкритичніші та що робити.")
    scan_df, prots = scan_all(P)
    show = scan_df.sort_values("ens_mwh", ascending=False).rename(columns={
        "id": "Елемент", "type": "Тип", "unsup_mw": "Без живлення, МВт", "unsup_pct": "Без живлення, %",
        "ens_mwh": "ENS, МВт·год", "cust": "Споживачів без живлення", "saidi": "SAIDI, хв/спож.",
        "max_line": "Макс. завант., %", "n_ops": "Операцій", "plan": "Ключові перемикання"})
    st.dataframe(show, hide_index=True, column_config={
        "Без живлення, МВт": st.column_config.NumberColumn(format="%.2f"),
        "Без живлення, %": st.column_config.NumberColumn(format="%.1f"),
        "ENS, МВт·год": st.column_config.NumberColumn(format="%.1f"),
        "SAIDI, хв/спож.": st.column_config.NumberColumn(format="%.0f"),
        "Макс. завант., %": st.column_config.ProgressColumn(min_value=0, max_value=150, format="%.0f%%")})
    allp = pd.concat([pd.DataFrame(v).drop(columns="type").assign(Елемент_відмови=k)
                      for k, v in prots.items()])
    st.download_button("⬇️ Усі протоколи (CSV)", allp.to_csv(index=False).encode("utf-8-sig"),
                       file_name="all_N1_protocols.csv", mime="text/csv")
    pick = st.selectbox("Переглянути протокол для елемента", scan_df.sort_values("ens_mwh", ascending=False)["id"],
                        format_func=fmt_elem)
    st.dataframe(pd.DataFrame(prots[pick]).drop(columns="type"), hide_index=True)

# ─────────────────────────── TAB 3 ───────────────────────────
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
    st.caption("Модель: GradientBoostingClassifier. Імовірність за горизонт: 1 − (1 − p_рік)^(міс./12). "
               "Синтетична історія слугує лише демонстрацією - для практики підключіть дані АСКОЕ, "
               "журнали ТО/ХАРГ та історію відмов.")

# ─────────────────────────── TAB 4 ───────────────────────────
with tab4:
    st.subheader("Пріоритезація: ймовірність відмови × наслідки для споживачів")
    st.caption("Наслідки беруться зі сценарного симулятора (N-1 для поточних умов), "
               "імовірності - з моделі прогнозу. Параметри - з вкладки «Прогноз відмов».")
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
