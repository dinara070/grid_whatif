# -*- coding: utf-8 -*-
"""Plotly-фігури мережі (схема, карта OSM, профіль напруг) та matplotlib-фігури SHAP. Без Streamlit."""
import math

import numpy as np
import plotly.graph_objects as go

from gc_model import *  # noqa: F401,F403

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
