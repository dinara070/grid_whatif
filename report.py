# -*- coding: utf-8 -*-
"""PDF-звіт диспетчера/керівника: умови, показники, схема, протокол перемикань, напруги, лінії, РЗА, ризики.

fpdf2 + matplotlib (Figure без pyplot - безпечно для потоків; шрифт DejaVu постачається з matplotlib,
тому кирилиця працює на будь-якій ОС). Без Streamlit.
"""
import io
import os
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.fonts import FontFace
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .model import EDGES, NODES, SOURCES, TPS
from .viz import STATUS_STYLE, ratio_color


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
