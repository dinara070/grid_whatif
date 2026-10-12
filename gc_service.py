# -*- coding: utf-8 -*-
"""Сервісний шар: єдина точка входу для UI (Streamlit) та API (FastAPI). Без Streamlit.

Усе, що потрібно інтерфейсу, але не є малюванням: побудова параметрів із прогнозом, зведення результатів,
таблиці ризиків, PDF-звіт, JSON-сумісні підсумки.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gc_forecast import forecast_load
from gc_model import ALL_ELEMS, DISTRICTS, EDGES, NODES, Params, make_params
from gc_powerflow import scan_all, solve_c
from gc_predictive import get_model, make_training, registry_features, risk_register, score_registry
from gc_protection import fault_report
from gc_report import render_report


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
