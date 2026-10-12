import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # модулі проєкту шукаємо поруч з app.py

try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    import gc_service as service
    from gc_model import (
        ALL_ELEMS,
        DISTRICTS,
        EDGES,
        OBLAST_CENTER,
        Params,
        SOURCES,
        TPS,
        fmt_elem,
        load_scale,
        make_params,
        pv_frac)
    from gc_powerflow import (daily_profile, health_check, scan_all, sens_grid, solve_c)
    from gc_protection import (
        K_N,
        K_V,
        K_ZAP,
        SWITCH_BREAK_KA,
        T_IN,
        T_MSZ,
        fault_report,
        protection_settings)
    from gc_viz import (
        PHASES,
        STATUS_STYLE,
        _map_cls,
        close_fig,
        draw_geo_map,
        draw_map,
        fig_summary,
        fig_waterfall,
        voltage_profile)
    from gc_predictive import (
        CAT_COLOR,
        FEATURES,
        FEAT_UA,
        HAS_SHAP,
        UA_NAMES,
        explain,
        get_model,
        get_shap,
        make_training,
        registry_features,
        risk_register,
        score_registry,
        shap_row,
        training_from_csv)
    from gc_forecast import (TOTAL_BASE_MW, forecast_load, oblast_weather)
    from gc_telemetry import (
        HAS_MQTT,
        classify_state,
        current_store,
        get_store,
        stop_store)
except Exception as _err:                           # noqa: BLE001  (на Streamlit Cloud текст винятків приховується - показуємо самі)
    st.set_page_config(page_title="Помилка завантаження", page_icon="⚠️", layout="wide")
    st.error(f"**Не вдалося завантажити модулі застосунку:** `{type(_err).__name__}: {_err}`")
    st.markdown(
        "**Що перевірити**\n\n"
        "1. У GitHub-репозиторії **поруч з `app.py`** мають лежати всі файли проєкту (папка `gridcore/` з усіма "
        "`.py`-файлами або, у плоскому варіанті, файли `gc_*.py`) та `telemetry_generator.py`.\n"
        "2. У `requirements.txt` мають бути всі залежності з архіву (`fpdf2`, `lightgbm`, `shap`, `paho-mqtt`, `pandas<3` …).\n"
        "3. Після завантаження файлів натисніть **Manage app → Reboot app**.\n\n"
        "Нижче - що зараз лежить поруч з `app.py` на сервері:")
    st.code("\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in HERE.iterdir())), language="text")
    st.stop()

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
