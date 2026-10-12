# -*- coding: utf-8 -*-
"""Прогноз відмов обладнання (ML), реєстр активів, SHAP-пояснення, ризик = P × наслідки. Без Streamlit."""
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from gc_cache import df_key
from gc_model import *  # noqa: F401,F403
from gc_powerflow import new_net, pp_state, topology  # noqa: F401

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
