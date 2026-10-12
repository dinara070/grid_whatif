# -*- coding: utf-8 -*-
"""ML-прогноз навантаження на завтра (LightGBM + Open-Meteo) та огляд області по районах. Без Streamlit."""
import io
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from gc_cache import ttl_cache
from gc_model import *  # noqa: F401,F403

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:                                   # noqa: BLE001
    HAS_LGB = False

def _now_local():
    """Поточний час у Києві (naive Timestamp); запасний варіант без tzdata: UTC+3."""
    try:
        return pd.Timestamp(datetime.now(ZoneInfo("Europe/Kyiv")).replace(tzinfo=None))
    except Exception:                                # noqa: BLE001
        return pd.Timestamp((datetime.now(timezone.utc) + timedelta(hours=3)).replace(tzinfo=None))




# ════════════════════════════════════════════════════════════════════════════
# 7b. ML-ПРОГНОЗ НАВАНТАЖЕННЯ НА ЗАВТРА (LightGBM + прогноз погоди Open-Meteo)
# ════════════════════════════════════════════════════════════════════════════
OM_FC_URL = "https://api.open-meteo.com/v1/forecast"
OM_AR_URL = "https://archive-api.open-meteo.com/v1/archive"
FC_FEATS = ["hour", "hour_sin", "hour_cos", "dow", "is_wkend", "temp", "t3", "t24", "cold", "heat"]
TOTAL_BASE_MW = sum(NODES[n]["load"] for n in TPS)       # навантаження мережі при коефіцієнті = 1.0


def _om_get(url, params, timeout=8):
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": "grid-twin-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _om_series(js, key="temperature_2m"):
    h = js["hourly"]
    return pd.Series(h[key], index=pd.to_datetime(h["time"]), dtype=float)


def synth_weather(start, end, seed=3):
    """Синтетична температура (запасний варіант, якщо Open-Meteo недоступний)."""
    idx = pd.date_range(start, end, freq="h")
    seasonal = 7.8 - 12.4 * np.cos(2 * np.pi * (idx.dayofyear.values - 15) / 365.0)   # клімат Вінниччини: січ ≈ -4.6, лип ≈ +20 °C
    daily = 4.0 * np.sin(2 * np.pi * (idx.hour.values - 9) / 24.0)
    rng = np.random.default_rng(seed)
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.985 * noise[i - 1] + rng.normal(0, 0.6)
    return pd.Series(seasonal + daily + noise, index=idx)


@ttl_cache(3600)
def get_weather(lat, lon, day_key):
    """Годинна температура: архів ERA5 (≈14 міс.) + прогноз на 3 доби; сонячна радіація - лише прогноз.
    Повертає (ряд T, джерело, часовий пояс, ряд радіації або None)."""
    today = pd.Timestamp(day_key)
    tz = "Europe/Kyiv"
    try:
        fcj = _om_get(OM_FC_URL, dict(latitude=lat, longitude=lon, hourly="temperature_2m,shortwave_radiation",
                                      past_days=10, forecast_days=3, timezone=tz))
        s_fc = _om_series(fcj)
        rad = _om_series(fcj, "shortwave_radiation") if "shortwave_radiation" in fcj["hourly"] else None
    except Exception as ex:                          # noqa: BLE001
        s = synth_weather(today - pd.Timedelta(days=420), today + pd.Timedelta(days=3))
        return s, f"синтетична погода (Open-Meteo недоступний: {type(ex).__name__})", tz, None
    try:
        arj = _om_get(OM_AR_URL, dict(latitude=lat, longitude=lon, hourly="temperature_2m", timezone=tz,
                                      start_date=(today - pd.Timedelta(days=420)).strftime("%Y-%m-%d"),
                                      end_date=(today - pd.Timedelta(days=11)).strftime("%Y-%m-%d")), timeout=20)
        s_ar, src = _om_series(arj), "Open-Meteo: архів ERA5 + прогноз"
    except Exception:                                # noqa: BLE001
        s_ar = synth_weather(s_fc.index[0] - pd.Timedelta(days=420), s_fc.index[0] - pd.Timedelta(hours=1))
        src = "Open-Meteo: прогноз (історія температур синтетична)"
    s = pd.concat([s_ar, s_fc])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.asfreq("h").interpolate(limit=12).ffill().bfill()
    return s, src, tz, rad


def _true_load(idx, temp):
    """Синтетична «істинна» поведінка: профіль доби × вихідні × холод/спека (для демо-історії)."""
    h, dow = idx.hour.values, idx.dayofweek.values
    base = np.asarray(HOURLY)[h]
    wk = np.where(dow == 5, 0.93, np.where(dow == 6, 0.88, 1.0))
    morning = np.where((dow >= 5) & (h >= 6) & (h <= 10), 0.94, 1.0)
    return (base * wk * morning * (1 + 0.012 * np.maximum(0, 5 - temp))
            * (1 + 0.008 * np.maximum(0, temp - 24)))


def _fc_features(ser):
    d = pd.DataFrame({"temp": ser})
    h = ser.index.hour
    d["hour"] = h
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["dow"] = ser.index.dayofweek
    d["is_wkend"] = (d["dow"] >= 5).astype(int)
    d["t3"] = ser.rolling(3, min_periods=1).mean()
    d["t24"] = ser.rolling(24, min_periods=1).mean()
    d["cold"] = np.maximum(0, 5 - ser)
    d["heat"] = np.maximum(0, ser - 24)
    return d


def _make_reg():
    if HAS_LGB:
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=0,
                                 verbose=-1), "LightGBM"
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0), "HistGradientBoosting (sklearn)"


def _read_hist(raw, hist_idx):
    """CSV (ts, load_mw; крок будь-який) -> коефіцієнт навантаження на сітці годин hist_idx."""
    dfh = pd.read_csv(io.BytesIO(raw))
    cols = {c.lower().strip(): c for c in dfh.columns}
    tcol = next((cols[k] for k in ("ts", "timestamp", "time", "datetime") if k in cols), None)
    lcol = next((cols[k] for k in ("load_mw", "load", "p_mw") if k in cols), None)
    if tcol is None or lcol is None:
        raise ValueError("потрібні стовпці ts та load_mw")
    s = pd.Series(pd.to_numeric(dfh[lcol], errors="coerce").values, index=pd.to_datetime(dfh[tcol]))
    s = s.dropna().sort_index().resample("h").mean() / TOTAL_BASE_MW
    y = s.reindex(hist_idx).dropna()
    if len(y) < 24 * 21:
        raise ValueError("потрібно щонайменше 21 добу даних, що перетинаються з погодою")
    return y


@ttl_cache(3600)
def forecast_load(lat, lon, day_key, hist_bytes=None):
    ser, w_src, tz, rad = get_weather(lat, lon, day_key)
    now_loc = _now_local()
    t_idx = pd.date_range(now_loc.normalize() + pd.Timedelta(days=1), periods=24, freq="h")
    if not set(t_idx).issubset(set(ser.index)):
        raise ValueError("у погодних даних немає всіх 24 годин завтрашньої доби")
    feats = _fc_features(ser)
    hist_idx = ser.index[ser.index <= now_loc.floor("h")]
    y, d_src = None, "синтетична історія навантаження (демо)"
    if hist_bytes:
        try:
            y = _read_hist(hist_bytes, hist_idx)
            d_src = f"ваш CSV ({len(y)} год)"
        except Exception as ex:                      # noqa: BLE001
            d_src = f"синтетична історія (CSV не підійшов: {ex})"
    if y is None:
        rng = np.random.default_rng(5)
        noise = np.zeros(len(hist_idx))
        for i in range(1, len(noise)):
            noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.012)
        y = pd.Series(_true_load(hist_idx, ser.loc[hist_idx].values) * (1 + noise), index=hist_idx)
    X = feats.loc[y.index, FC_FEATS]
    cut = y.index[-1] - pd.Timedelta(days=14)
    tr, te = y.index <= cut, y.index > cut
    m_cv, name = _make_reg()
    m_cv.fit(X[tr], y[tr])
    pred_te = m_cv.predict(X[te])
    yt = y[te].values
    stat_te = (np.asarray(HOURLY)[X.loc[te, "hour"].astype(int).values]
               * (1 + 0.012 * np.maximum(0, 5 - X.loc[te, "temp"].values)))
    m, _ = _make_reg()
    m.fit(X, y)
    Xt = feats.loc[t_idx, FC_FEATS]
    prof = np.clip(m.predict(Xt), 0.2, 2.5)
    stat = np.asarray(HOURLY) * (1 + 0.012 * np.maximum(0, 5 - Xt["temp"].values))
    k = min(96, len(yt))
    if rad is not None and set(t_idx).issubset(set(rad.index)):
        pv_f = np.clip(rad.reindex(t_idx).fillna(0).values / 1000.0 * 0.85, 0, 1)     # PR ≈ 0.85
        pv_src = "Open-Meteo: сонячна радіація (shortwave_radiation)"
    else:
        pv_f = np.array([pv_profile(h) for h in range(24)])
        pv_src = "типова зимова крива (прогноз радіації недоступний)"
    return dict(pv_frac=[float(v) for v in pv_f], pv_src=pv_src,
                
        date=t_idx[0].strftime("%d.%m.%Y"), tz=tz, weather_src=w_src, data_src=d_src, model=name, n_train=int(len(y)),
        mae_ml=float(np.mean(np.abs(pred_te - yt))), mae_static=float(np.mean(np.abs(stat_te - yt))),
        mape_ml=float(100 * np.mean(np.abs(pred_te - yt) / np.maximum(yt, 1e-6))),
        profile=[float(v) for v in prof], static_profile=[float(v) for v in stat],
        temps=[float(v) for v in Xt["temp"].values],
        bt=dict(ts=[t.strftime("%Y-%m-%d %H:%M") for t in y[te].index[-k:]], actual=[float(v) for v in yt[-k:]],
                ml=[float(v) for v in pred_te[-k:]], static=[float(v) for v in stat_te[-k:]]))


@ttl_cache(3600)
def oblast_weather(day_key):
    """Умови на завтра по районах: T мін/макс, пік навантаження типової мережі, потенціал PV."""
    names = list(DISTRICTS)
    t_idx = pd.date_range(_now_local().normalize() + pd.Timedelta(days=1), periods=24, freq="h")
    series = []
    try:
        js = _om_get(OM_FC_URL, dict(latitude=",".join(str(DISTRICTS[n]["lat"]) for n in names),
                                     longitude=",".join(str(DISTRICTS[n]["lon"]) for n in names),
                                     hourly="temperature_2m,shortwave_radiation", forecast_days=3,
                                     timezone="Europe/Kyiv"), timeout=15)
        js = [js] if isinstance(js, dict) else js
        if len(js) != len(names):
            raise ValueError("несподівана відповідь API")
        for j in js:
            series.append((_om_series(j).reindex(t_idx).values,
                           _om_series(j, "shortwave_radiation").reindex(t_idx).fillna(0).values))
        src = "Open-Meteo"
    except Exception as ex:                          # noqa: BLE001
        series = []
        base = synth_weather(t_idx[0] - pd.Timedelta(days=2), t_idx[-1] + pd.Timedelta(days=1)).reindex(t_idx).values
        for i, n in enumerate(names):
            off = (49.0 - DISTRICTS[n]["lat"]) * 0.6 + 0.2 * (i % 3)
            series.append((base + off, np.array([pv_profile(h) / 0.7 * 400.0 for h in range(24)])))
        src = f"синтетичні дані (Open-Meteo недоступний: {type(ex).__name__})"
    rows = []
    for n, (T, R) in zip(names, series):
        f = np.asarray(HOURLY) * (1 + 0.012 * np.maximum(0, 5 - T))
        rows.append(dict(district=n, lat=DISTRICTS[n]["lat"], lon=DISTRICTS[n]["lon"], tmin=float(np.nanmin(T)),
                         tmax=float(np.nanmax(T)), peak_factor=float(f.max()), peak_mw=float(f.max() * TOTAL_BASE_MW),
                         pv_yield=float(np.nansum(R) / 1000.0 * 0.85)))
    return pd.DataFrame(rows), src, t_idx[0].strftime("%d.%m.%Y")
