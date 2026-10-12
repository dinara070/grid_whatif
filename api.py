# -*- coding: utf-8 -*-
"""FastAPI-бекенд цифрового двійника мережі 10 кВ (Вінницька область).

Той самий обчислювальний шар (gridcore.service), що й у Streamlit-застосунку, але через HTTP/JSON:
інтегрується з SCADA/ГІС/іншими системами без UI.

Запуск:   uvicorn api:app --host 0.0.0.0 --port 8000      (документація: http://localhost:8000/docs)
Захист:   якщо задано змінну середовища GRID_API_KEY, потрібен заголовок X-API-Key.
"""
import os
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from gridcore import service
from gridcore.model import ALL_ELEMS, DISTRICTS


def _auth(x_api_key: Optional[str] = Header(default=None)):
    key = os.environ.get("GRID_API_KEY")
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="Невірний або відсутній X-API-Key")


app = FastAPI(title="Grid Twin API - Вінницька область", version="3.0",
              description="Сценарії N-1/N-k, режими pandapower, РЗА, прогноз навантаження, ризики, PDF-звіти. "
                          "Мережа УМОВНА (синтетична).",
              dependencies=[Depends(_auth)])


class ParamsIn(BaseModel):
    """Параметри режиму (одиниці як у бічній панелі застосунку: відсотки - у відсотках)."""
    district: str = "Вінниця"
    temp: float = Field(-10.0, ge=-40, le=45, description="°C; з use_forecast=true береться з прогнозу")
    hour: int = Field(18, ge=0, le=23)
    sens: float = Field(1.2, ge=0, le=5, description="%/°C")
    src_cap: float = Field(12.0, gt=0, le=100, description="допустима потужність кожної ПС, МВт")
    tol_pct: float = Field(0.0, ge=0, le=50)
    t_detect: float = Field(5.0, ge=0)
    t_switch: float = Field(4.0, ge=0)
    repair_h: float = Field(8.0, gt=0)
    vmin: float = Field(0.90, ge=0.5, le=1.0)
    vmax: float = Field(1.10, ge=1.0, le=1.5)
    u_ps: float = Field(10.5, ge=9.5, le=11.5)
    cosphi: float = Field(0.95, ge=0.5, le=1.0)
    der: bool = True
    pv_k: float = Field(1.0, ge=0, le=5)
    ins_pct: float = Field(80.0, ge=0, le=100)
    bess_mode: Literal["reserve", "peak_shaving"] = "reserve"
    soc_pct: float = Field(80.0, ge=0, le=100)
    island: bool = True
    use_forecast: bool = Field(False, description="ML-профіль навантаження та PV на завтра (Open-Meteo)")

    def resolve(self):
        try:
            P, fc, err = service.resolve_params(
                district=self.district, use_forecast=self.use_forecast, temp=self.temp, hour=self.hour,
                sens=self.sens, src_cap=self.src_cap, tol=self.tol_pct / 100, t_detect=self.t_detect,
                t_switch=self.t_switch, repair_h=self.repair_h, vmin=self.vmin, vmax=self.vmax, u_ps=self.u_ps,
                cosphi=self.cosphi, der=self.der, pv_k=self.pv_k, ins=self.ins_pct,
                bess_mode=0 if self.bess_mode == "reserve" else 1, soc=self.soc_pct / 100, island=self.island)
        except ValueError as ex:
            raise HTTPException(status_code=422, detail=str(ex))
        return P, fc, err


class ScenarioIn(BaseModel):
    params: ParamsIn = ParamsIn()
    elements: List[str] = Field(..., examples=[["ТП-103"]], description="Елементи, що виходять з ладу (N-k)")


class RiskIn(BaseModel):
    params: ParamsIn = ParamsIn()
    horizon_months: int = Field(12, ge=1, le=24)
    growth_pct: float = Field(0.0, ge=0, le=40)
    winter_days: int = Field(60, ge=0, le=150)
    top: int = Field(15, ge=1, le=100)


class ReportIn(ScenarioIn):
    include_risk: bool = True
    horizon_months: int = Field(12, ge=1, le=24)


def _elements(els):
    try:
        return service.validate_elements(els)
    except ValueError as ex:
        raise HTTPException(status_code=422, detail=str(ex))


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok", "elements": len(ALL_ELEMS), "districts": len(DISTRICTS)}


@app.get("/districts", tags=["grid"])
def districts():
    return [dict(name=n, **c) for n, c in DISTRICTS.items()]


@app.get("/grid", tags=["grid"])
def grid():
    """Склад умовної мережі: вузли (ПС/ТП) та лінії."""
    return service.grid_description()


@app.post("/scenario", tags=["calc"])
def scenario(body: ScenarioIn):
    """Аварія -> ізоляція -> відновлення (перемички/острів) -> ГОП; підсумки, протокол перемикань, режим ліній."""
    P, fc, err = body.params.resolve()
    out = service.summarize_scenario(service.solve_c(P, _elements(body.elements)))
    out["profile"] = service.profile_label(P, fc)
    if err:
        out["forecast_warning"] = err
    return out


@app.post("/n1", tags=["calc"])
def n1(body: ParamsIn):
    """N-1 для всіх елементів мережі (довго: ≈10–40 с першого разу, далі з кешу)."""
    P, _, _ = body.resolve()
    df, _ = service.scan_all(P)
    return service.df_records(df.sort_values("ens_mwh", ascending=False))


@app.post("/protection", tags=["calc"])
def protection(body: ScenarioIn):
    """Струми КЗ у місці пошкодження та перевірка уставок РЗА після перемикань."""
    P, _, _ = body.params.resolve()
    faults, post_df, in_df, isl = service.fault_report(P, _elements(body.elements))
    return dict(faults=service.df_records(faults.drop(columns="lvl")), feeders_after=service.df_records(post_df.drop(columns="lvl")),
                inputs=service.df_records(in_df), islands=isl)


@app.get("/forecast", tags=["ml"])
def forecast(district: str = Query("Вінниця")):
    """ML-прогноз навантаження та PV на завтра (LightGBM + Open-Meteo) для району."""
    if district not in DISTRICTS:
        raise HTTPException(status_code=422, detail=f"невідомий район: {district}")
    try:
        fc = service.forecast_load(DISTRICTS[district]["lat"], DISTRICTS[district]["lon"], service.day_key())
    except Exception as ex:                          # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"прогноз недоступний: {type(ex).__name__}: {ex}")
    return {k: v for k, v in fc.items() if k != "bt"}


@app.post("/risk", tags=["ml"])
def risk(body: RiskIn):
    """Ризик = P(відмови, ML) × наслідки (N-1): топ активів за очікуваним недовідпуском."""
    P, _, _ = body.params.resolve()
    mg, _ = service.risk_table(P, body.horizon_months, body.growth_pct / 100, body.winter_days)
    cols = ["id", "type", "p_h", "ens_mwh", "unsup_mw", "cust", "exp_ens", "exp_saidi", "cat", "reco"]
    return service.df_records(mg[cols].head(body.top))


@app.post("/report/pdf", tags=["report"], response_class=Response)
def report_pdf(body: ReportIn):
    """PDF-звіт для диспетчера/керівника (application/pdf)."""
    P, fc, _ = body.params.resolve()
    pdf = service.build_pdf(P, _elements(body.elements), body.params.district, body.include_risk,
                            body.horizon_months, fc=fc)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="grid_twin_report.pdf"'})
