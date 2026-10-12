> **Плоский варіант:** усі файли лежать в одному рівні (без папок): модулі ядра - `gc_*.py` (відповідають `gridcore/*.py` з архіву з папкою). Просто завантажте ВСІ файли в корінь репозиторію.

# Цифровий двійник мережі 10 кВ · Вінницька область

Умовна (синтетична) розподільча мережа 2 ПС 110/10 кВ · 4 фідери · 20 ТП, розміщена в обраному районі
Вінницької області. **Це не реальні об'єкти й не дані обленерго.**

## Архітектура

```
app.py  (Streamlit, лише візуальна частина)      api.py  (FastAPI, HTTP/JSON)
        \                                          /
         └──────────── gridcore.service ──────────┘      <- фасад: параметри, зведення, ризики, PDF
                              │
   ┌────────────┬─────────────┼──────────────┬──────────────┬───────────────┐
 model        powerflow     protection     forecast       predictive      telemetry / simulator
 мережа,      pandapower,   КЗ, уставки    LightGBM +     ML відмов,      LiveStore (MQTT /
 профілі,     відновлення,  РЗА            Open-Meteo     SHAP, ризик     вбудований), симулятор
 Params       N-1/N-k                                                     
                              viz (plotly/matplotlib)      report (PDF: fpdf2 + matplotlib)
```

* `gridcore/` **не імпортує Streamlit** (перевіряється тестами) - ядро можна викликати зі скриптів, ноутбуків,
  планувальників, API.
* Єдине джерело правди про мережу - `gridcore.model.build_grid()`; симулятор телеметрії бере склад активів звідти ж.
* Кеші ядра - `functools.lru_cache` / `gridcore.cache.ttl_cache` (на процес). API без стану між запитами.

## Запуск

```bash
pip install -r requirements.txt
streamlit run app.py                         # інтерфейс

pip install -r requirements-api.txt
uvicorn api:app --port 8000                  # API, документація: http://localhost:8000/docs
GRID_API_KEY=secret uvicorn api:app ...      # увімкнути перевірку заголовка X-API-Key

python telemetry_generator.py --host broker.hivemq.com --prefix vinnytsia-twin-<унікальний_суфікс>   # MQTT-потік
pytest -q                                    # pip install -r requirements-dev.txt
```

Streamlit Cloud: основний файл `app.py`, у корені репозиторію мають лежати `gridcore/`, `telemetry_generator.py`,
`requirements.txt`. Версія `pandas<3` обов'язкова для pandapower < 3.2.

## PDF-звіт

У вкладці «Сценарний симулятор» → «PDF-звіт для диспетчера / керівника»: умови, показники, схема мережі, протокол
перемикань, профіль напруги, завантаження ліній, РЗА після перемикань і (опційно) матриця ризиків.
Через API: `POST /report/pdf`.

## API

| Метод | Шлях | Призначення |
|---|---|---|
| GET | `/health`, `/districts`, `/grid` | службове, райони, склад мережі |
| POST | `/scenario` | аварія N-k → протокол перемикань, режим ліній, напруги |
| POST | `/n1` | N-1 для всіх елементів |
| POST | `/protection` | струми КЗ та перевірка РЗА |
| GET | `/forecast?district=` | ML-прогноз навантаження та PV на завтра |
| POST | `/risk` | ризик = P(відмови) × наслідки |
| POST | `/report/pdf` | PDF-звіт |

Приклад: `curl -X POST localhost:8000/scenario -H 'Content-Type: application/json' -d '{"elements":["ТП-103"]}'`

## Обмеження

* Історія навантаження й відмов для ML синтетична, доки не підключено дані SCADA/CMMS (CSV через інтерфейс).
* Мережа однакова для всіх районів; відрізняються розташування, погода й сонячна радіація.
* Звіт і розрахунки демонстраційні й не замінюють погоджених бланків перемикань.
