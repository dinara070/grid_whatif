# -*- coding: utf-8 -*-
"""
Генератор синтетичної телеметрії для цифрового двійника мережі 10 кВ (Вінницька область).

Імітує RTU/SCADA: кожні --interval секунд публікує в MQTT JSON-повідомлення по кожному
трансформатору (2 ПС + 20 ТП): струм, завантаження, температуру масла та обмотки, напругу.
Час у генераторі «віртуальний» (прискорений): кожен тік = --vstep хвилин, тому добовий цикл
навантаження видно за кілька хвилин.

Топіки:   <prefix>/telemetry/<ID активу>      напр.  vinnytsia-twin/telemetry/ТП-103
          <prefix>/control                    команди {"cmd":"inject","asset":"ТП-103","kind":"overload|cooling","ticks":12} / {"cmd":"clear"}
Запуск:
    pip install paho-mqtt numpy
    python telemetry_generator.py --host broker.hivemq.com --prefix vinnytsia-twin-<ваш_унікальний_суфікс>
    python telemetry_generator.py --dry-run          # без брокера, лише друк у консоль

УВАГА: публічні брокери (hivemq, mosquitto.org) бачать усі; задайте унікальний --prefix.
У застосунку (вкладка «Телеметрія») оберіть «MQTT-брокер» з тими самими host/port/prefix.
"""
import argparse
import json
import math
import random
import time

import numpy as np

HOURLY = [0.52, 0.48, 0.46, 0.45, 0.46, 0.52, 0.65, 0.80, 0.85, 0.80, 0.76, 0.75,
          0.74, 0.73, 0.74, 0.78, 0.90, 0.98, 1.00, 0.97, 0.92, 0.85, 0.72, 0.60]


def build_assets():
    """Копія логіки build_grid() з app.py (той самий seed => ті самі ТП, навантаження, кВА)."""
    rng = np.random.default_rng(42)
    std_kva = [250, 400, 630, 1000, 1250, 1600]
    assets = {"ПС-1": dict(kind="ps", kva=12600, load=0.0, feeder_of="ПС-1"),
              "ПС-2": dict(kind="ps", kva=12600, load=0.0, feeder_of="ПС-2")}
    feeders = {"Ф-1": ("ПС-1", 101), "Ф-2": ("ПС-1", 201), "Ф-3": ("ПС-2", 301), "Ф-4": ("ПС-2", 401)}
    for fname, (src, base) in feeders.items():
        for i in range(5):
            name = f"ТП-{base + i}"
            load = round(float(rng.uniform(0.35, 0.85)), 2)
            cands = [k for k in std_kva if 0.35 <= load / (0.95 * k / 1000) <= 0.85]
            kva = int(rng.choice(cands)) if cands else 1000
            rng.integers(90, 450)                       # cons (щоб зберегти послідовність rng)
            rng.choice([1, 2, 3], p=[0.15, 0.55, 0.30])  # prio
            rng.uniform(0.7, 1.9)                       # довжина сегмента
            assets[name] = dict(kind="tp", kva=kva, load=load, feeder_of=src)
    # перемички в build_grid() додаються після і не впливають на ТП
    for s in ("ПС-1", "ПС-2"):
        assets[s]["load"] = sum(a["load"] for a in assets.values() if a["kind"] == "tp" and a["feeder_of"] == s)
    return assets


class Simulator:
    def __init__(self, ambient=-5.0, sens=1.2, seed=None, start_hour=6.0, vstep=15.0):
        self.assets = build_assets()
        self.rng = random.Random(seed)
        self.ambient, self.sens = ambient, sens
        self.vt = start_hour * 60.0
        self.vstep = vstep
        self.oil = {a: ambient + 15.0 for a in self.assets}
        self.events = {}          # asset -> (kind, ticks_left)

    def rated_a(self, a):
        d = self.assets[a]
        return d["kva"] / (math.sqrt(3) * (10.5 if d["kind"] == "ps" else 10.0))

    def tick(self):
        self.vt += self.vstep
        h = (self.vt / 60.0) % 24.0
        amb = self.ambient + 4.0 * math.sin(2 * math.pi * (h - 9.0) / 24.0)
        tf = 1.0 + self.sens / 100.0 * max(0.0, 5.0 - amb)
        scale = HOURLY[int(h)] * tf
        # випадкові події
        tps = [a for a, d in self.assets.items() if d["kind"] == "tp"]
        if self.rng.random() < 0.04:
            self.events[self.rng.choice(tps)] = ("overload", 8)
        if self.rng.random() < 0.02:
            self.events[self.rng.choice(tps)] = ("cooling", 20)
        msgs = []
        for a, d in self.assets.items():
            cap_mw = d["kva"] * 0.95 / 1000.0
            loading = d["load"] * scale / cap_mw
            kind, left = self.events.get(a, (None, 0))
            if kind == "overload":
                loading *= 1.5
            loading = max(0.0, loading * (1 + self.rng.gauss(0, 0.03)))
            gain = 1.35 if kind == "cooling" else 1.0
            target = amb + 52.0 * gain * loading ** 1.8
            self.oil[a] += 0.15 * (target - self.oil[a])
            oil = self.oil[a] + self.rng.gauss(0, 0.4)
            wind = oil + 23.0 * loading ** 2 + self.rng.gauss(0, 0.5)
            u = 10.5 - 0.35 * min(loading, 1.3) + self.rng.gauss(0, 0.03)
            i_a = loading * self.rated_a(a)
            alarm = "OVERLOAD" if loading > 1.0 else ("OIL_HIGH" if oil > 95 else "")
            msgs.append(dict(asset=a, kind=d["kind"], ts=time.time(), vt=f"{int(h):02d}:{int((h % 1) * 60):02d}",
                             hour=round(h, 2), i_a=round(i_a, 1), i_rated_a=round(self.rated_a(a), 1),
                             loading=round(loading, 4), oil_c=round(oil, 1), winding_c=round(wind, 1),
                             u_kv=round(u, 3), p_mw=round(loading * cap_mw, 3), ambient_c=round(amb, 1),
                             alarm=alarm))
            if left > 0:
                self.events[a] = (kind, left - 1) if left > 1 else (None, 0)
        return msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--prefix", default="vinnytsia-twin")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--interval", type=float, default=2.0, help="пауза між тіками, с")
    ap.add_argument("--vstep", type=float, default=15.0, help="віртуальних хвилин за тік")
    ap.add_argument("--ambient", type=float, default=-5.0, help="середня температура повітря, °C")
    ap.add_argument("--sens", type=float, default=1.2, help="чутливість навантаження, %%/°C")
    ap.add_argument("--start-hour", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=0, help="0 = нескінченно")
    ap.add_argument("--dry-run", action="store_true", help="не підключатись до брокера, друкувати JSON")
    a = ap.parse_args()

    sim = Simulator(a.ambient, a.sens, a.seed, a.start_hour, a.vstep)

    client = None
    if not a.dry_run:
        import paho.mqtt.client as mqtt
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:                       # paho-mqtt 1.x
            client = mqtt.Client()
        if a.user:
            client.username_pw_set(a.user, a.password)
        ctrl_topic = f"{a.prefix}/control"

        def on_connect(cl, ud, flags, rc, *args):
            cl.subscribe(ctrl_topic)                 # команди з застосунку: інжекція/скидання аномалій

        def on_message(cl, ud, msg):
            try:
                c = json.loads(msg.payload.decode("utf-8"))
                if c.get("cmd") == "clear":
                    sim.events.clear()
                    print("команда: скинути аномалії")
                elif c.get("cmd") == "inject" and c.get("asset") in sim.assets:
                    sim.events[c["asset"]] = (c.get("kind", "overload"), int(c.get("ticks", 12)))
                    print(f"команда: аномалія {c.get('kind')} на {c['asset']}")
            except Exception:                        # noqa: BLE001
                pass
        client.on_connect, client.on_message = on_connect, on_message
        client.reconnect_delay_set(1, 10)
        client.connect(a.host, a.port, 30)
        client.loop_start()
        print(f"Підключено до {a.host}:{a.port}, топік {a.prefix}/telemetry/<актив>, керування {ctrl_topic}")

    n = 0
    try:
        while a.ticks == 0 or n < a.ticks:
            msgs = sim.tick()
            for m in msgs:
                payload = json.dumps(m, ensure_ascii=False)
                if client is None:
                    print(payload)
                else:
                    client.publish(f"{a.prefix}/telemetry/{m['asset']}", payload, qos=0)
            n += 1
            if client is not None:
                alarms = [m["asset"] + ":" + m["alarm"] for m in msgs if m["alarm"]]
                print(f"тік {n}: віртуальний час {msgs[0]['vt']}, повідомлень {len(msgs)}"
                      + (f", тривоги: {', '.join(alarms)}" if alarms else ""))
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()
