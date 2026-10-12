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
import time

from gc_simulator import Simulator     # той самий симулятор, що й у застосунку (одне джерело істини)


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
