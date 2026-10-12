# -*- coding: utf-8 -*-
"""Симулятор телеметрії трансформаторів (2 ПС + 20 ТП): струм, температура масла/обмотки, напруга.

Легкий модуль (numpy/stdlib): його використовують і вбудований генератор застосунку, і telemetry_generator.py.
Склад активів і навантаження беруться з model.build_grid() - одне джерело істини з електричною моделлю.
"""
import math
import random
import time

from gc_model import HOURLY, build_grid

FEEDER_SRC = {"Ф-1": "ПС-1", "Ф-2": "ПС-1", "Ф-3": "ПС-2", "Ф-4": "ПС-2"}


def build_assets():
    nodes, _ = build_grid()
    assets = {n: dict(kind="ps", kva=12600, load=0.0, feeder_of=n) for n, d in nodes.items() if d["type"] == "src"}
    for n, d in nodes.items():
        if d["type"] == "tp":
            assets[n] = dict(kind="tp", kva=d["kva"], load=d["load"], feeder_of=FEEDER_SRC[d["feeder"]])
    for s in [a for a, d in assets.items() if d["kind"] == "ps"]:
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
