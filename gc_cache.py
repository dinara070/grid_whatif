# -*- coding: utf-8 -*-
"""Прості кеші без Streamlit: TTL-кеш для мережевих/ML-викликів та ключ для DataFrame."""
import functools
import threading
import time

import pandas as pd


def ttl_cache(ttl, maxsize=32):
    """Кеш із часом життя ttl (с). Аргументи мають бути хешованими; винятки не кешуються."""
    def deco(fn):
        store, lock = {}, threading.Lock()

        @functools.wraps(fn)
        def wrapper(*a, **k):
            key = (a, tuple(sorted(k.items())))
            now = time.time()
            with lock:
                hit = store.get(key)
                if hit is not None and now - hit[0] < ttl:
                    return hit[1]
            val = fn(*a, **k)
            with lock:
                store[key] = (now, val)
                while len(store) > maxsize:
                    store.pop(next(iter(store)))
            return val
        wrapper.cache_clear = store.clear
        return wrapper
    return deco


def df_key(df):
    """Хешований ключ вмісту DataFrame (для кешів моделей)."""
    return (df.shape, int(pd.util.hash_pandas_object(df, index=True).sum()) % (2 ** 63))
