import json
import threading
import time
from functools import wraps
from flask import jsonify
from redis_client import redis_client
from cache_key import make_cache_key

CACHE_TTL = 60 * 60 * 6          # 6 hours
STALE_TTL = 60 * 60 * 24         # 24 hours

def cached_endpoint(prefix="api"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = make_cache_key(prefix)

            cached_raw = redis_client.get(cache_key)
            if cached_raw:
                cached = json.loads(cached_raw)

                # STALE-WHILE-REVALIDATE
                if cached["expires_at"] < time.time():
                    threading.Thread(
                        target=_refresh_cache,
                        args=(func, cache_key, args, kwargs),
                        daemon=True
                    ).start()

                return jsonify(cached["data"])

            # NO CACHE → run original code (UNCHANGED)
            response = func(*args, **kwargs)
            data = response.get_json()

            _store(cache_key, data)
            return response

        return wrapper
    return decorator


def _store(key, data):
    payload = {
        "data": data,
        "expires_at": time.time() + CACHE_TTL
    }
    redis_client.setex(
        key,
        STALE_TTL,
        json.dumps(payload)
    )


def _refresh_cache(func, key, args, kwargs):
    try:
        response = func(*args, **kwargs)
        data = response.get_json()
        _store(key, data)
    except Exception:
        pass
