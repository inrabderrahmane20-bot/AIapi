import os
import redis

REDIS_URL = os.environ.get("REDIS_URL")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL is not set")

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
)
