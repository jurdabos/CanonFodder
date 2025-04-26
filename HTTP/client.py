"""
Light-weight HTTP helpers that are agnostic of CanonFodder’s domain logic.
"""
from __future__ import annotations
import os
import requests
from time import sleep
USER_AGENT = "CanonFodder/1.0 (balazs.torda@iu-study.org)"


def make_request(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 8,
):
    """
    Generic GET with retry/back-off.
    """
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers)
            if r.ok:
                return r
            if 500 <= r.status_code < 600:
                print(f"{url} → {r.status_code}  retry {attempt}/{max_retries}")
                sleep(2)
                continue
            # 4xx or other unexpected
            print(f"{url} → {r.status_code}\n{r.text[:300]}")
            return r
        except requests.RequestException as exc:
            print(f"network error {exc}  retry {attempt}/{max_retries}")
            sleep(2)
    return None
