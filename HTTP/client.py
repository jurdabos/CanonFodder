"""
Supplies a single helper that performs a GET request with retry and
back-off while adding CanonFodder’s User-Agent per API documentation requirement.
"""
from __future__ import annotations
import requests
from time import sleep
USER_AGENT = "CanonFodder/1.3 (balazs.torda@iu-study.org)"


def make_request(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 8,
):
    """
    Sends a GET request that retries `max_retries` times on 5xx or
    network errors.
    Args:
        url: absolute URL to fetch
        params: optional query-string mapping
        headers: optional header mapping merged into a default
        max_retries: total attempts before giving up
    Returns:
        requests.Response when the call succeeds or after the last
        non-retriable error, otherwise None
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
                attempt = attempt + 1
                sleep(2)
                continue
            # 4xx or other unexpected
            print(f"{url} → {r.status_code}\n{r.text[:300]}")
            return r
        except requests.RequestException as exc:
            print(f"network error {exc}  retry {attempt}/{max_retries}")
            sleep(2)
    return None
