"""
csv.autofetch
=============

Automates the download of a last.fm scrobble history through
https://benjaminbenben.com/lastfm-to-csv/ and stores the resulting file
in *out_dir*.  A lock-file throttles execution so the scraper runs at
most once per chosen period.
"""
from __future__ import annotations
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.microsoft import EdgeChromiumDriverManager
# os.environ["WDM_SSL_VERIFY"] = "0"
LOGGER = logging.getLogger(__name__)
DEFAULT_URL = "https://benjaminbenben.com/lastfm-to-csv/"
DEFAULT_WAIT = 1_700  # seconds to let the site crunch
LOCK_NAME = "last_run.lock"


def fetch_scrobbles_csv(
        username: str,
        out_dir: str | os.PathLike = "CSV",
        once_per: str = "week",
        headless: bool = True
) -> Path | None:
    """
    Downloads a fresh scrobbles CSV for *username*
    ia benjaminbenben.com/lastfm-to-csv/ and throttles repeated runs with a lock file.
    Args:
        username: last.fm user name.
        out_dir: target directory that will contain the CSV file
        once_per: throttling period 'day' or 'week'
        headless: Launches Microsoft Edge in headless mode when true.
    Returns:
        pathlib.Path to the saved CSV file or None if the run is throttled or any error occurs.
    Raises:
        RuntimeError: lock-file exists but is unreadable
        selenium.common.TimeoutException: page elements do not appear within the expected time
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------ lock
    lock = out_dir / LOCK_NAME
    if lock.exists():
        last = datetime.strptime(lock.read_text().strip(), "%Y-%m-%d")
        now = datetime.now()
        same_period = (
            last.date() == now.date() if once_per == "day"
            else last.isocalendar()[1] == now.isocalendar()[1]
        )
        if same_period:
            LOGGER.info("Fetcher already ran this %s; skipping", once_per)
            return None
    # ---------------------------------------------------------------- browser
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    with webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=opts) as driver:
        try:
            LOGGER.info("Loading converter page …")
            driver.get(DEFAULT_URL)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "lastfm-user"))
            ).send_keys(username + Keys.RETURN)
            LOGGER.info("Waiting %s s for server‑side build", f"{DEFAULT_WAIT:,} s")
            time.sleep(DEFAULT_WAIT)
            # Polling <a class="btn-success">Save 1234 KB</a> until size stabilises
            stable, last_size = 0, None
            LOCATOR = (By.XPATH,
                       "//a[@href='#download' and contains(@class,'btn')]")
            while stable < 3:
                try:
                    btn = WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located(LOCATOR)
                    )
                except TimeoutException:
                    # button not visible **yet** → wait and retry
                    time.sleep(10)
                    continue
                try:
                    size_kb = int(
                        btn.find_element(By.TAG_NAME, "small").text.rstrip("KB").strip()
                    )
                except (IndexError, ValueError):
                    size_kb = None
                if size_kb and size_kb == last_size:
                    stable += 1
                else:
                    stable, last_size = 0, size_kb
                time.sleep(5)
            if stable < 3:
                LOGGER.error("Download button never stabilised – aborting")
                return None
            # else we have a valid btn
            LOGGER.info("File size stabilised at %s KB → downloading", last_size)
            time.sleep(10)  # let browser finish writing

            dl_dir = Path.home() / "Downloads"
            tmp = dl_dir / f"{username}.csv"
            if not tmp.exists():
                LOGGER.error("Downloaded file not found: %s", tmp)
                return None
            final = out_dir / f"{username}_{datetime.now():%Y%m%d}.csv"
            tmp.replace(final)
            # touch lock
            lock.write_text(datetime.now().strftime("%Y-%m-%d"))
            LOGGER.info("Saved → %s", final)
            return final
        except Exception as exc:
            LOGGER.exception("CSV fetch failed: %s", exc)
            return None
