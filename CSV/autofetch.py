from __future__ import annotations
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from selenium import webdriver
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
    Download a fresh scrobbles CSV for *username* via the Benjamin Foxall site.
    Returns the Path to the saved file, or None on failure / throttled run.
    *once_per* may be "day" or "week" -> lock‑file prevents rerun in same period.
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

    driver = webdriver.Edge(
        service=Service(EdgeChromiumDriverManager().install()),
        options=opts
    )
    try:
        LOGGER.info("Loading converter page …")
        driver.get(DEFAULT_URL)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "lastfm-user"))
        ).send_keys(username + Keys.RETURN)
        LOGGER.info("Waiting %s s for server‑side build", DEFAULT_WAIT)
        time.sleep(DEFAULT_WAIT)
        # Polling <a class="btn-success">Save 1234 KB</a> until size stabilises
        stable, last_size = 0, None
        while stable < 3:
            btn = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//a[contains(@class,'btn-success') and contains(text(),'Save')]")
                )
            )
            try:
                size_kb = int(btn.text.split()[1].rstrip("KB"))
            except (IndexError, ValueError):
                size_kb = None
            if size_kb and size_kb == last_size:
                stable += 1
            else:
                stable, last_size = 0, size_kb
            time.sleep(5)
        LOGGER.info("File size stabilised at %s KB → downloading", last_size)
        btn.click()
        time.sleep(10)  # let the browser finish writing

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
    finally:
        driver.quit()
