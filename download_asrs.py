#!/usr/bin/env python3
"""Download ASRS Database Online reports as CSV, in 6-month batches.

Drives the public ASRS Query Wizard
(https://dbol1-prod.arc.nasa.gov/ASRSPublicQueryWizard/) over plain HTTP, the
same way a browser does: for each window it starts a new search, filters on
"Date of Incident", runs the search, reads how many reports matched and
exports the result as CSV.

- Windows are calendar half-years (Jan-Jun, Jul-Dec), clipped to --start/--end.
- A window with more than --limit reports is split in half (6 -> 3 -> 2/1
  months) until every part fits. A single month that is still too large is
  reported and skipped (the wizard can't filter on days).
- Files are saved as <out>/<year>-<first month>-<last month>.csv
  (e.g. data/2023-01-06.csv), in the wizard's own CSV format, so
  merge_csv.py can combine them with the existing exports.
- Files that already exist are skipped, so an interrupted run can be restarted.
- Each export is checked: it must have the ASRS header and as many rows as
  the search reported.

Usage:
    python download_asrs.py                          # 1988-01 .. 2023-12 -> data/
    python download_asrs.py --start 2015-01 --end 2023-12
    python download_asrs.py --dry-run                # only print report counts
"""

import argparse
import csv
import io
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://dbol1-prod.arc.nasa.gov/ASRSPublicQueryWizard/"
FILTER_PAGE = "QueryWizard_Filter.aspx"
NEW_QUERY_PAGE = "QueryWizard_StartNewQuery.aspx"
EXPORT_PAGE = "QueryWizard_ExportExcel.aspx?ExportType=CSV"
# Id of the "Date of Incident was between ..." statement on the filter page.
DATE_STATEMENT_ID = "2"
DATE_POPUP_PAGE = f"QueryWizard_DatePopup.aspx?statementId={DATE_STATEMENT_ID}&statementType=Filter"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
HEADER_ROWS = 2
TIMEOUT = (30, 600)  # (connect, read) seconds; large exports take a while
RETRIES = 3

log = logging.getLogger("asrs")
T = TypeVar("T")


class WizardError(Exception):
    """The wizard returned something other than what the next step expects."""


@dataclass(frozen=True)
class Window:
    """A range of whole months within one year."""

    year: int
    first: int  # 1-12
    last: int   # 1-12, inclusive

    def __str__(self) -> str:
        if self.first == self.last:
            return f"{MONTHS[self.first - 1][:3]} {self.year}"
        return f"{MONTHS[self.first - 1][:3]}-{MONTHS[self.last - 1][:3]} {self.year}"

    @property
    def filename(self) -> str:
        return f"{self.year}-{self.first:02d}-{self.last:02d}.csv"

    @property
    def criteria(self) -> str:
        """How the results page echoes this date filter."""
        return f"{MONTHS[self.first - 1]}-{self.year}, {MONTHS[self.last - 1]}-{self.year}"

    def split(self) -> list["Window"]:
        mid = (self.first + self.last) // 2
        return [Window(self.year, self.first, mid), Window(self.year, mid + 1, self.last)]


def half_years(start: tuple[int, int], end: tuple[int, int]) -> list[Window]:
    windows = []
    for year in range(start[0], end[0] + 1):
        for first, last in ((1, 6), (7, 12)):
            lo = max(first, start[1]) if year == start[0] else first
            hi = min(last, end[1]) if year == end[0] else last
            if lo <= hi:
                windows.append(Window(year, lo, hi))
    return windows


def form_state(html: str) -> tuple[BeautifulSoup, dict[str, str]]:
    """Parse a page and collect its hidden ASP.NET fields (__VIEWSTATE etc.)."""
    soup = BeautifulSoup(html, "html.parser")
    fields = {i["name"]: i.get("value", "")
              for i in soup.find_all("input", type="hidden") if i.get("name")}
    if "__VIEWSTATE" not in fields:
        raise WizardError("page has no ASP.NET form state")
    return soup, fields


def page_text(soup: BeautifulSoup) -> str:
    return " ".join(soup.get_text(" ").split())


class QueryWizard:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.current: Window | None = None  # window of the search held by the session
        self.reset()

    def reset(self) -> None:
        """Start over with a fresh server session."""
        self.session.close()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (compatible; ASRS batch downloader)"
        self.current = None

    def _get(self, page: str) -> requests.Response:
        r = self.session.get(BASE_URL + page, timeout=TIMEOUT)
        r.raise_for_status()
        return r

    def _post(self, page: str, data: dict[str, str]) -> requests.Response:
        r = self.session.post(BASE_URL + page, data=data, timeout=TIMEOUT)
        r.raise_for_status()
        return r

    def search(self, w: Window) -> int:
        """Run a search for reports in `w` and return how many matched."""
        self.current = None

        # 1. New, empty search (redirects to the filter page).
        page = self._get(NEW_QUERY_PAGE)
        _, state = form_state(page.text)

        # 2. Add the "Date of Incident" statement (a browser would open the date popup now).
        self._post(FILTER_PAGE, {**state, "__EVENTTARGET": f"-{DATE_STATEMENT_ID}",
                                 "__EVENTARGUMENT": ""})

        # 3. Fill in and save the date popup.
        soup, state = form_state(self._get(DATE_POPUP_PAGE).text)
        years = {o.get("value") for o in soup.select("#DropDownList1 option")}
        if not years:
            raise WizardError("date popup has no year list")
        if str(w.year) not in years:
            raise ValueError(f"{w.year} is not available in the wizard "
                             f"(it offers {min(years)}-{max(years)})")
        saved = self._post(DATE_POPUP_PAGE, {
            **state,
            "DropDownList1": str(w.year), "DropDownList2": MONTHS[w.first - 1],
            "DropDownList3": str(w.year), "DropDownList4": MONTHS[w.last - 1],
            "SaveButton.x": "1", "SaveButton.y": "1",
        })
        if "window.opener" not in saved.text:
            raise WizardError("date popup did not accept the date range")

        # 4. Run the search.
        soup, state = form_state(self._get(FILTER_PAGE).text)
        run = soup.find("input", alt=re.compile("Perform this search"))
        if run is None:
            raise WizardError("Run Search button not found")
        results = self._post(FILTER_PAGE, {**state, f"{run['name']}.x": "1",
                                           f"{run['name']}.y": "1"})

        # 5. Read the result count and check the search used our dates.
        soup = BeautifulSoup(results.text, "html.parser")
        text = page_text(soup)
        if f"Date of Incident was between {w.criteria}" not in text:
            raise WizardError(f"results page does not show the expected criteria ({w.criteria})")
        match = re.search(r"returned\s+([\d,]+)\s+ACN", text)
        if match is None:
            raise WizardError("result count not found on results page")
        self.current = w
        return int(match.group(1).replace(",", ""))

    def export_csv(self, w: Window) -> bytes:
        if self.current != w:
            self.search(w)
        return self._get(EXPORT_PAGE).content


def count_rows(data: bytes) -> int:
    """Number of non-blank report rows in an ASRS CSV export."""
    text = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if len(rows) < HEADER_ROWS or not rows[1] or rows[1][0] != "ACN":
        raise WizardError("export is not an ASRS CSV (header rows missing)")
    return sum(1 for row in rows[HEADER_ROWS:] if any(cell.strip() for cell in row))


class Downloader:
    def __init__(self, out_dir: Path, limit: int, delay: float, dry_run: bool) -> None:
        self.wizard = QueryWizard()
        self.out_dir = out_dir
        self.limit = limit
        self.delay = delay
        self.dry_run = dry_run
        self.failed: list[str] = []
        self.warnings: list[str] = []
        self.files = 0
        self.rows = 0

    def retry(self, what: str, fn: Callable[[], T]) -> T:
        for attempt in range(1, RETRIES + 1):
            try:
                return fn()
            except (requests.RequestException, WizardError) as e:
                if attempt == RETRIES:
                    raise
                wait = 30 * attempt
                log.warning("%s failed (%s); retrying in %ds", what, e, wait)
                time.sleep(wait)
                self.wizard.reset()
        raise AssertionError("unreachable")

    def process(self, w: Window) -> None:
        path = self.out_dir / w.filename
        if path.exists():
            log.info("%s: %s already exists, skipping", w, path.name)
            return

        time.sleep(self.delay)
        try:
            count = self.retry(f"search {w}", lambda: self.wizard.search(w))
        except (requests.RequestException, WizardError) as e:
            log.error("%s: search failed: %s", w, e)
            self.failed.append(f"{w}: search failed ({e})")
            return

        if count == 0:
            log.info("%s: no reports", w)
            return
        if count > self.limit:
            if w.first == w.last:
                log.error("%s: %d reports exceed the limit of %d and a single month "
                          "can't be split further; skipping", w, count, self.limit)
                self.failed.append(f"{w}: {count} reports in one month (over --limit)")
                return
            parts = w.split()
            log.info("%s: %d reports > %d, splitting into %s", w, count, self.limit,
                     " and ".join(map(str, parts)))
            for part in parts:
                self.process(part)
            return
        if self.dry_run:
            log.info("%s: %d reports -> would write %s", w, count, path.name)
            self.files += 1
            self.rows += count
            return

        log.info("%s: %d reports, exporting...", w, count)
        try:
            data = self.retry(f"export {w}", lambda: self.wizard.export_csv(w))
            rows = count_rows(data)
        except (requests.RequestException, WizardError) as e:
            log.error("%s: export failed: %s", w, e)
            self.failed.append(f"{w}: export failed ({e})")
            return

        if rows != count:
            msg = f"{w}: search reported {count} reports but the export has {rows}"
            log.warning(msg)
            self.warnings.append(msg)

        tmp = path.with_suffix(".csv.part")
        tmp.write_bytes(data)
        tmp.replace(path)
        self.files += 1
        self.rows += rows
        log.info("%s: wrote %s (%d rows, %.1f MB)", w, path.name, rows, len(data) / 1e6)


def year_month(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", value)
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM, got {value!r}")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=year_month, default=(1988, 1),
                        help="first month, YYYY-MM (default: 1988-01, the start of the database)")
    parser.add_argument("--end", type=year_month, default=(2023, 12),
                        help="last month, YYYY-MM (default: 2023-12)")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=5000,
                        help="max reports per export; larger windows are split (default: 5000)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds to wait before each search, to go easy on the server")
    parser.add_argument("--dry-run", action="store_true",
                        help="only run the searches and print report counts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    if args.start > args.end:
        parser.error("--start is after --end")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    windows = half_years(args.start, args.end)
    dl = Downloader(args.out_dir, args.limit, args.delay, args.dry_run)
    try:
        for i, w in enumerate(windows, 1):
            log.info("[%d/%d] %s", i, len(windows), w)
            dl.process(w)
    except ValueError as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        log.warning("Interrupted; run again to continue where it stopped")
        sys.exit(130)

    verb = "Would write" if args.dry_run else "Wrote"
    log.info("%s %d files, %d reports, to %s", verb, dl.files, dl.rows, args.out_dir)
    for msg in dl.warnings:
        log.warning(msg)
    if dl.failed:
        log.error("%d windows were not downloaded:", len(dl.failed))
        for msg in dl.failed:
            log.error("  %s", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
