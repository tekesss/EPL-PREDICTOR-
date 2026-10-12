#!/usr/bin/env python3
"""
EPL data ingestion — Tier 1 free sources.

Pulls two complementary layers and writes them to ./data/raw/:

  1. football-data.co.uk  -> results (1993/94 ->), match stats (2000/01 ->),
                             and bookmaker odds incl. closing lines.
  2. understat.com        -> per-match xG / npxG / PPDA / deep / xPts (2014/15 ->)
                             plus Understat's own pre-match win probabilities
                             and the upcoming fixture list.

Deliberately NOT included here:
  - FBref (possession %, Opta xG): behind Cloudflare. Use the `soccerdata`
    package, which drives a real browser. See notes at the bottom.
  - Any API requiring a key.

Usage:
    pip install requests pandas pyarrow
    python fetch_epl_data.py --from-season 2000 --to-season 2026
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/raw")
UA = {"User-Agent": "Mozilla/5.0 (compatible; epl-research/1.0)"}

FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
US_URL = "https://understat.com/league/EPL/{year}"
# Newer Understat builds load data via XHR instead of inline <script> JSON.
US_XHR = "https://understat.com/getLeagueData/EPL/{year}"

# Understat embeds its payloads as JSON.parse('\x7B...\x7D') inside <script> tags.
US_BLOCK = re.compile(r"var\s+(\w+)\s*=\s*JSON\.parse\(\s*'(.*?)'\s*\)\s*;", re.S)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def season_code(start_year: int) -> str:
    """2024 -> '2425'  (football-data.co.uk directory convention)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2024 -> '2024-25'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def get(url: str, retries: int = 3, pause: float = 2.0) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            print(f"    HTTP {r.status_code} on {url}", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(pause * (attempt + 1))
    return None


# --------------------------------------------------------------------------- #
# 1. football-data.co.uk
# --------------------------------------------------------------------------- #
def fetch_football_data(start: int, end: int) -> pd.DataFrame:
    frames = []
    for year in range(start, end + 1):
        url = FD_URL.format(code=season_code(year))
        print(f"  football-data.co.uk {season_label(year)} ...")
        r = get(url)
        if r is None:
            print(f"    no file for {season_label(year)} (season not started?)")
            continue
        # Files are latin-1 and occasionally carry trailing all-empty columns.
        df = pd.read_csv(StringIO(r.content.decode("latin-1")), on_bad_lines="skip")
        df = df.dropna(axis=1, how="all").dropna(subset=["Date"])
        df["season"] = season_label(year)
        frames.append(df)
        time.sleep(1.0)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    # Two date formats coexist across the archive (dd/mm/yy and dd/mm/yyyy).
    out["date"] = pd.to_datetime(out["Date"], dayfirst=True, format="mixed", errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. understat.com
# --------------------------------------------------------------------------- #
def parse_understat(html: str) -> dict[str, object]:
    """Return every JSON.parse(...) payload on the page, keyed by JS var name."""
    blobs = {}
    for name, raw in US_BLOCK.findall(html):
        # The payload is a JS string literal of \xNN escapes.
        decoded = raw.encode("utf-8").decode("unicode_escape")
        try:
            blobs[name] = json.loads(decoded)
        except json.JSONDecodeError as exc:
            print(f"    could not decode '{name}': {exc}", file=sys.stderr)
    return blobs


def fetch_understat(start: int, end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (team_match_rows, fixture_rows).

    team_match_rows: one row per team per match — xG, npxG, PPDA, deep, xPts.
    fixture_rows:    one row per match — teams, goals, and Understat's own
                     pre-match forecast probabilities (w / d / l).
    """
    team_rows, fixture_rows = [], []

    for year in range(start, end + 1):
        print(f"  understat EPL {season_label(year)} ...")
        r = get(US_URL.format(year=year))
        if r is None:
            continue
        blobs = parse_understat(r.text)
        if not blobs.get("datesData"):
            # Fallback: page no longer embeds JSON -> hit the XHR endpoint.
            x = requests.get(US_XHR.format(year=year), timeout=30, headers={
                **UA, "X-Requested-With": "XMLHttpRequest",
                "Referer": US_URL.format(year=year)})
            if x.ok:
                j = x.json()
                blobs = {"teamsData": j.get("teams") or j.get("teamsData"),
                         "datesData": j.get("dates") or j.get("datesData")}
            else:
                print(f"    understat XHR HTTP {x.status_code}", file=sys.stderr)

        # --- per-team match history -------------------------------------- #
        for team in (blobs.get("teamsData") or {}).values():
            name = team.get("title")
            for match in team.get("history", []):
                row = {"season": season_label(year), "team": name}
                row.update(match)
                team_rows.append(row)

        # --- fixtures + forecasts ---------------------------------------- #
        for m in blobs.get("datesData") or []:
            fixture_rows.append(
                {
                    "season": season_label(year),
                    "match_id": m.get("id"),
                    "datetime": m.get("datetime"),
                    "is_result": m.get("isResult"),
                    "home_team": (m.get("h") or {}).get("title"),
                    "away_team": (m.get("a") or {}).get("title"),
                    "home_goals": (m.get("goals") or {}).get("h"),
                    "away_goals": (m.get("goals") or {}).get("a"),
                    "home_xg": (m.get("xG") or {}).get("h"),
                    "away_xg": (m.get("xG") or {}).get("a"),
                    "p_home": (m.get("forecast") or {}).get("w"),
                    "p_draw": (m.get("forecast") or {}).get("d"),
                    "p_away": (m.get("forecast") or {}).get("l"),
                }
            )
        time.sleep(2.0)

    tdf = pd.DataFrame(team_rows)
    fdf = pd.DataFrame(fixture_rows)
    if not fdf.empty:
        fdf["datetime"] = pd.to_datetime(fdf["datetime"], errors="coerce")
        fdf = fdf.sort_values("datetime").reset_index(drop=True)
    return tdf, fdf


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-season", type=int, default=2000,
                    help="first season START year (2000 = 2000/01)")
    ap.add_argument("--to-season", type=int, default=2026,
                    help="last season START year (2026 = 2026/27)")
    ap.add_argument("--skip-understat", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/2] football-data.co.uk")
    fd = fetch_football_data(args.from_season, args.to_season)
    if not fd.empty:
        fd.to_parquet(OUT / "football_data_e0.parquet", index=False)
        fd.to_csv(OUT / "football_data_e0.csv", index=False)
        print(f"  -> {len(fd):,} matches, {fd.shape[1]} columns, "
              f"{fd['date'].min():%Y-%m-%d} to {fd['date'].max():%Y-%m-%d}")

    if not args.skip_understat:
        print("[2/2] understat.com (xG available from 2014/15 only)")
        lo = max(args.from_season, 2014)
        tdf, fdf = fetch_understat(lo, args.to_season)
        if not tdf.empty:
            tdf.to_parquet(OUT / "understat_team_matches.parquet", index=False)
            print(f"  -> {len(tdf):,} team-match rows")
        if not fdf.empty:
            fdf.to_parquet(OUT / "understat_fixtures.parquet", index=False)
            played = int(fdf["is_result"].fillna(False).astype(bool).sum())
            print(f"  -> {len(fdf):,} fixtures ({played:,} played, "
                  f"{len(fdf) - played:,} upcoming)")

    print(f"\nWritten to {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# --------------------------------------------------------------------------- #
# Possession % and Opta xG (FBref) — separate path, needs a browser:
#
#   pip install "soccerdata[selenium]"
#   import soccerdata as sd
#   fb = sd.FBref("ENG-Premier League", ["2017-2018", ..., "2025-2026"])
#   schedule   = fb.read_schedule()
#   team_stats = fb.read_team_match_stats(stat_type="possession")
#
# Hard limit: 10 requests / minute on *.sports-reference.com. Exceed it and the
# session is blocked for up to a day. Cache aggressively; never run it in CI
# without a local cache layer.
# --------------------------------------------------------------------------- #
