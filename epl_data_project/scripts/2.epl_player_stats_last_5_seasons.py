#!/usr/bin/env python3
"""
Pull EPL player stats from Understat for all seasons since a start year (default 2020)
up to the current season (by start year; Aug=season rollover).

Outputs:
  ../data/raw/player_stats/understat_epl_players_since2020.parquet
  (optional) CSV alongside it if --csv is passed

Usage:
  python understat_epl_since2020.py                # 2020 .. current season
  python understat_epl_since2020.py --since 2018   # 2018 .. current season
  python understat_epl_since2020.py --seasons 2020 2021 2022 2023 2024 --csv
"""

import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List

import pandas as pd
import requests

UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{season}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; stats-pipeline/1.0; +https://example.com)"
}

# ---------------- helpers ----------------

def seasons_since(start_year: int = 2020, tz: str = "Europe/London") -> List[int]:
    """
    Return [start_year, ..., current_start_year], where current_start_year
    is the football season start year (Aug=rollover).
    """
    now_uk = pd.Timestamp.now(tz=tz)
    current_start = now_uk.year if now_uk.month >= 8 else now_uk.year - 1
    return list(range(start_year, current_start + 1))

def fetch_html(url: str, session: requests.Session, retries: int = 3, backoff: float = 1.5) -> str:
    for attempt in range(1, retries + 1):
        resp = session.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.text
        if attempt < retries:
            time.sleep(backoff ** attempt)
    resp.raise_for_status()
    return ""  # unreachable

def _decode_jsonparse_payload(s: str) -> Any:
    unescaped = bytes(s, "utf-8").decode("unicode_escape")
    return json.loads(unescaped)

def extract_players_data(html: str) -> List[Dict[str, Any]]:
    """
    Understat embeds playersData either as JSON.parse('...') or a raw array.
    """
    m = re.search(r"playersData\s*=\s*JSON\.parse\('([^']+)'\)", html)
    if m:
        return _decode_jsonparse_payload(m.group(1))

    m2 = re.search(r"playersData\s*=\s*(\[[\s\S]*?\]);", html)
    if m2:
        raw = m2.group(1)
        return json.loads(raw)

    raise ValueError("Could not locate playersData in page")

def flatten_player_record(p: Dict[str, Any], season: int) -> Dict[str, Any]:
    out = dict(p)
    out["season"] = int(season)

    # coerce numerics
    num_as_float = [
        "xG", "xA", "npxG", "xGChain", "xGBuildup",
        "npxGPer90", "xGPer90", "xAPer90"
    ]
    num_as_int = ["games", "time", "goals", "assists", "shots", "key_passes",
                  "yellow_cards", "red_cards"]

    for k in num_as_float:
        if k in out:
            try: out[k] = float(out[k])
            except Exception: pass

    for k in num_as_int:
        if k in out:
            try: out[k] = int(out[k])
            except Exception:
                try: out[k] = int(float(out[k]))
                except Exception: pass

    # some pages use "team" vs "team_title"
    if "team" in out and "team_title" not in out:
        out["team_title"] = out["team"]

    return out

def fetch_epl_players_for_season(season: int, session: requests.Session) -> pd.DataFrame:
    url = UNDERSTAT_LEAGUE_URL.format(season=season)
    html = fetch_html(url, session=session)
    players = extract_players_data(html)
    rows = [flatten_player_record(p, season) for p in players]
    df = pd.DataFrame(rows)

    # column preference/order
    preferred = [
        "season", "id", "player_name", "position", "team_title",
        "games", "time", "goals", "assists", "shots", "key_passes",
        "xG", "xA", "npxG", "xGChain", "xGBuildup",
        "xGPer90", "xAPer90", "npxGPer90",
        "yellow_cards", "red_cards", "age"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Pull EPL player stats from Understat since a start year (default 2020).")
    ap.add_argument("--since", type=int, default=2020, help="Start season (by start year), default 2020.")
    ap.add_argument("--seasons", nargs="*", type=int,
                    help="Explicit list of seasons (start years), overrides --since.")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests.")
    ap.add_argument("--out", type=str, default="understat_epl_players_since2020.parquet", help="Output Parquet filename.")
    ap.add_argument("--csv", action="store_true", help="Also export CSV next to the Parquet.")
    args = ap.parse_args()

    if args.seasons:
        seasons = sorted(set(int(s) for s in args.seasons))
    else:
        seasons = seasons_since(args.since)

    print(f"Fetching EPL player stats for seasons: {seasons}")

    dfs = []
    with requests.Session() as session:
        for s in seasons:
            try:
                print(f"  - Season {s} …", end="", flush=True)
                df = fetch_epl_players_for_season(s, session)
                print(f" got {len(df):,} players")
                dfs.append(df)
            except Exception as e:
                print(f" failed: {e}")
            time.sleep(args.sleep)

    if not dfs:
        raise SystemExit("No data fetched. Aborting.")

    all_df = pd.concat(dfs, ignore_index=True)

    # De-dup in case of retries or page quirks:
    # keep one row per (season, player id, team), since players can transfer mid-season
    subset = [c for c in ["season", "id", "team_title"] if c in all_df.columns]
    if subset:
        all_df = all_df.drop_duplicates(subset=subset).reset_index(drop=True)
    else:
        all_df = all_df.drop_duplicates().reset_index(drop=True)

    # Save Parquet under ../data/raw/player_stats/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "player_stats"))
    os.makedirs(base_dir, exist_ok=True)
    parquet_path = os.path.join(base_dir, os.path.basename(args.out))
    all_df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet to {parquet_path}")

    if args.csv:
        csv_path = parquet_path.replace(".parquet", ".csv")
        all_df.to_csv(csv_path, index=False)
        print(f"Also saved CSV to {csv_path}")

if __name__ == "__main__":
    main()
