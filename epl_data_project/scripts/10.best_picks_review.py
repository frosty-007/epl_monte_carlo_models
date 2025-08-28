#!/usr/bin/env python3
"""
10.best_picks_review.py  (ROUNDS-FIRST VERSION)

Grade picks (singles + accas) using results from:
  1) ESPN public scoreboard (no API key, default ON),
  2) Football-Data.co.uk season CSVs (E0, no API key, default ON),
  3) Understat (no key; sometimes lags — fallback),
  4) API-Football (optional key)  <-- used to pull official ROUND (1..38),
  5) Optional local CSV/JSON fallback.

NEW IN THIS VERSION:
- INPUT picking by **EPL Round** via --picks-round (file pattern: picks_next7_epl_<season>_r_<NN>.json).
- OUTPUT filename uses **EPL Round** only: epl_<season>_rNN_pick_analysis.json or epl_<season>_rAA-BB_... if multiple rounds are present.
- Each matched pick carries a `round` (int) based on API-Football's league.round ("Regular Season - N").
- If no round info is available at all, we keep working but fall back to a picks-basename output filename.

Examples:
  python 10.best_picks_review.py --debug
  python 10.best_picks_review.py --api-football --api-football-key YOUR_KEY
  python 10.best_picks_review.py --picks-round 2 --picks-season 2025 --api-football --api-football-key YOUR_KEY
  python 10.best_picks_review.py --out ../data/output/pick_analysis/custom.json

Requires:
  pip install aiohttp understat pandas numpy requests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from difflib import SequenceMatcher

# ---------- JSON helper ----------
def _json_default(o):
    if isinstance(o, pd.Timestamp):
        return None if pd.isna(o) else o.isoformat()
    if isinstance(o, pd.Timedelta):
        return str(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)

# ---------- Defaults ----------
DEFAULT_PICKS_PRIMARY  = Path("../data/output/picks/picks_next7.json")
DEFAULT_PICKS_FALLBACK = Path("../data/output/picks/best_value_next7.json")
DEFAULT_PICKS_DIR      = Path("../data/output/picks")
DEFAULT_OUT_DIR        = Path("../data/output/pick_analysis")
DEFAULT_LEAGUE         = "EPL"

# ---------- Canonicalization ----------
def canon_team(s: str) -> str:
    s0 = (s or "").lower()
    s1 = re.sub(r"[^a-z0-9]+", "", s0)
    s1 = re.sub(r"fc$", "", s1)
    syn = {
        "manchesterunited":"manutd","manutd":"manutd","manchesterutd":"manutd","mufc":"manutd",
        "manchestercity":"mancity","mancity":"mancity","mcfc":"mancity",
        "tottenhamhotspur":"tottenham","tottenham":"tottenham","spurs":"tottenham",
        "arsenal":"arsenal","chelsea":"chelsea","liverpool":"liverpool",
        "westhamunited":"westham","westham":"westham","hammers":"westham",
        "newcastleunited":"newcastle","newcastle":"newcastle","magpies":"newcastle",
        "brightonandhovealbion":"brighton","brightonhovealbion":"brighton","brighton":"brighton","seagulls":"brighton",
        "wolverhamptonwanderers":"wolves","wolverhampton":"wolves","wolves":"wolves",
        "nottinghamforest":"forest","nottmforest":"forest","forest":"forest",
        "sheffieldunited":"sheffieldutd","sheffieldutd":"sheffieldutd","sheffutd":"sheffieldutd",
        "crystalpalace":"crystalpalace","palace":"crystalpalace",
        "everton":"everton","fulham":"fulham","brentford":"brentford",
        "bournemouth":"bournemouth","burnley":"burnley","southampton":"southampton",
        "astonvilla":"astonvilla","villa":"astonvilla",
        "lutontown":"luton","luton":"luton",
        "ipswichtown":"ipswich","ipswich":"ipswich",
        "leicestercity":"leicester","leicester":"leicester",
        "westbromwichalbion":"westbrom","westbrom":"westbrom",
    }
    return syn.get(s1, s1)

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Grade picks using ESPN → Football-Data → Understat → API-Football → optional local fallback.")

    # INPUT picks
    ap.add_argument("--picks", type=Path, default=DEFAULT_PICKS_PRIMARY,
                    help=f"Explicit picks JSON (default {DEFAULT_PICKS_PRIMARY}, fallback {DEFAULT_PICKS_FALLBACK})")
    ap.add_argument("--picks-round", type=int, default=None,
                    help="EPL round (1..38) to load picks file named picks_next_epl_<season>_mw_<NN>.json (alias of --picks-mw)")
    ap.add_argument("--picks-mw", type=int, default=None,
                    help="Alias for --picks-round; same behavior.")
    ap.add_argument("--picks-season", type=int, default=None,
                    help="Season starting year (e.g., 2025). If omitted, inferred from today's UK date.")

    # Providers & general
    ap.add_argument("--league", type=str, default=DEFAULT_LEAGUE,
                    help="Understat league key (e.g., 'EPL' or 'epl')")
    ap.add_argument("--seasons", type=str, default=None,
                    help="Comma-separated seasons, e.g. '2024,2025'. If omitted, inferred from picks.")
    ap.add_argument("--api-football", action="store_true", help="Enable API-Football fetch (recommended for official ROUND).")
    ap.add_argument("--api-football-key", type=str, default=None, help="API-Football key (or set env API_FOOTBALL_KEY)")
    ap.add_argument("--api-football-league-id", type=int, default=39, help="API-Football league id (EPL=39)")
    ap.add_argument("--no-espn", action="store_true", help="Disable ESPN public scoreboard provider (enabled by default).")
    ap.add_argument("--fallback-results", type=Path, default=None,
                    help="Optional CSV/JSON with results to use if needed.")

    # Output & debug
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Output graded JSON. Default uses inferred/forced EPL ROUND: {DEFAULT_OUT_DIR}/epl_<season>_rNN[_NN]_pick_analysis.json")
    ap.add_argument("--force-round", type=int, default=None, help="Force EPL round number for output naming (1..38).")
    ap.add_argument("--debug", action="store_true", help="Print extra logs and write debug_unmatched.csv")

    return ap.parse_args()

# ---------- Picks I/O ----------
def load_picks_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _season_year_now_uk() -> int:
    d = pd.Timestamp.now(tz="Europe/London").date()
    return d.year if d.month >= 8 else d.year - 1

def resolve_picks_path(p: Path) -> Path:
    if p.exists():
        return p
    raise FileNotFoundError(f"Picks file not found at {p}")

def resolve_picks_path_by_round(
    round_num: int,
    season_year: Optional[int] = None,
    base_dir: Path = DEFAULT_PICKS_DIR,
    strict: bool = False,
) -> Path:
    """
    Return the path to the picks file for a given EPL round using pattern:
      picks_next_epl_<season>_mw_<NN>.json
    Accepts a few common variants if strict=False.
    """
    if round_num is None or int(round_num) < 1 or int(round_num) > 38:
        raise ValueError(f"round_num must be 1..38, got {round_num}")

    season_year = int(season_year) if season_year is not None else _season_year_now_uk()
    rr = int(round_num)

    candidates = [
        base_dir / f"picks_next_epl_{season_year}_mw_{rr:02d}.json",
        base_dir / f"picks_next_epl_{season_year}_r_{rr:02d}.json",
    ]
    if not strict:
        candidates += [
            base_dir / f"picks_next_epl_{season_year}_mw{rr:02d}.json",
            base_dir / f"picks_next_epl_{season_year}_r{rr:02d}.json",
            # legacy 'next7' variants kept for backward compatibility
            base_dir / f"picks_next7_epl_{season_year}_mw_{rr:02d}.json",
            base_dir / f"picks_next7_epl_{season_year}_mw{rr:02d}.json",
            base_dir / f"picks_next7_epl_{season_year}_r_{rr:02d}.json",
            base_dir / f"picks_next7_epl_{season_year}_r{rr:02d}.json",
        ]
        patt = f"picks_*epl_{season_year}_*{rr:02d}*.json"
        for p in base_dir.glob(patt):
            candidates.append(p)

    for p in candidates:
        if p.exists():
            return p

    tried = ", ".join([str(p) for p in candidates[:5]])
    raise FileNotFoundError(
        f"No picks file found for season={season_year}, round={rr}. Tried: {tried} (and scanned {base_dir})"
    )

def infer_seasons_from_picks(singles: List[dict]) -> List[int]:
    """Season is the starting year (e.g., 2024 for 2024/25)."""
    if not singles:
        d = pd.Timestamp.now(tz="Europe/London").date()
        return [d.year if d.month >= 8 else d.year - 1]
    dt = pd.to_datetime(
        [s.get("kickoff_uk") or s.get("kickoff_utc") for s in singles],
        utc=True, errors="coerce"
    ).tz_convert("Europe/London").dropna()
    if len(dt) == 0:
        d = pd.Timestamp.now(tz="Europe/London").date()
        return [d.year if d.month >= 8 else d.year - 1]
    seasons = np.where(dt.month >= 8, dt.year, dt.year - 1)
    return sorted(pd.unique(seasons).tolist())

# ---------- Utilities ----------
def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _parse_score_like(x) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(x):
        return None, None
    s = str(x)
    m = re.match(r"^\s*(\d+)\s*[:\-]\s*(\d+)\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def _ratio(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _build_pair_key(hk: str, ak: str) -> tuple:
    return tuple(sorted([str(hk or ""), str(ak or "")]))

def _nearest_by_dt(cand: pd.DataFrame, ref_dt: pd.Timestamp) -> pd.Series:
    cand = cand.copy()
    cand["dt"] = pd.to_datetime(cand["kickoff_uk"], errors="coerce")
    cand["abs_diff"] = (cand["dt"] - ref_dt).abs()
    return cand.sort_values(["abs_diff"]).iloc[0]

def _to_uk_aware(x) -> pd.Timestamp:
    t = pd.to_datetime(x, errors="coerce")
    if t.tzinfo is None:
        return t.tz_localize("Europe/London")
    return t.tz_convert("Europe/London")

# ---------- ESPN public scoreboard (no key) ----------
def fetch_espn_results(date_from, date_to, debug: bool=False) -> pd.DataFrame:
    """
    Fetch Premier League **completed** results from ESPN public scoreboard (no API key).
    Endpoint:
      https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates=YYYYMMDD
    """
    import requests
    base = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
    rows = []

    def _fetch_day(day_str: str) -> Optional[dict]:
        for q in ({"dates": day_str}, {"dates": f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:]}"}):
            try:
                r = requests.get(base, params=q, timeout=30)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if debug:
                    print(f"[warn] ESPN fetch failed for {q['dates']}: {e}")
        return None

    dmin = _to_uk_aware(date_from); dmax = _to_uk_aware(date_to)
    if dmin > dmax: dmin, dmax = dmax, dmin

    for day in pd.date_range(dmin.date(), dmax.date(), freq="D"):
        datestr = day.strftime("%Y%m%d")
        data = _fetch_day(datestr)
        if not data:
            continue

        events = data.get("events", []) or []
        for ev in events:
            comps = ev.get("competitions", []) or []
            comp = comps[0] if comps else {}
            status = (comp.get("status") or {}).get("type") or {}
            if not status.get("completed", False):
                continue

            competitors = comp.get("competitors", []) or []
            home_name = away_name = None
            home_score = away_score = None
            for c in competitors:
                t = (c.get("team") or {})
                name = t.get("name") or t.get("shortDisplayName") or t.get("displayName")
                score = c.get("score")
                if c.get("homeAway") == "home":
                    home_name, home_score = name, score
                elif c.get("homeAway") == "away":
                    away_name, away_score = name, score

            if home_name is None or away_name is None:
                continue
            try:
                hg = int(home_score); ag = int(away_score)
            except Exception:
                continue

            ko = pd.to_datetime(ev.get("date") or comp.get("date"), utc=True, errors="coerce")
            rows.append({
                "home_team": home_name, "away_team": away_name,
                "home_key": canon_team(home_name), "away_key": canon_team(away_name),
                "home_goals": hg, "away_goals": ag,
                "kickoff_utc": ko,
                "kickoff_uk": ko.tz_convert("Europe/London") if ko is not None else pd.NaT,
                "kick_date_uk": (ko.tz_convert("Europe/London").date() if ko is not None else None),
                # No round info in ESPN endpoint; leave None
                "matchweek": None,
            })

    out = pd.DataFrame(rows)
    if debug:
        print(f"[debug] ESPN: {'no rows' if out.empty else f'{len(out)} rows'}")
    return out

# ---------- Football-Data.co.uk (no key) ----------
def fetch_football_data_results(seasons: List[int], date_from, date_to, debug: bool=False) -> pd.DataFrame:
    """
    football-data.co.uk season CSVs (no API key). EPL = E0.
    URL: https://www.football-data.co.uk/mmz4281/YYZZ/E0.csv  (YYZZ e.g. 2526 for 2025/26)
    """
    import requests
    rows = []
    dmin = _to_uk_aware(date_from); dmax = _to_uk_aware(date_to)
    if dmin > dmax: dmin, dmax = dmax, dmin

    for s in seasons:
        yy = f"{s%100:02d}{(s+1)%100:02d}"
        url = f"https://www.football-data.co.uk/mmz4281/{yy}/E0.csv"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
        except Exception as e:
            if debug: print(f"[warn] football-data fetch failed for {s}: {e}")
            continue

        if not {"Date","HomeTeam","AwayTeam","FTHG","FTAG"}.issubset(df.columns):
            if debug: print(f"[warn] football-data unexpected schema for {s}: {list(df.columns)[:12]}")
            continue

        dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        dt_uk = dt.dt.tz_localize("Europe/London", nonexistent="NaT", ambiguous="NaT")
        mask = (dt_uk >= dmin) & (dt_uk <= dmax)
        df = df.loc[mask].copy()

        df["home_team"]  = df["HomeTeam"].astype(str)
        df["away_team"]  = df["AwayTeam"].astype(str)
        df["home_key"]   = df["home_team"].map(canon_team)
        df["away_key"]   = df["away_team"].map(canon_team)
        df["home_goals"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["away_goals"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df["kickoff_uk"] = dt_uk.loc[df.index]
        df["kick_date_uk"] = df["kickoff_uk"].dt.date
        df["matchweek"] = None

        out = df[["home_team","away_team","home_key","away_key",
                  "home_goals","away_goals","kickoff_uk","kick_date_uk","matchweek"]].dropna(subset=["home_goals","away_goals"])
        rows.append(out)

    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(
        columns=["home_team","away_team","home_key","away_key","home_goals","away_goals","kickoff_uk","kick_date_uk","matchweek"]
    )
    if debug and not out.empty:
        print(f"[info] football-data rows fetched: {len(out)} (UK: {out['kickoff_uk'].min()} .. {out['kickoff_uk'].max()})")
    return out

# ---------- Understat fetch & standardize ----------
def _extract_title_from_obj(v: Any, id2title: Optional[Dict[int, str]]) -> Optional[str]:
    if isinstance(v, dict):
        for k in ("title","name","teamTitle","team_name","short_title","shortName"):
            if k in v and v[k]:
                return str(v[k])
        if "team" in v and isinstance(v["team"], dict):
            for k in ("title","name","short_title"):
                if k in v["team"] and v["team"][k]:
                    return str(v["team"][k])
        for k in ("id","team_id","teamId"):
            if k in v and v[k] is not None:
                try:
                    return id2title.get(int(v[k])) if id2title else str(v[k])
                except Exception:
                    return str(v[k])
        return None
    try:
        vi = int(v)
        return id2title.get(vi) if id2title else str(vi)
    except Exception:
        return str(v) if v is not None else None

def _series_title_from_candidates(df: pd.DataFrame, cands: List[str], id2title: Optional[Dict[int,str]]) -> Optional[pd.Series]:
    for c in cands:
        if c in df.columns:
            s = df[c]
            if s.dtype == "object":
                vals = s.apply(lambda v: _extract_title_from_obj(v, id2title))
                if vals.notna().any():
                    return vals.astype(str)
            return s.astype(str)
    return None

def _standardize_understat(df: pd.DataFrame, id2title: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    cols = ["home_team","away_team","home_key","away_key",
            "home_goals","away_goals","kickoff_utc","kickoff_uk","kick_date_uk","matchweek"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    if {"h.title","a.title","goals.h","goals.a","datetime"}.issubset(df.columns):
        out = pd.DataFrame()
        out["home_team"]  = df["h.title"].astype(str)
        out["away_team"]  = df["a.title"].astype(str)
        out["home_key"]   = out["home_team"].map(canon_team)
        out["away_key"]   = out["away_team"].map(canon_team)
        out["home_goals"] = pd.to_numeric(df["goals.h"], errors="coerce")
        out["away_goals"] = pd.to_numeric(df["goals.a"], errors="coerce")
        out["kickoff_utc"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        out["kickoff_uk"]  = out["kickoff_utc"].dt.tz_convert("Europe/London")
        out["kick_date_uk"] = out["kickoff_uk"].dt.date
        out["matchweek"] = None
        out = out.dropna(subset=["home_goals","away_goals"], how="any")
        return out.reindex(columns=cols)

    home_title = _series_title_from_candidates(df, [
        "h_title","home.title","homeTeam.title","team_h.title","homeTeamTitle","home_team",
        "home.team.title","h.title","homeTitle","home.name","homeTeamName","home"
    ], id2title)
    away_title = _series_title_from_candidates(df, [
        "a_title","away.title","awayTeam.title","team_a.title","awayTeamTitle","away_team",
        "away.team.title","a.title","awayTitle","away.name","awayTeamName","away"
    ], id2title)
    if home_title is None or away_title is None:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame()
    out["home_team"] = home_title.astype(str)
    out["away_team"] = away_title.astype(str)
    out["home_key"]  = out["home_team"].map(canon_team)
    out["away_key"]  = out["away_team"].map(canon_team)

    hg_col = _first_existing(df, ["home_goals","goals_home","h_goals","home.score","scores.home","result.home","goals.h","goals.home"])
    ag_col = _first_existing(df, ["away_goals","goals_away","a_goals","away.score","scores.away","result.away","goals.a","goals.away"])
    hg = pd.to_numeric(df[hg_col], errors="coerce") if hg_col else pd.NA
    ag = pd.to_numeric(df[ag_col], errors="coerce") if ag_col else pd.NA
    if (isinstance(hg, pd.Series) and hg.isna().all()) and (isinstance(ag, pd.Series) and ag.isna().all()):
        sc_col = _first_existing(df, ["score","ftScore","fulltime","finalScore"])
        if sc_col:
            parsed = df[sc_col].apply(_parse_score_like)
            tmp = pd.DataFrame(parsed.tolist(), columns=["H","A"])
            hg, ag = tmp["H"], tmp["A"]

    out["home_goals"] = pd.to_numeric(hg, errors="coerce") if isinstance(hg, pd.Series) else pd.NA
    out["away_goals"] = pd.to_numeric(ag, errors="coerce") if isinstance(ag, pd.Series) else pd.NA

    dt_col = _first_existing(df, ["datetime","kickoff_time","kickoff","date","startTime","start_time","time","datetimeUTC","eventDate"])
    out["kickoff_utc"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce") if dt_col else pd.NaT
    out["kickoff_uk"]  = out["kickoff_utc"].dt.tz_convert("Europe/London")
    out["kick_date_uk"] = out["kickoff_uk"].dt.date
    out["matchweek"] = None
    out = out.dropna(subset=["home_goals","away_goals"], how="any")
    return out.reindex(columns=cols)

async def fetch_understat_matches(league: str, seasons: List[int], debug: bool=False) -> pd.DataFrame:
    try:
        from understat import Understat
        import aiohttp
    except Exception as e:
        if debug: print(f"[warn] Understat import failed: {e}")
        return pd.DataFrame(columns=[
            "home_team","away_team","home_key","away_key","home_goals","away_goals","kickoff_utc","kickoff_uk","kick_date_uk","matchweek"
        ])

    EMPTY = pd.DataFrame(columns=[
        "home_team","away_team","home_key","away_key","home_goals","away_goals","kickoff_utc","kickoff_uk","kick_date_uk","matchweek"
    ])

    try:
        league_key = (league or "").strip().lower()
        frames = []
        async with aiohttp.ClientSession() as session:
            us = Understat(session)
            for s in seasons:
                for season_try in (s, s - 1):
                    get_fixtures = getattr(us, "get_league_fixtures", None)
                    get_matches  = getattr(us, "get_league_matches", None)
                    data = await (get_fixtures(league_key, season_try) if get_fixtures else get_matches(league_key, season_try))
                    raw = pd.json_normalize(data) if data is not None else pd.DataFrame()
                    if raw.empty:
                        if season_try == s: continue
                        else: break
                    std = _standardize_understat(raw, id2title=None)
                    if std.empty:
                        if debug:
                            print(f"[warn] Understat std empty for {league_key} {season_try} (schema?)")
                            print("[debug] columns sample:", list(raw.columns)[:40])
                        if season_try == s: continue
                        else: break
                    frames.append(std); break
        if not frames: return EMPTY
        out = pd.concat(frames, ignore_index=True, sort=False)
        out["kick_date_uk"] = pd.to_datetime(out["kick_date_uk"], errors="coerce").dt.date
        if debug and not out.empty:
            print(f"[info] Understat rows fetched: {len(out)}")
        return out
    except Exception as e:
        if debug: print(f"[warn] Understat fetch failed: {e}")
        return EMPTY

# ---------- API-Football (prefers round; includes scheduled fixtures) ----------
def fetch_api_football_results(league_id: int, seasons: List[int], date_from, date_to, api_key: Optional[str], debug: bool=False) -> pd.DataFrame:
    import requests
    if not api_key:
        if debug: print("[warn] API-Football key not provided; skipping.")
        return pd.DataFrame(columns=[
            "home_team","away_team","home_key","away_key","home_goals","away_goals",
            "kickoff_utc","kickoff_uk","kick_date_uk","matchweek","status_short"
        ])

    base = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    rows = []
    dmin = _to_uk_aware(date_from); dmax = _to_uk_aware(date_to)
    if dmin > dmax: dmin, dmax = dmax, dmin

    for season in seasons:
        for day in pd.date_range(dmin.date(), dmax.date(), freq="D"):
            params = {"league": int(league_id), "season": int(season), "date": day.strftime("%Y-%m-%d")}
            try:
                r = requests.get(base, headers=headers, params=params, timeout=30)
                r.raise_for_status()
                resp = r.json().get("response", [])
            except Exception as e:
                if debug: print(f"[warn] API-Football {day.date()} request failed: {e}")
                continue
            for item in resp:
                st = (item.get("fixture", {}).get("status", {}) or {}).get("short")
                home = item["teams"]["home"]["name"]; away = item["teams"]["away"]["name"]
                goals = item.get("goals", {}) or {}
                hg = goals.get("home"); ag = goals.get("away")
                ko_utc = pd.to_datetime(item["fixture"]["date"], utc=True, errors="coerce")
                rnd = (item.get("league", {}) or {}).get("round")
                mw = None
                if isinstance(rnd, str):
                    m = re.search(r"(\d+)$", rnd)  # "Regular Season - 3" -> 3
                    if m:
                        mw = int(m.group(1))
                rows.append({
                    "home_team": home, "away_team": away,
                    "home_key": canon_team(home), "away_key": canon_team(away),
                    "home_goals": (None if hg is None else int(hg)),
                    "away_goals": (None if ag is None else int(ag)),
                    "kickoff_utc": ko_utc,
                    "kickoff_uk": ko_utc.tz_convert("Europe/London") if ko_utc is not None else pd.NaT,
                    "kick_date_uk": ko_utc.tz_convert("Europe/London").date() if ko_utc is not None else None,
                    "matchweek": mw,       # <-- official ROUND (1..38)
                    "status_short": st,    # e.g., NS, 1H, 2H, FT
                })
    out = pd.DataFrame(rows)
    if debug and not out.empty:
        print(f"[info] API-Football rows fetched: {len(out)} (UK: {out['kickoff_uk'].min()} .. {out['kickoff_uk'].max()})")
    return out

# ---------- Local fallback loaders ----------
def _load_results_csv(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=["home_team","away_team","home_key","away_key","home_goals","away_goals","kickoff_uk","kick_date_uk","matchweek"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        return None
    htc = pick("home_team","home","hometeam")
    atc = pick("away_team","away","awayteam")
    hgc = pick("home_goals","hg","fthg","home_score")
    agc = pick("away_goals","ag","ftag","away_score")
    kuc = pick("kickoff_uk","kickoff","datetime","date","kickoff_local")
    mwc = pick("matchweek","week","round")
    out = pd.DataFrame(columns=["home_team","away_team","home_key","away_key","home_goals","away_goals","kickoff_uk","kick_date_uk","matchweek"])
    if not all([htc, atc, hgc, agc, kuc]): return out
    out["home_team"]  = df[htc].astype(str)
    out["away_team"]  = df[atc].astype(str)
    out["home_key"]   = out["home_team"].map(canon_team)
    out["away_key"]   = out["away_team"].map(canon_team)
    out["home_goals"] = pd.to_numeric(df[hgc], errors="coerce")
    out["away_goals"] = pd.to_numeric(df[agc], errors="coerce")
    ku = pd.to_datetime(df[kuc], errors="coerce")
    if ku.dt.tz is None: ku = ku.dt.tz_localize("Europe/London")
    else: ku = ku.dt.tz_convert("Europe/London")
    out["kickoff_uk"] = ku
    out["kick_date_uk"] = out["kickoff_uk"].dt.date
    out["matchweek"] = pd.to_numeric(df[mwc], errors="coerce") if mwc else None  # may be round if present
    out = out.dropna(subset=["home_goals","away_goals"], how="any")
    return out

def _load_results_json(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=["home_team","away_team","home_key","away_key","home_goals","away_goals","kickoff_uk","kick_date_uk","matchweek"])
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", data)
    df = pd.json_normalize(rows)
    tmp = Path(str(path) + ".tmp.csv")
    df.to_csv(tmp, index=False)
    out = _load_results_csv(tmp)
    try: tmp.unlink()
    except Exception: pass
    return out

# ---------- Market grading ----------
def label_from_score(hg: int, ag: int) -> dict:
    res_1x2 = "H" if hg > ag else "A" if ag > hg else "D"
    total = hg + ag
    return {"res_1x2": res_1x2, "btts_yes": (hg > 0 and ag > 0), "total": total}

def _normalize_selection_1x2(sel: Optional[str]) -> Optional[str]:
    if sel is None: return None
    s = str(sel).strip().lower()
    m = {"h":"H","home":"H","home win":"H","1":"H",
         "a":"A","away":"A","away win":"A","2":"A",
         "d":"D","draw":"D","x":"D"}
    return m.get(s)

def _normalize_selection_btts(sel: Optional[str]) -> Optional[bool]:
    if sel is None: return None
    s = str(sel).strip().lower()
    m = {"yes": True, "y": True, "1": True, "both":"yes",
         "no": False, "n": False, "0": False}
    v = m.get(s, None)
    if v == "yes": return True
    return v

def _parse_totals_market(market: str) -> Optional[float]:
    if not market: return None
    m = re.search(r"totals?\s*([0-9]+(?:\.[0-9]+)?)", market, flags=re.I)
    return float(m.group(1)) if m else None

def grade_single(pick: dict, final_labels: dict) -> dict:
    labs = {"res_1x2": None, "btts_yes": None, "total": None}
    won = None
    market = (pick.get("market") or "").strip()
    sel    = pick.get("selection")

    hg_raw = final_labels.get("home_goals")
    ag_raw = final_labels.get("away_goals")
    if hg_raw is not None and ag_raw is not None and not (pd.isna(hg_raw) or pd.isna(ag_raw)):
        hg = int(hg_raw); ag = int(ag_raw)
        labs = label_from_score(hg, ag)
        if market.upper() in ("1X2","MATCH_ODDS"):
            sel_norm = _normalize_selection_1x2(sel)
            if sel_norm is not None:
                won = (sel_norm == labs["res_1x2"]) 
        elif "BTTS" in market.upper():
            want_yes = _normalize_selection_btts(sel)
            if want_yes is not None:
                won = (want_yes == labs["btts_yes"]) 
        elif "TOTAL" in market.upper():
            line = _parse_totals_market(market)
            if line is not None and sel:
                s = str(sel).strip().lower()
                if s.startswith("under"):
                    won = ((hg + ag) < line)
                elif s.startswith("over"):
                    won = ((hg + ag) > line)

    profit = (float(pick.get("best_odds") or 0.0) - 1.0) if won else (-1.0 if won is False else None)
    return {**pick, **final_labels, **labs, "won": won, "profit": profit}

# ---------- Round inference helper (from API-Football) ----------
def infer_round_from_apifootball_for_picks(api_df: pd.DataFrame, picks_df: pd.DataFrame) -> Optional[int]:
    """
    Try to infer a single round number for the current picks window using API-Football rows.
    We match by (home_key, away_key) within ±2 days, then take the mode of 'matchweek'.
    Returns an integer round (1..38) or None.
    """
    if api_df is None or api_df.empty or picks_df is None or picks_df.empty:
        return None

    tmp_api = api_df.copy()
    tmp_api["kick_date_uk"] = pd.to_datetime(tmp_api["kick_date_uk"], errors="coerce").dt.date
    tmp_p = picks_df.copy()
    tmp_p["kick_date_uk"] = pd.to_datetime(tmp_p["kickoff_uk_dt"], errors="coerce").dt.tz_convert("Europe/London").dt.date

    rounds = []
    for idx, r in tmp_p.iterrows():
        hk, ak, d = r.get("home_key"), r.get("away_key"), r.get("kick_date_uk")
        if pd.isna(d) or not hk or not ak:
            continue
        d_ts = pd.to_datetime(d)
        cand = tmp_api[
            (tmp_api["home_key"].isin([hk, ak])) & (tmp_api["away_key"].isin([hk, ak])) &
            (pd.to_datetime(tmp_api["kick_date_uk"]).between(d_ts - pd.Timedelta(days=2),
                                                             d_ts + pd.Timedelta(days=2)))
        ]
        if cand.empty:
            continue
        exact = cand[(cand["home_key"] == hk) & (cand["away_key"] == ak)]
        use = exact if not exact.empty else cand
        mw = pd.to_numeric(use["matchweek"], errors="coerce").dropna()
        if not mw.empty:
            rounds.extend(mw.astype(int).tolist())

    if not rounds:
        return None
    return int(pd.Series(rounds).mode().iloc[0])

# ---------- Main ----------
def main():
    args = parse_args()
    api_key = args.api_football_key or os.getenv("API_FOOTBALL_KEY")

    # --- robust picks path resolution (supports round/mw-based filenames) ---
    picks_mw     = getattr(args, "picks_mw", None)
    picks_round  = picks_mw if picks_mw is not None else getattr(args, "picks_round", None)
    picks_season = getattr(args, "picks_season", None)

    if picks_round is not None:
        picks_path = resolve_picks_path_by_round(
            round_num=picks_round,
            season_year=picks_season,
            base_dir=DEFAULT_PICKS_DIR,
            strict=False,
        )
    else:
        picks_path = resolve_picks_path(args.picks)

    # Picks
    picks = load_picks_json(picks_path)
    singles = picks.get("singles") or picks.get("picks") or []
    accas   = picks.get("accas") or []

    if args.seasons:
        seasons = sorted({int(s.strip()) for s in args.seasons.split(",") if s.strip()})
    else:
        seasons = infer_seasons_from_picks(singles)

    # Build picks table
    s = pd.DataFrame(singles)
    if s.empty:
        print("[warn] No singles in picks JSON; accas may still be graded.")
        s = pd.DataFrame(columns=["home_team","away_team","kickoff_uk","kickoff_utc","market","selection","best_odds","model_prob"])
    s["home_key"] = s.get("home_team", pd.Series(dtype="object")).map(canon_team)
    s["away_key"] = s.get("away_team", pd.Series(dtype="object")).map(canon_team)
    uk  = s["kickoff_uk"] if "kickoff_uk" in s.columns else pd.Series([pd.NA]*len(s))
    utc = s["kickoff_utc"] if "kickoff_utc" in s.columns else pd.Series([pd.NA]*len(s))
    kick_raw = uk.fillna(utc)
    s["kickoff_uk_dt"] = pd.to_datetime(kick_raw, utc=True, errors="coerce").dt.tz_convert("Europe/London")
    s["kick_date_uk"]  = s["kickoff_uk_dt"].dt.date

    # Date window for providers
    if s["kickoff_uk_dt"].notna().any():
        dmin = s["kickoff_uk_dt"].min() - pd.Timedelta(days=2)
        dmax = s["kickoff_uk_dt"].max() + pd.Timedelta(days=2)
    else:
        now = pd.Timestamp.now(tz="Europe/London")
        dmin, dmax = now - pd.Timedelta(days=7), now + pd.Timedelta(days=7)

    # ---------- Providers (order: ESPN → Football-Data → Understat → API-Football → Local) ----------
    sources: List[Tuple[str, pd.DataFrame]] = []
    if not args.no_espn:
        espn_df = fetch_espn_results(dmin, dmax, debug=args.debug)
        if args.debug: print(f"[debug] ESPN: {'no rows' if espn_df.empty else f'{len(espn_df)} rows'}")
        sources.append(("ESPN", espn_df))

    fd_df = fetch_football_data_results(seasons, dmin, dmax, debug=args.debug)
    if args.debug: print(f"[debug] Football-Data: {'no rows' if fd_df.empty else f'{len(fd_df)} rows'}")
    sources.append(("Football-Data", fd_df))

    us_df = asyncio.run(fetch_understat_matches(args.league, seasons, debug=args.debug))
    if args.debug: print(f"[debug] Understat: {'no rows' if us_df.empty else f'{len(us_df)} rows'}")
    sources.append(("Understat", us_df))

    api_df_for_rounds = None
    if args.api_football:
        af_df = fetch_api_football_results(args.api_football_league_id, seasons, dmin, dmax, api_key, debug=args.debug)
        if args.debug: print(f"[debug] API-Football: {'no rows' if af_df.empty else f'{len(af_df)} rows'}")
        sources.append(("API-Football", af_df))
        api_df_for_rounds = af_df.copy()

    if args.fallback_results:
        p = args.fallback_results
        fb_df = _load_results_csv(p) if p.suffix.lower() == ".csv" else _load_results_json(p)
        if args.debug: print(f"[debug] Local fallback: {'no rows' if fb_df.empty else f'{len(fb_df)} rows'}")
        sources.append(("Local", fb_df))

    # Prepare working join frame
    j = s.copy()
    for col in ["home_goals","away_goals","kickoff_uk_res","round"]:
        if col not in j.columns:
            j[col] = pd.NA

    # Matching passes applied per-source until filled
    for src_name, src_df in sources:
        if src_df is None or src_df.empty:
            continue
        src = src_df.copy()
        src["kick_date_uk"] = pd.to_datetime(src["kick_date_uk"], errors="coerce").dt.date
        src["pair_key"] = src.apply(lambda r: _build_pair_key(r.get("home_key"), r.get("away_key")), axis=1)

        before = j["home_goals"].notna().sum()

        # Pass 1: exact
        mask = j["home_goals"].isna()
        if mask.any():
            left = j.loc[mask, ["home_key","away_key","kick_date_uk"]].copy()
            left["__idx"] = left.index
            cols = ["home_key","away_key","kick_date_uk","home_goals","away_goals","kickoff_uk"]
            if "matchweek" in src.columns: cols.append("matchweek")
            right = src[cols].copy()
            m1 = left.merge(right, on=["home_key","away_key","kick_date_uk"], how="left").set_index("__idx")
            fill_idx = m1.index
            if len(fill_idx):
                j.loc[fill_idx, "home_goals"] = j.loc[fill_idx, "home_goals"].combine_first(m1["home_goals"])
                j.loc[fill_idx, "away_goals"] = j.loc[fill_idx, "away_goals"].combine_first(m1["away_goals"])
                j.loc[fill_idx, "kickoff_uk_res"] = j.loc[fill_idx, "kickoff_uk_res"].combine_first(m1["kickoff_uk"])
                if "matchweek" in m1.columns:
                    j.loc[fill_idx, "round"] = j.loc[fill_idx, "round"].fillna(m1["matchweek"])

        # Pass 2: swapped
        mask = j["home_goals"].isna()
        if mask.any():
            sw = src.rename(columns={
                "home_key":"away_key","away_key":"home_key",
                "home_team":"away_team_src","away_team":"home_team_src",
                "home_goals":"away_goals","away_goals":"home_goals"
            })
            left = j.loc[mask, ["home_key","away_key","kick_date_uk"]].copy()
            left["__idx"] = left.index
            cols = ["home_key","away_key","kick_date_uk","home_goals","away_goals","kickoff_uk"]
            if "matchweek" in sw.columns: cols.append("matchweek")
            right = sw[cols].copy()
            m2 = left.merge(right, on=["home_key","away_key","kick_date_uk"], how="left").set_index("__idx")
            fill_idx = m2.index
            if len(fill_idx):
                j.loc[fill_idx, "home_goals"] = j.loc[fill_idx, "home_goals"].combine_first(m2["home_goals"])
                j.loc[fill_idx, "away_goals"] = j.loc[fill_idx, "away_goals"].combine_first(m2["away_goals"])
                j.loc[fill_idx, "kickoff_uk_res"] = j.loc[fill_idx, "kickoff_uk_res"].combine_first(m2["kickoff_uk"])
                if "matchweek" in m2.columns:
                    j.loc[fill_idx, "round"] = j.loc[fill_idx, "round"].fillna(m2["matchweek"])

        # Pass 3: unordered ±2 days, nearest kickoff
        mask_idx = j.index[j["home_goals"].isna()]
        if len(mask_idx):
            src_tmp = src.copy()
            src_tmp["kick_ts"] = pd.to_datetime(src_tmp["kickoff_uk"], errors="coerce")
            for idx in mask_idx:
                hk = str(j.at[idx, "home_key"] or ""); ak = str(j.at[idx, "away_key"] or ""); d  = j.at[idx, "kick_date_uk"]
                if not hk or not ak or pd.isna(d): continue
                pair = _build_pair_key(hk, ak)
                d_ts = pd.to_datetime(d)
                cand = src_tmp[(src_tmp["pair_key"] == pair) &
                               (pd.to_datetime(src_tmp["kick_date_uk"]).between(d_ts - pd.Timedelta(days=2),
                                                                                d_ts + pd.Timedelta(days=2)))]
                if cand.empty: continue
                ref_dt = pd.to_datetime(j.at[idx, "kickoff_uk"] or j.at[idx, "kickoff_uk_dt"], errors="coerce")
                if pd.isna(ref_dt): ref_dt = d_ts
                best = _nearest_by_dt(cand, ref_dt)
                if best.get("home_key") == hk and best.get("away_key") == ak:
                    j.at[idx, "home_goals"] = best.get("home_goals")
                    j.at[idx, "away_goals"] = best.get("away_goals")
                else:
                    j.at[idx, "home_goals"] = best.get("away_goals")
                    j.at[idx, "away_goals"] = best.get("home_goals")
                j.at[idx, "kickoff_uk_res"] = best.get("kickoff_uk")
                if "matchweek" in best.index:
                    j.at[idx, "round"] = j.at[idx, "round"] if pd.notna(j.at[idx, "round"]) else best.get("matchweek")

        # Pass 4: fuzzy titles within ±2 days, nearest kickoff
        mask_idx = j.index[j["home_goals"].isna()]
        if len(mask_idx):
            src_tmp = src.copy()
            src_tmp["kick_ts"] = pd.to_datetime(src_tmp["kickoff_uk"], errors="coerce")
            for idx in mask_idx:
                ph = str(j.at[idx, "home_team"] or ""); pa = str(j.at[idx, "away_team"] or "")
                hk = str(j.at[idx, "home_key"] or "");   ak = str(j.at[idx, "away_key"] or "")
                d  = j.at[idx, "kick_date_uk"]
                if pd.isna(d): continue
                d_ts = pd.to_datetime(d)
                cand = src_tmp[pd.to_datetime(src_tmp["kick_date_uk"]).between(d_ts - pd.Timedelta(days=2),
                                                                               d_ts + pd.Timedelta(days=2))].copy()
                if cand.empty: continue
                cand["score"] = (
                    cand["home_key"].apply(lambda x: _ratio(x, hk)) * 0.6 +
                    cand["away_key"].apply(lambda x: _ratio(x, ak)) * 0.6 +
                    cand["home_team"].apply(lambda x: _ratio(str(x), ph)) * 0.2 +
                    cand["away_team"].apply(lambda x: _ratio(str(x), pa)) * 0.2
                )
                top = cand.sort_values(["score"], ascending=False).head(5)
                best = top.iloc[0]
                if best["score"] < 0.80:
                    continue
                ref_dt = pd.to_datetime(j.at[idx, "kickoff_uk"] or j.at[idx, "kickoff_uk_dt"], errors="coerce")
                if pd.isna(ref_dt): ref_dt = d_ts
                top["dt"] = pd.to_datetime(top["kickoff_uk"], errors="coerce")
                top["abs_diff"] = (top["dt"] - ref_dt).abs()
                best = top.sort_values(["abs_diff"]).iloc[0]
                if _ratio(best.get("home_key",""), hk) >= _ratio(best.get("away_key",""), hk):
                    j.at[idx, "home_goals"] = best.get("home_goals")
                    j.at[idx, "away_goals"] = best.get("away_goals")
                else:
                    j.at[idx, "home_goals"] = best.get("away_goals")
                    j.at[idx, "away_goals"] = best.get("home_goals")
                j.at[idx, "kickoff_uk_res"] = best.get("kickoff_uk")
                if "matchweek" in best.index:
                    j.at[idx, "round"] = j.at[idx, "round"] if pd.notna(j.at[idx, "round"]) else best.get("matchweek")

        after = j["home_goals"].notna().sum()
        if args.debug:
            print(f"[debug] {src_name}: matched +{after - before}, now matched {after}/{len(j)}")
        if after == len(j):
            break  # all done

    # Build final labels
    finals = []
    for _, r in j.iterrows():
        hg = r.get("home_goals"); ag = r.get("away_goals")
        finals.append({
            "home_goals": int(hg) if pd.notna(hg) else None,
            "away_goals": int(ag) if pd.notna(ag) else None,
            "round": (int(r.get("round")) if pd.notna(r.get("round")) else None),
        })
    j_final = pd.DataFrame(finals, index=j.index).astype(object)

    # Debug CSV of unmatched (always write something in --debug)
    if args.debug:
        unmatched_idx = j_final.index[j_final["home_goals"].isna()].tolist()
        rows = []
        if len(unmatched_idx):
            for idx in unmatched_idx:
                rows.append({
                    "pick_home": s.at[idx, "home_team"] if "home_team" in s.columns else None,
                    "pick_away": s.at[idx, "away_team"] if "away_team" in s.columns else None,
                    "pick_date": s.at[idx, "kick_date_uk"] if "kick_date_uk" in s.columns else None,
                    "reason": "no result matched"
                })
        dbg = pd.DataFrame(rows)
        out_dbg = (DEFAULT_OUT_DIR / "debug_unmatched.csv")
        out_dbg.parent.mkdir(parents=True, exist_ok=True)
        dbg.to_csv(out_dbg, index=False)
        print(f"[debug] wrote {out_dbg} ({len(dbg)} rows)")

    # Grade singles
    graded_singles = []
    for idx in j_final.index:
        pick = s.loc[idx].to_dict()
        fin  = j_final.loc[idx].to_dict()
        graded_singles.append(grade_single(pick, fin))

    # Grade accas
    singles_by_key = {(d.get("home_team"), d.get("away_team"), d.get("kickoff_uk")): d for d in graded_singles}
    graded_accas = []
    for a in accas:
        legs = a.get("legs", [])
        glegs = []
        any_unknown = False
        all_win = True
        prod_odds = 1.0
        prod_prob = 1.0
        for l in legs:
            key = (l.get("home_team"), l.get("away_team"), l.get("kickoff_uk"))
            gs = singles_by_key.get(key)
            if gs is None:
                key2_h = canon_team(l.get("home_team")); key2_a = canon_team(l.get("away_team"))
                for ss in graded_singles:
                    if canon_team(ss.get("home_team")) == key2_h and canon_team(ss.get("away_team")) == key2_a and ss.get("kickoff_uk") == l.get("kickoff_uk"):
                        gs = ss; break
            if gs is None or gs.get("won") is None:
                any_unknown = True
            else:
                all_win = all_win and bool(gs["won"])
            glegs.append(gs if gs is not None else {**l, "won": None})
            try: prod_odds *= float((gs or l).get("best_odds") or (gs or l).get("odds") or 0.0)
            except Exception: pass
            try:
                prob_i = float((gs or l).get("model_prob") or (gs or l).get("p") or 0.0)
                prod_prob *= prob_i if prob_i > 0 else 1.0
            except Exception: pass

        acca_won = (None if any_unknown else all_win) if len(glegs) == len(legs) else None
        acca_return = (prod_odds if acca_won else (0.0 if acca_won is False else None))
        acca_profit = (acca_return - 1.0) if acca_return is not None else None

        graded_accas.append({
            "legs": glegs,
            "legs_count": len(glegs),
            "acca_odds": prod_odds if prod_odds > 0 else None,
            "acca_prob_model": prod_prob if prod_prob > 0 else None,
            "won": acca_won,
            "return": acca_return,
            "profit": acca_profit,
        })

    # -------- Summaries --------
    gs_df = pd.DataFrame(graded_singles)
    graded_mask_s = gs_df["won"].notna() if "won" in gs_df else pd.Series([], dtype="boolean")
    n_s_total = len(gs_df)
    gs_df_g = gs_df.loc[graded_mask_s] if n_s_total else gs_df
    n_s = len(gs_df_g)
    n_sw = int(gs_df_g["won"].sum()) if n_s else 0
    pnl_s = float(pd.to_numeric(gs_df_g["profit"], errors="coerce").fillna(0).sum()) if n_s else 0.0
    roi_s = pnl_s / n_s if n_s else None
    hit_s = (n_sw / n_s) if n_s else None

    ga_df_all = pd.DataFrame(graded_accas)
    graded_mask_a = ga_df_all["won"].notna() if not ga_df_all.empty else pd.Series([], dtype="boolean")
    ga_df = ga_df_all.loc[graded_mask_a] if not ga_df_all.empty else ga_df_all
    n_a = len(ga_df); n_aw = int(ga_df["won"].sum()) if n_a else 0
    pnl_a = float(pd.to_numeric(ga_df["profit"], errors="coerce").fillna(0).sum()) if n_a else 0.0
    roi_a = pnl_a / n_a if n_a else None
    hit_a = (n_aw / n_a) if n_a else None

    # ---------- Filename logic (ALWAYS 'mw' NAMING, USING ROUND NUMBERS) ----------
    def _round_span_from_frame(frame: pd.DataFrame) -> Optional[Tuple[int,int,int]]:
        if frame is None or frame.empty or "round" not in frame.columns:
            return None
        rr = pd.to_numeric(frame["round"], errors="coerce").dropna().astype(int)
        if rr.empty:
            return None
        return int(rr.mode().iloc[0]), int(rr.min()), int(rr.max())

    # Season from picks (dominant)
    season_for_name = None
    if len(s) and s["kickoff_uk_dt"].notna().any():
        dts = s["kickoff_uk_dt"].dt.tz_convert("Europe/London").dt.date
        seas = pd.Series([(d.year if d.month >= 8 else d.year - 1) for d in dts])
        season_for_name = int(seas.mode().iloc[0]) if not seas.empty else None
    else:
        seasons_from_picks = infer_seasons_from_picks(singles)
        season_for_name = seasons_from_picks[-1] if seasons_from_picks else None
    if season_for_name is None:
        season_for_name = _season_year_now_uk()

    # Prefer ROUND recovered from matches (or forced)
    forced_round = getattr(args, "force_round", None)
    round_mode = round_min = round_max = None
    if forced_round is not None:
        round_mode = round_min = round_max = int(forced_round)
    else:
        span = _round_span_from_frame(j_final)
        if span:
            round_mode, round_min, round_max = span
        elif api_df_for_rounds is not None:
            r_try = infer_round_from_apifootball_for_picks(api_df_for_rounds, s)
            if r_try is not None:
                round_mode = round_min = round_max = int(r_try)

    # Build final output path (prefer CLI/file MW if present)
    if args.out is not None:
        out_path = args.out
    else:
        if 'picks_round' in locals() and picks_round is not None:
            fname = f"epl_{season_for_name}_mw{int(picks_round):02d}_pick_analysis.json"
        elif 'picks_mw' in locals() and picks_mw is not None:
            fname = f"epl_{season_for_name}_mw{int(picks_mw):02d}_pick_analysis.json"
        elif round_mode is not None:
            if round_min is not None and round_max is not None and round_min != round_max:
                fname = f"epl_{season_for_name}_mw{round_min:02d}-{round_max:02d}_pick_analysis.json"
            else:
                fname = f"epl_{season_for_name}_mw{round_mode:02d}_pick_analysis.json"
        else:
            fname = f"epl_{season_for_name}_mw00_pick_analysis.json"
        out_path = DEFAULT_OUT_DIR / fname

    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": {
            "picks": str(resolve_picks_path(picks_path)),
            "espn": {"enabled": not args.no_espn},
            "football_data": {"enabled": True},
            "understat": {"league": args.league, "seasons": seasons},
            "api_football": {"enabled": args.api_football, "league_id": args.api_football_league_id, "used_key": bool(api_key)},
            "fallback_results": str(args.fallback_results) if args.fallback_results else None
        },
        "summary": {
            "singles": {
                "count_total": n_s_total,
                "count_graded": n_s,
                "wins": n_sw,
                "hit_rate": hit_s,
                "profit": pnl_s,
                "roi_per_bet": roi_s
            },
            "accas":   {
                "count_total": len(ga_df_all),
                "count_graded": n_a,
                "wins": n_aw,
                "hit_rate": hit_a,
                "profit": pnl_a,
                "roi_per_bet": roi_a
            },
            "n_with_round": int(pd.to_numeric(j_final.get("round"), errors="coerce").notna().sum()) if "round" in j_final.columns else 0
        },
        "singles": graded_singles,
        "accas": graded_accas,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    unmatched = int((pd.Series([d.get("won") for d in graded_singles]).isna()).sum())
    print(f"[done] singles total={n_s_total} graded={n_s} (wins={n_sw}, unmatched={unmatched}), "
          f"accas total={len(ga_df_all)} graded={n_a} (wins={n_aw}) → {out_path}")
    if n_s:
        print(f"Singles ROI: {roi_s:+.3f}  |  Accas ROI: {roi_a:+.3f}" if n_a else f"Singles ROI: {roi_s:+.3f}")

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
