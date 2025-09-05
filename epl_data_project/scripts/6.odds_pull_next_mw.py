#!/usr/bin/env python3
"""
fetch_epl_odds_next_matchweek.py

Fetch EPL odds (UK region) for the **NEXT** Premier League matchweek:
- Match Result: h2h (2/3-way depending on book) + h2h_3_way
- Totals (Over/Under)
- BTTS (Yes/No)
- Correct Score
- Asian Handicap (spreads)

Matchweek detection order (no API keys):
  1) FPL fixtures API (event == GW)
  2) ESPN scoreboard (round/week number)
  3) Fallback to 7-day window from now

Writes (non-overwriting):
  ../data/raw/odds/epl_odds_<SEASONSTART>_MWxx.parquet
  or (if MW can't be inferred): epl_odds_<SEASONSTART>_YYYYMMDD-YYYYMMDD.parquet

Dependencies:
  pip install requests pandas pyarrow
"""
from __future__ import annotations

import os
import time
import argparse
import datetime as dt
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import pandas as pd
from pathlib import Path

# --------- CONFIG (edit these) ---------
API_KEY = "08062ccde5beb16ce04ae49a237472ef"   # your The Odds API key
SPORT = "soccer_epl"
REGION = "uk"               # UK bookmakers
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"
DEFAULT_OUT = "../data/raw/odds"               # default to a directory
SLEEP_BETWEEN_EVENT_CALLS = 0.25
FALLBACK_DAYS = 7
# --------------------------------------

API_HOST = "https://api.the-odds-api.com/v4"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

# ----------------- time helpers -----------------
def iso_utc(dt_obj: dt.datetime) -> str:
    return dt_obj.replace(microsecond=0, tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def to_uk(ts):
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if isinstance(t, pd.Series):
        return t.dt.tz_convert("Europe/London")
    return t.tz_convert("Europe/London")

def season_start_year_from_date(d_uk: pd.Timestamp) -> int:
    """Premier League season starts in Aug. Aug–Dec => same year; Jan–May => previous year."""
    return int(d_uk.year if d_uk.month >= 8 else d_uk.year - 1)

# ----------------- market normalization -----------------
def _canon_market_key(mkey: str) -> str:
    mk = (mkey or "").strip().lower()
    if mk in {"h2h", "match_winner", "match_result"}: return "h2h"
    if mk in {"h2h_3_way", "3_way", "1x2"}:          return "h2h_3_way"
    if mk in {"totals", "over_under", "over/under"}: return "totals"
    if mk in {"btts", "both_teams_to_score"}:        return "btts"
    if mk in {"correct_score", "correct score", "cs"}: return "correct_score"
    if mk in {"spreads", "asian_handicap", "asian handicap", "asian", "asian_lines", "handicap", "ah"}:
        return "asian_handicap"
    return mk

def _norm_h2h_outcome(name: str, home_team: str, away_team: str) -> str:
    nm = (name or "").strip().lower()
    if nm == "draw": return "draw"
    ht = (home_team or "").strip().lower()
    at = (away_team or "").strip().lower()
    if ht and (nm == ht or ht in nm): return "home"
    if at and (nm == at or at in nm): return "away"
    return "unknown"

def _norm_btts_outcome(name: str) -> str:
    nm = (name or "").strip().lower()
    if nm == "yes": return "yes"
    if nm == "no":  return "no"
    return "unknown"

def _norm_totals_outcome(name: str) -> str:
    nm = (name or "").strip().lower()
    if nm.startswith("over"):  return "over"
    if nm.startswith("under"): return "under"
    return "unknown"

def _parse_cs(name: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract 'H-A' like '2-1', '1 : 1', etc."""
    if not name: return None, None
    s = str(name)
    m = pd.Series([s]).str.extract(r"(\d+)\D+(\d+)", expand=True)
    if m.isna().any(axis=None): return None, None
    return int(m.iloc[0,0]), int(m.iloc[0,1])

# ----------------- odds api -----------------
def fetch_featured_odds(api_key: str, start_iso: str, end_iso: str) -> List[dict]:
    """Featured markets: h2h, totals, spreads (to cover Asian Handicap)."""
    url = f"{API_HOST}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": "h2h,totals,spreads",
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
        "commenceTimeFrom": start_iso,
        "commenceTimeTo": end_iso,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    for k in ("x-requests-remaining", "x-requests-used", "x-requests-last"):
        if k in r.headers:
            print(f"[quota] {k}: {r.headers[k]}")
    return r.json()

def fetch_event_odds(api_key: str, event_id: str, markets: str) -> Optional[dict]:
    """Extra markets per event: btts, h2h_3_way, correct_score (and spreads again if needed)."""
    url = f"{API_HOST}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def flatten_event(event: dict, include_keys: Optional[Iterable[str]] = None) -> List[Dict]:
    """Flatten one event into rows (standardizing market keys)."""
    rows: List[Dict] = []
    match_id = event.get("id")
    commence_time = event.get("commence_time")
    home_team = event.get("home_team")
    away_team = event.get("away_team")

    for bk in (event.get("bookmakers") or []):
        bk_key = bk.get("key")
        bk_title = bk.get("title")
        bk_last = bk.get("last_update")
        for m in (bk.get("markets") or []):
            raw_key = (m.get("key") or "").strip()
            mkey = _canon_market_key(raw_key)
            if include_keys and mkey not in include_keys:
                continue
            last_update = m.get("last_update") or bk_last
            for oc in (m.get("outcomes") or []):
                outcome_name = oc.get("name")
                price = oc.get("price")
                point = oc.get("point")
                outcome = "unknown"
                cs_h = cs_a = None

                if mkey in ("h2h", "h2h_3_way"):
                    outcome = _norm_h2h_outcome(outcome_name, home_team, away_team)
                elif mkey == "btts":
                    outcome = _norm_btts_outcome(outcome_name)
                elif mkey == "totals":
                    outcome = _norm_totals_outcome(outcome_name)
                elif mkey == "asian_handicap":
                    # The Odds API typically uses outcomes 'home'/'away' and 'point' is the handicap for that side
                    outcome = _norm_h2h_outcome(outcome_name, home_team, away_team)
                elif mkey == "correct_score":
                    cs_h, cs_a = _parse_cs(outcome_name)

                rows.append({
                    "match_id": match_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker_key": bk_key,
                    "bookmaker_title": bk_title,
                    "market": mkey,                      # standardized key
                    "outcome": outcome,                  # standardized outcome for that market
                    "outcome_name": outcome_name,        # original book text (e.g., "Over 2.5", "1 - 0")
                    "price": price,
                    "point": point,                      # totals & asian handicap lines
                    "cs_home": cs_h,                     # for correct_score
                    "cs_away": cs_a,                     # for correct_score
                    "last_update": last_update,
                    "region": REGION,
                    "odds_format": ODDS_FORMAT,
                })
    return rows

# ----------------- FPL / ESPN matchweek inference (no keys) -----------------
def find_next_matchweek_window_via_fpl(now_uk: pd.Timestamp) -> Optional[Tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Use FPL fixtures: earliest event (GW) with kickoff_time >= now."""
    try:
        r = requests.get(FPL_FIXTURES, timeout=20)
        r.raise_for_status()
        fixtures = r.json()
        if not isinstance(fixtures, list):
            return None
    except Exception:
        return None

    rows = []
    for fx in fixtures:
        ev = fx.get("event")
        ko = fx.get("kickoff_time")
        if ev is None or not ko:
            continue
        ko_uk = to_uk(pd.Timestamp(ko))
        if ko_uk >= now_uk:
            rows.append((int(ev), ko_uk))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["event", "ko_uk"])
    next_ev = int(df["event"].min())
    in_ev = df[df["event"] == next_ev]["ko_uk"]
    return next_ev, in_ev.min(), in_ev.max()

def _espn_fetch_day(date_str: str) -> Optional[dict]:
    for d in (date_str, f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"):
        try:
            r = requests.get(ESPN_SCOREBOARD, params={"dates": d}, timeout=20)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            continue
    return None

def _extract_mw_from_competition(comp: dict) -> Optional[int]:
    for path in [("round", "number"), ("week", "number")]:
        d = comp; ok = True
        for k in path:
            if not isinstance(d, dict) or k not in d:
                ok = False; break
            d = d[k]
        if ok:
            try: return int(d)
            except Exception: pass
    for path in [("round", "text"), ("round", "name"), ("week", "text")]:
        d = comp
        for k in path:
            if isinstance(d, dict) and k in d: d = d[k]
            else: d = None; break
        if isinstance(d, str):
            m = pd.Series([d]).str.extract(r"(\d+)").iloc[0,0]
            if pd.notna(m): return int(m)
    return None

def find_next_matchweek_window_via_espn(now_uk: pd.Timestamp, horizon_days: int = 14) -> Optional[Tuple[int, pd.Timestamp, pd.Timestamp]]:
    dmin = now_uk.normalize()
    dmax = dmin + pd.Timedelta(days=horizon_days)
    mw_to_times: Dict[int, List[pd.Timestamp]] = {}
    for day in pd.date_range(dmin.date(), dmax.date(), freq="D"):
        ds = day.strftime("%Y%m%d")
        data = _espn_fetch_day(ds)
        if not data: continue
        events = data.get("events", []) or []
        for ev in events:
            comps = ev.get("competitions", []) or []
            if not comps: continue
            comp = comps[0]
            ev_date = pd.to_datetime(ev.get("date") or comp.get("date"), utc=True, errors="coerce")
            if ev_date is None or pd.isna(ev_date): continue
            ev_uk = to_uk(ev_date)
            if ev_uk < now_uk: continue
            mw = _extract_mw_from_competition(comp)
            if mw is None: continue
            mw_to_times.setdefault(int(mw), []).append(ev_uk)
    if not mw_to_times:
        return None
    next_mw = min(mw_to_times.keys())
    times = pd.Series(mw_to_times[next_mw])
    return next_mw, times.min(), times.max()

# ----------------- output filename builder -----------------
def build_output_path_with_mw(out_arg: str | Path,
                              dt_min_uk: pd.Timestamp,
                              dt_max_uk: pd.Timestamp,
                              mw_numbers: List[int],
                              prefix: str = "epl_odds") -> Path:
    """
    If `out_arg` is a directory, create epl_odds_<season>_<slug>.parquet under it.
    If `out_arg` is a file, insert _<season>_<slug> before extension.
    Slug = MWxx or MWxx-MWyy; fallback: YYYYMMDD-YYYYMMDD.
    """
    p = Path(out_arg)
    season = season_start_year_from_date(dt_min_uk)
    if mw_numbers:
        slug = f"MW{mw_numbers[0]:02d}" if len(mw_numbers) == 1 else f"MW{mw_numbers[0]:02d}-MW{mw_numbers[-1]:02d}"
    else:
        slug = f"{dt_min_uk.strftime('%Y%m%d')}-{dt_max_uk.strftime('%Y%m%d')}"
    name_piece = f"{prefix}_{season}_{slug}"
    if p.suffix:
        return p.with_name(f"{p.stem}_{name_piece}{p.suffix}")
    else:
        return p / f"{name_piece}.parquet"

# ----------------- main collection (NEXT MW) -----------------
def collect_next_matchweek(api_key: str,
                           out_path: str | Path,
                           sleep_between_event_calls: float = 0.25,
                           fallback_days: int = FALLBACK_DAYS) -> Path:
    now = utc_now()
    now_uk = to_uk(pd.Timestamp(now))

    # 1) FPL inference
    fpl = find_next_matchweek_window_via_fpl(now_uk)
    if fpl:
        mw, start_uk, end_uk = fpl
        print(f"[info] Next MW via FPL: MW{mw:02d} | {start_uk} → {end_uk}")
        mw_numbers = [mw]
    else:
        # 2) ESPN inference
        espn = find_next_matchweek_window_via_espn(now_uk)
        if espn:
            mw, start_uk, end_uk = espn
            print(f"[info] Next MW via ESPN: MW{mw:02d} | {start_uk} → {end_uk}")
            mw_numbers = [mw]
        else:
            # 3) Fallback window (7 days)
            start_uk = now_uk.normalize()
            end_uk = start_uk + pd.Timedelta(days=fallback_days)
            print(f"[warn] Could not infer next MW; falling back to date window {start_uk} → {end_uk}")
            mw_numbers = []

    # Convert to ISO UTC bounds for odds API (pad a bit)
    start_utc = start_uk.tz_convert("UTC").to_pydatetime().replace(tzinfo=dt.timezone.utc) - dt.timedelta(hours=6)
    end_utc = end_uk.tz_convert("UTC").to_pydatetime().replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=6)
    start_iso = iso_utc(start_utc)
    end_iso = iso_utc(end_utc)

    print(f"[info] fetching EPL featured odds (h2h, totals, asian_handicap) for {start_iso} → {end_iso} (region={REGION})")
    base_events = fetch_featured_odds(api_key, start_iso, end_iso)

    all_rows: List[Dict] = []
    for ev in base_events:
        # base markets
        all_rows.extend(flatten_event(ev))

        # extra markets: BTTS + 3-way + Correct Score
        ev_id = ev.get("id")
        if ev_id:
            try:
                extra = fetch_event_odds(
                    api_key,
                    ev_id,
                    markets="btts,h2h_3_way,correct_score"  # spreads already requested in base call
                )
            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", "?")
                print(f"[warn] extra markets {ev_id} HTTP {code}: {e}")
                extra = None
            if extra:
                # Only include the new markets from this per-event call
                all_rows.extend(flatten_event(extra, include_keys={"btts","h2h_3_way","correct_score"}))

        time.sleep(sleep_between_event_calls)

    # empty guard
    cols = ["match_id","commence_time","home_team","away_team","bookmaker_key","bookmaker_title",
            "market","outcome","outcome_name","price","point","cs_home","cs_away",
            "last_update","region","odds_format"]
    if not all_rows:
        print("[warn] no odds rows collected.")
        df = pd.DataFrame(columns=cols)
        final_out = build_output_path_with_mw(out_path, start_uk, end_uk, mw_numbers)
        Path(final_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(final_out, index=False)
        print(f"[done] wrote empty {final_out}")
        return final_out

    # build DF + guard to window + de-dup
    df = pd.DataFrame(all_rows)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["last_update"] = pd.to_datetime(df["last_update"], utc=True, errors="coerce")
    mask = (df["commence_time"] >= pd.Timestamp(start_iso)) & (df["commence_time"] <= pd.Timestamp(end_iso))
    df = df.loc[mask].reset_index(drop=True)

    # normalize/clean
    # de-duplicate identical rows (same match/book/market/outcome/point/price)
    dedup_cols = ["match_id","bookmaker_key","bookmaker_title","market","outcome","outcome_name","point","price"]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # actual UK window present in data (for file naming)
    dt_min_uk = to_uk(df["commence_time"]).min() if not df.empty else start_uk
    dt_max_uk = to_uk(df["commence_time"]).max() if not df.empty else end_uk

    final_out = build_output_path_with_mw(out_path, dt_min_uk, dt_max_uk, mw_numbers)
    Path(final_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(final_out, index=False)
    print(f"[done] wrote {final_out} with {len(df):,} rows across "
          f"{df['match_id'].nunique()} matches and {df['bookmaker_key'].nunique()} UK bookmakers.")
    return final_out

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Fetch EPL odds (UK) for the NEXT matchweek; filename includes Premier League Matchweek."
    )
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output path OR directory. If file, season & MW/date slug inserted before extension; "
                         "if directory, file is created as epl_odds_<season>_<slug>.parquet.")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_EVENT_CALLS, help="Delay between per-event calls (s)")
    ap.add_argument("--fallback-days", type=int, default=FALLBACK_DAYS, help="Days to fetch if MW inference fails")
    return ap.parse_args()

# ----------------- entrypoint -----------------
if __name__ == "__main__":
    args = parse_args()

    if not API_KEY or API_KEY == "PUT_YOUR_THEODDSAPI_KEY_HERE":
        raise SystemExit("ERROR: please put your API key into API_KEY at the top of this script.")

    final_path = collect_next_matchweek(
        api_key=API_KEY,
        out_path=args.out,
        sleep_between_event_calls=args.sleep,
        fallback_days=args.fallback_days,
    )
    print(f"[info] odds saved to {final_path}")
