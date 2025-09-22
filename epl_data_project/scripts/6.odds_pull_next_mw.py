#!/usr/bin/env python3
"""
6.odds_pull_next_mw.py — EPL odds for the NEXT matchweek (UK time)

Raw odds rows written to:
  ../data/raw/odds/epl_odds_<SEASONSTART>_MWnn.parquet

Always fetched in one bulk call:
  - h2h (match winner)
  - totals (over/under)
  - spreads (asian handicap)

Best-effort per-event (to avoid 422 in bulk):
  - btts
  - h2h_3_way

CLI:
  --out PATH_OR_DIR           default ../data/raw/odds
  --extras LIST               comma list of extra markets to attempt per-event (default btts,h2h_3_way)
  --fallback-days N           default 7
  --debug
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import pandas as pd
import requests

# ----------------- CONFIG -----------------
API_KEY = "08062ccde5beb16ce04ae49a237472ef"  # your key
SPORT = "soccer_epl"
REGION = "uk"
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"
DEFAULT_OUT = "../data/raw/odds"
FALLBACK_DAYS = 7

API_HOST = "https://api.the-odds-api.com/v4"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"

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
    return int(d_uk.year if d_uk.month >= 8 else d_uk.year - 1)

# ----------------- normalization -----------------
def canon_market_key(mkey: str) -> str:
    mk = (mkey or "").strip().lower()
    if mk in {"h2h", "match_winner", "match result"}: return "h2h"
    if mk in {"h2h_3_way", "3_way", "1x2"}:          return "h2h_3_way"
    if mk in {"totals", "over_under", "over/under"}: return "totals"
    if mk in {"spreads", "asian_handicap", "asian handicap", "handicap", "ah"}: return "spreads"
    if mk in {"btts", "both_teams_to_score"}:        return "btts"
    return mk

def norm_h2h_outcome(name: str, home_team: str, away_team: str) -> str:
    nm = (name or "").strip().lower()
    if nm == "draw":
        return "draw"
    ht = (home_team or "").strip().lower()
    at = (away_team or "").strip().lower()
    if ht and (nm == ht or ht in nm): return "home"
    if at and (nm == at or at in nm): return "away"
    return (name or "").strip()

def norm_totals_outcome(name: str) -> str:
    nm = (name or "").strip().lower()
    if nm.startswith("over"):  return "over"
    if nm.startswith("under"): return "under"
    return nm

def norm_btts_outcome(name: str) -> str:
    nm = (name or "").strip().lower()
    if nm in {"yes", "no"}: return nm
    return nm

# ----------------- Odds API -----------------
def fetch_bulk_odds(api_key: str, start_iso: str, end_iso: str, markets_csv: str) -> List[dict]:
    """Bulk odds call for multiple guaranteed markets; API may 422 if unsupported market is present."""
    url = f"{API_HOST}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": markets_csv,
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

def fetch_event_market(api_key: str, event_id: str, market: str) -> Optional[dict]:
    """Safely fetch ONE market for ONE event; return None on 404/422."""
    url = f"{API_HOST}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": market,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code in (404, 422):
        return None
    r.raise_for_status()
    return r.json()

def flatten_event(event: dict, include_keys: Optional[Iterable[str]] = None) -> List[Dict]:
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
            mkey_raw = (m.get("key") or "").strip()
            mkey = canon_market_key(mkey_raw)
            if include_keys and mkey not in include_keys:
                continue
            last_update = m.get("last_update") or bk_last
            for oc in (m.get("outcomes") or []):
                outcome_name = oc.get("name")
                price = oc.get("price")
                point = oc.get("point")
                out_norm = None
                if mkey in ("h2h", "h2h_3_way", "spreads"):
                    out_norm = norm_h2h_outcome(outcome_name, home_team, away_team)
                elif mkey == "totals":
                    out_norm = norm_totals_outcome(outcome_name)
                elif mkey == "btts":
                    out_norm = norm_btts_outcome(outcome_name)

                rows.append({
                    "match_id": match_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker_key": bk_key,
                    "bookmaker_title": bk_title,
                    "market_raw": mkey_raw,
                    "market": mkey,
                    "outcome": out_norm if out_norm else (outcome_name or "").strip(),
                    "outcome_name": outcome_name,
                    "price": price,
                    "point": point,
                    "last_update": last_update,
                    "region": REGION,
                    "odds_format": ODDS_FORMAT,
                })
    return rows

# ----------------- MW inference -----------------
def find_next_matchweek_window_via_fpl(now_uk: pd.Timestamp) -> Optional[Tuple[int, pd.Timestamp, pd.Timestamp]]:
    try:
        r = requests.get(FPL_FIXTURES, timeout=20)
        r.raise_for_status()
        fixtures = r.json()
        if not isinstance(fixtures, list): return None
    except Exception:
        return None

    rows = []
    for fx in fixtures:
        ev = fx.get("event")
        ko = fx.get("kickoff_time")
        if ev is None or not ko: continue
        ko_uk = to_uk(pd.Timestamp(ko))
        if ko_uk >= now_uk: rows.append((int(ev), ko_uk))
    if not rows: return None

    df = pd.DataFrame(rows, columns=["event", "ko_uk"])
    next_ev = int(df["event"].min())
    in_ev = df[df["event"] == next_ev]["ko_uk"]
    return next_ev, in_ev.min(), in_ev.max()

def _espn_fetch_day(date_str: str) -> Optional[dict]:
    for d in (date_str, f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"):
        try:
            r = requests.get(ESPN_SCOREBOARD, params={"dates": d}, timeout=20)
            if r.status_code == 404: continue
            r.raise_for_status()
            return r.json()
        except Exception:
            continue
    return None

def _extract_mw_from_competition(comp: dict) -> Optional[int]:
    for path in [("round","number"), ("week","number")]:
        d = comp; ok = True
        for k in path:
            if not isinstance(d, dict) or k not in d: ok=False; break
            d = d[k]
        if ok:
            try: return int(d)
            except Exception: pass
    for path in [("round","text"), ("round","name"), ("week","text")]:
        d = comp
        for k in path:
            if isinstance(d, dict) and k in d: d = d[k]
            else: d=None; break
        if isinstance(d, str):
            m = pd.Series([d]).str.extract(r"(\d+)").iloc[0,0]
            if pd.notna(m): return int(m)
    return None

def find_next_matchweek_window_via_espn(now_uk: pd.Timestamp, horizon_days: int = 14) -> Optional[Tuple[int, pd.Timestamp, pd.Timestamp]]:
    dmin = now_uk.normalize(); dmax = dmin + pd.Timedelta(days=horizon_days)
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
    if not mw_to_times: return None
    next_mw = min(mw_to_times.keys())
    times = pd.Series(mw_to_times[next_mw])
    return next_mw, times.min(), times.max()

# ----------------- output name -----------------
def build_output_path_with_mw(out_arg: str | Path,
                              dt_min_uk: pd.Timestamp,
                              dt_max_uk: pd.Timestamp,
                              mw_numbers: List[int],
                              prefix: str = "epl_odds") -> Path:
    p = Path(out_arg)
    season = season_start_year_from_date(dt_min_uk)
    slug = f"MW{mw_numbers[0]:02d}" if mw_numbers else f"{dt_min_uk.strftime('%Y%m%d')}-{dt_max_uk.strftime('%Y%m%d')}"
    if p.suffix:
        return p.with_name(f"{p.stem}_{prefix}_{season}_{slug}{p.suffix}")
    return p / f"{prefix}_{season}_{slug}.parquet"

# ----------------- main collection -----------------
def collect_next_matchweek(api_key: str,
                           out_path: str | Path,
                           extras: List[str],
                           fallback_days: int = FALLBACK_DAYS,
                           debug: bool = False) -> Path:
    now_uk = to_uk(pd.Timestamp(utc_now()))

    # 1) find MW window
    fpl = find_next_matchweek_window_via_fpl(now_uk)
    if fpl:
        mw, start_uk, end_uk = fpl
        print(f"[info] Next MW via FPL: MW{mw:02d} | {start_uk} → {end_uk}")
        mw_numbers = [mw]
    else:
        espn = find_next_matchweek_window_via_espn(now_uk)
        if espn:
            mw, start_uk, end_uk = espn
            print(f"[info] Next MW via ESPN: MW{mw:02d} | {start_uk} → {end_uk}")
            mw_numbers = [mw]
        else:
            start_uk = now_uk.normalize()
            end_uk = start_uk + pd.Timedelta(days=fallback_days)
            print(f"[warn] Could not infer next MW; falling back to date window {start_uk} → {end_uk}")
            mw_numbers = []

    # 2) padded UTC window
    start_iso = iso_utc(start_uk.tz_convert("UTC").to_pydatetime() - dt.timedelta(hours=6))
    end_iso   = iso_utc(end_uk.tz_convert("UTC").to_pydatetime() + dt.timedelta(hours=6))

    # 3) base markets (guaranteed)
    base_markets_csv = "h2h,totals,spreads"
    print(f"[info] fetching base markets [{base_markets_csv}] for {start_iso} → {end_iso} (region={REGION})")
    try:
        base_events = fetch_bulk_odds(api_key, start_iso, end_iso, base_markets_csv)
    except requests.HTTPError as e:
        raise SystemExit(f"ERROR fetching base markets: {e}") from e

    rows_all: List[Dict] = []
    event_ids: List[str] = []

    for ev in base_events:
        rows_all.extend(flatten_event(ev))
        if ev.get("id"):
            event_ids.append(ev["id"])

    # 4) extras per event (avoid 422)
    extras = [m.strip() for m in extras if m.strip()]
    if extras and event_ids:
        for ev_id in event_ids:
            for mk in extras:
                try:
                    data = fetch_event_market(api_key, ev_id, mk)
                except requests.HTTPError as e:
                    print(f"[warn] {mk} for event {ev_id}: HTTP {e.response.status_code}")
                    data = None
                if not data:
                    continue
                # ---- FIXED LINE (removed stray brace) ----
                rows_all.extend(flatten_event(data, include_keys={canon_market_key(mk)}))

    # 5) DataFrame + guard to window
    cols = ["match_id","commence_time","home_team","away_team","bookmaker_key","bookmaker_title",
            "market_raw","market","outcome","outcome_name","price","point","last_update","region","odds_format"]
    if rows_all:
        df = pd.DataFrame(rows_all)
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
        df["last_update"]   = pd.to_datetime(df["last_update"],   utc=True, errors="coerce")
        mask = (df["commence_time"] >= pd.Timestamp(start_iso)) & (df["commence_time"] <= pd.Timestamp(end_iso))
        df = df.loc[mask].reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=cols)

    # 6) write parquet (MW naming)
    dt_min_uk = to_uk(df["commence_time"]).min() if not df.empty else start_uk
    dt_max_uk = to_uk(df["commence_time"]).max() if not df.empty else end_uk
    final_out = build_output_path_with_mw(out_path, dt_min_uk, dt_max_uk, mw_numbers)

    Path(final_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(final_out, index=False)

    print(f"[done] wrote {final_out} with {len(df):,} rows across "
          f"{df['match_id'].nunique() if 'match_id' in df else 0} matches and "
          f"{df['bookmaker_key'].nunique() if 'bookmaker_key' in df else 0} UK bookmakers.")
    if debug and not df.empty:
        print(df.head(10).to_string(index=False))
    return final_out

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Fetch EPL raw odds (UK) for the NEXT matchweek; extras attempted per-event.")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output path OR directory. If directory, creates epl_odds_<season>_MWnn.parquet in it.")
    ap.add_argument("--extras", default="btts,h2h_3_way",
                    help="Comma-separated extras tried per event (skipped quietly if unsupported).")
    ap.add_argument("--fallback-days", type=int, default=FALLBACK_DAYS,
                    help="Days to fetch if MW inference fails")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

# ----------------- entrypoint -----------------
if __name__ == "__main__":
    args = parse_args()
    if not API_KEY or API_KEY == "PUT_YOUR_THEODDSAPI_KEY_HERE":
        raise SystemExit("ERROR: please put your The Odds API key into API_KEY at the top of this script.")

    extras = [s.strip() for s in (args.extras or "").split(",") if s.strip()]
    final_path = collect_next_matchweek(
        api_key=API_KEY,
        out_path=args.out,
        extras=extras,
        fallback_days=args.fallback_days,
        debug=args.debug,
    )
    print(f"[info] odds saved to {final_path}")
