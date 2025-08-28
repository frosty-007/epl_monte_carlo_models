#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import timedelta
from typing import List, Optional, Tuple, Dict, Any, Set

import pandas as pd
import numpy as np
import requests

# -------- paths --------
ODDS_PARQUET = Path("../data/raw/odds/odds.parquet")
OUT_DIR      = Path("../data/output/odds")  # directory; final filename is built with season + MW

# -------- external endpoints (no keys) --------
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"

# -------- time / naming helpers --------
def to_uk(ts):
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if isinstance(t, pd.Series):
        return t.dt.tz_convert("Europe/London")
    return t.tz_convert("Europe/London")

def season_start_year_from_date(d_uk: pd.Timestamp) -> int:
    """PL season starts in Aug. Aug–Dec => same year; Jan–May => previous year."""
    return int(d_uk.year if d_uk.month >= 8 else d_uk.year - 1)

def build_output_path_with_mw(out_dir: Path,
                              dt_min_uk: pd.Timestamp,
                              dt_max_uk: pd.Timestamp,
                              mw_numbers: List[int],
                              prefix: str = "bookmaker_odds_compared") -> Path:
    """
    Create <prefix>_<season>_MWxx.json (or MWxx-MWyy if spanning) under out_dir.
    Fallback slug: YYYYMMDD-YYYYMMDD if MW unknown.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    season = season_start_year_from_date(dt_min_uk)
    if mw_numbers:
        slug = f"MW{mw_numbers[0]:02d}" if len(mw_numbers) == 1 else f"MW{mw_numbers[0]:02d}-MW{mw_numbers[-1]:02d}"
    else:
        slug = f"{dt_min_uk.strftime('%Y%m%d')}-{dt_max_uk.strftime('%Y%m%d')}"
    return out_dir / f"{prefix}_{season}_{slug}.json"

# -------- MW inference (FPL first, ESPN fallback) --------
def infer_matchweeks_from_fpl(date_from_uk: pd.Timestamp,
                              date_to_uk: pd.Timestamp) -> List[int]:
    """
    Use FPL fixtures (public) to get unique 'event' numbers (GW) within the window.
    """
    try:
        r = requests.get(FPL_FIXTURES, timeout=20)
        r.raise_for_status()
        fixtures = r.json()
        if not isinstance(fixtures, list):
            return []
    except Exception:
        return []

    dmin = min(date_from_uk, date_to_uk)
    dmax = max(date_from_uk, date_to_uk)
    mws: Set[int] = set()
    for fx in fixtures:
        ev = fx.get("event")
        ko = fx.get("kickoff_time")
        if ev is None or not ko:
            continue
        ko_uk = to_uk(pd.Timestamp(ko))
        if dmin <= ko_uk <= dmax:
            try:
                mws.add(int(ev))
            except Exception:
                pass
    return sorted(mws)

def _espn_fetch_day(date_str: str) -> Optional[dict]:
    """Try YYYYMMDD (primary) then YYYY-MM-DD (fallback)."""
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
    # Try round.number or week.number first
    for path in [("round", "number"), ("week", "number")]:
        d = comp
        ok = True
        for k in path:
            if not isinstance(d, dict) or k not in d:
                ok = False; break
            d = d[k]
        if ok:
            try:
                return int(d)
            except Exception:
                pass
    # Parse text fallback (e.g., "Round 3")
    for path in [("round", "text"), ("round", "name"), ("week", "text")]:
        d = comp
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                d = None; break
        if isinstance(d, str):
            m = pd.Series([d]).str.extract(r"(\d+)").iloc[0, 0]
            if pd.notna(m):
                return int(m)
    return None

def infer_matchweeks_from_espn(date_from_uk: pd.Timestamp,
                               date_to_uk: pd.Timestamp) -> List[int]:
    dmin = min(date_from_uk, date_to_uk)
    dmax = max(date_from_uk, date_to_uk)
    mws: Set[int] = set()
    for day in pd.date_range(dmin.date(), dmax.date(), freq="D"):
        ds = day.strftime("%Y%m%d")
        data = _espn_fetch_day(ds)
        if not data:
            continue
        events = data.get("events", []) or []
        for ev in events:
            comps = ev.get("competitions", []) or []
            if not comps:
                continue
            mw = _extract_mw_from_competition(comps[0])
            if isinstance(mw, int):
                mws.add(mw)
    return sorted(mws)

# -------- existing logic (unchanged) --------
def load_and_filter_next7_epl(odds: pd.DataFrame) -> pd.DataFrame:
    """Filter raw odds to EPL fixtures in the next 7 days (UTC)."""
    df = odds.copy()
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")

    now_utc = pd.Timestamp.now(tz="UTC")
    end_utc = now_utc + timedelta(days=7)
    df = df[(df["commence_time"] >= now_utc) & (df["commence_time"] <= end_utc)]

    # EPL filter (robust across feeds)
    masks = []
    if "sport_key" in df.columns:
        sk = df["sport_key"].astype(str).str.lower()
        masks.append(sk.eq("soccer_epl") | sk.str.contains("epl", na=False) | sk.str.contains("england.*premier", na=False))
    if "league_title" in df.columns:
        lt = df["league_title"].astype(str).str.lower()
        masks.append(lt.str.contains("premier league", na=False) & lt.str.contains("england|english", na=False))

    if masks:
        m = masks[0]
        for mm in masks[1:]:
            m = m | mm
        df = df[m].copy()

    keep = [c for c in [
        "match_id","commence_time","home_team","away_team",
        "market","outcome","point","bookmaker_title","price"
    ] if c in df.columns]
    return df[keep].copy()

def by_bookmaker_pivot(df: pd.DataFrame, index_cols, col_name="outcome", val="price"):
    """
    One row per bookmaker with columns per outcome (e.g., home/draw/away or yes/no).
    Aggregates duplicate rows by taking max price per bookmaker/outcome.
    Resilient to duplicate column names.
    """
    if df.empty:
        return pd.DataFrame(columns=index_cols + ["bookmaker_title"])
    df = df.loc[:, ~df.columns.duplicated()].copy()  # avoid 1-D grouper errors
    g = (df.groupby(index_cols + ["bookmaker_title", col_name])[val]
           .max()
           .unstack(col_name)
           .reset_index())
    return g

def collect_markets_for_match(df: pd.DataFrame) -> dict:
    """Build nested dict of markets for a single match (df filtered by match_id)."""
    out = {}

    # ---- 1X2 (match result) ----
    h2h = df[df["market"].isin(["h2h", "h2h_3_way"])]
    if not h2h.empty:
        h2h_bm = by_bookmaker_pivot(h2h, index_cols=[], col_name="outcome", val="price")
        cols = [c for c in ["home", "draw", "away"] if c in h2h_bm.columns]
        out["h2h"] = [
            {"bookmaker": r["bookmaker_title"], **{k: float(r[k]) for k in cols if pd.notna(r[k])}}
            for _, r in h2h_bm.iterrows()
        ]

    # ---- BTTS (Yes/No) ----
    btts = df[df["market"] == "btts"]
    if not btts.empty:
        btts = btts[btts["outcome"].isin(["yes", "no"])]
        btts_bm = by_bookmaker_pivot(btts, index_cols=[], col_name="outcome", val="price")
        cols = [c for c in ["yes", "no"] if c in btts_bm.columns]
        out["btts"] = [
            {"bookmaker": r["bookmaker_title"], **{k: float(r[k]) for k in cols if pd.notna(r[k])}}
            for _, r in btts_bm.iterrows()
        ]

    # ---- Totals (Over/Under by line/point) ----
    # primary label "totals"; fall back to any market containing "total"
    tot = df[(df["market"] == "totals") & (df["outcome"].isin(["over", "under"]))].copy()
    if tot.empty:
        tot = df[df["market"].astype(str).str.contains("total", case=False, na=False) &
                 df["outcome"].isin(["over", "under"])].copy()

    if not tot.empty:
        pts = pd.to_numeric(tot.get("point"), errors="coerce")
        tot["point_str"] = pts.map(lambda x: None if pd.isna(x) else f"{x:g}")
        tot = tot.dropna(subset=["point_str"])

        out["totals"] = {}
        for pt, sub in tot.groupby("point_str", dropna=False):
            # keep columns we need; do NOT rename to "point" (avoids duplicate-name grouper)
            sub2 = sub[["bookmaker_title", "outcome", "price", "point_str"]].copy()
            bm = by_bookmaker_pivot(sub2, index_cols=["point_str"], col_name="outcome", val="price")
            cols = [c for c in ["over", "under"] if c in bm.columns]
            out["totals"][pt] = [
                {"bookmaker": r["bookmaker_title"], **{k: float(r[k]) for k in cols if pd.notna(r[k])}}
                for _, r in bm.iterrows()
            ]

    # ---- Correct Score (match score) ----
    cs = df[df["market"].astype(str).str.contains("correct", case=False, na=False)]
    if cs.empty:
        cs = df[df["market"].isin(["correct_score", "cs"])]
    if not cs.empty:
        # best price per bookmaker+scoreline
        cs_best = (cs.groupby(["bookmaker_title", "outcome"])["price"]
                     .max()
                     .reset_index()
                     .rename(columns={"outcome": "scoreline", "price": "odds"}))

        books = []
        for bk, sub in cs_best.groupby("bookmaker_title"):
            books.append({
                "bookmaker": bk,
                "prices": [{"score": s["scoreline"], "price": float(s["odds"])} for _, s in sub.iterrows()]
            })
        out["correct_score"] = books

    return out

# -------- main --------
def main():
    if not ODDS_PARQUET.exists():
        raise FileNotFoundError(f"{ODDS_PARQUET} not found")

    raw = pd.read_parquet(ODDS_PARQUET)
    odds7 = load_and_filter_next7_epl(raw)

    fixtures = (odds7[["match_id", "commence_time", "home_team", "away_team"]]
                .drop_duplicates("match_id")
                .rename(columns={"commence_time": "kickoff_utc"}))
    fixtures["kickoff_uk"] = fixtures["kickoff_utc"].dt.tz_convert("Europe/London")

    # Determine window for naming
    if fixtures.empty:
        # If no fixtures, use next 7-day window for naming
        now_uk = pd.Timestamp.now(tz="Europe/London")
        dt_min_uk = now_uk
        dt_max_uk = now_uk + pd.Timedelta(days=7)
    else:
        dt_min_uk = fixtures["kickoff_uk"].min()
        dt_max_uk = fixtures["kickoff_uk"].max()

    # Infer MWs (FPL first, ESPN fallback)
    mws = infer_matchweeks_from_fpl(dt_min_uk, dt_max_uk)
    if not mws:
        mws = infer_matchweeks_from_espn(dt_min_uk, dt_max_uk)

    out_path = build_output_path_with_mw(OUT_DIR, dt_min_uk, dt_max_uk, mws)

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "fixtures_count": int(len(fixtures)),
        "matchweeks": mws if mws else None,
        "fixtures": []
    }

    by_match = dict(tuple(odds7.groupby("match_id", sort=False)))

    for _, fx in fixtures.sort_values("kickoff_utc").iterrows():
        mid = fx["match_id"]
        md = {
            "match_id": int(mid) if pd.notna(mid) and str(mid).isdigit() else str(mid),
            "kickoff_utc": fx["kickoff_utc"].isoformat(),
            "kickoff_uk": fx["kickoff_uk"].isoformat() if pd.notna(fx["kickoff_uk"]) else None,
            "home_team": fx["home_team"],
            "away_team": fx["away_team"],
            "markets": {}
        }
        df_m = by_match.get(mid, pd.DataFrame(columns=odds7.columns))
        md["markets"] = collect_markets_for_match(df_m)
        payload["fixtures"].append(md)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote {out_path}")
    print(f"[info] fixtures included (EPL, next 7 days): {payload['fixtures_count']}")
    if mws:
        print(f"[info] inferred matchweek(s): {mws}")

if __name__ == "__main__":
    main()
