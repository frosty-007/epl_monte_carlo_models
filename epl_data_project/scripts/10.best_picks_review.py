#!/usr/bin/env python3
"""
10.best_picks_review.py  (MATCHWEEK + ESPN-ONLY, FIXED 2025 INPUT SCHEME)

What it does
------------
Grades picks (singles + accas) using **ESPN public scoreboard** results only (no API keys).
Input picks file is fixed to the 2025 season naming scheme:

  ../data/output/picks/picks_next_epl_2025_mwNN.json

…where NN is the **matchweek** provided via CLI (--mw). Output uses:

  ../data/output/pick_analysis/epl_2025_mwNN_pick_analysis.json

Notes
-----
- All legacy "round" logic is removed; the script is matchweek-first.
- ESPN’s endpoint doesn’t expose matchweek; we use your CLI arg for naming and propagate
  a `matchweek` field in outputs with that value when helpful (results themselves come
  from ESPN and don’t contain MW numbers).
- Matching uses exact (home, away, date), swapped home/away, nearest kickoff within ±2 days,
  and light fuzzy name matching.

Usage
-----
  python 10.best_picks_review.py --mw 7
  python 10.best_picks_review.py --mw 7 --debug
  python 10.best_picks_review.py --mw 7 --out ../data/output/pick_analysis/custom.json

Requires
--------
  pip install pandas numpy requests
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from datetime import datetime, date
from difflib import SequenceMatcher

# ---------- Config ----------
SEASON_FOR_NAME = 2025  # fixed season naming per request
DEFAULT_PICKS_DIR = Path("../data/output/picks")
DEFAULT_OUT_DIR   = Path("../data/output/pick_analysis")

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
    ap = argparse.ArgumentParser(description="Grade picks using ESPN (matchweek-first; ESPN-only; fixed 2025 input).")
    ap.add_argument("--mw", type=int, required=True,
                    help="EPL matchweek (1..38). Input will be ../data/output/picks/picks_next_epl_2025_mwNN.json")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Output graded JSON (default {DEFAULT_OUT_DIR}/epl_2025_mwNN_pick_analysis.json)")
    ap.add_argument("--debug", action="store_true", help="Print extra logs and write debug_unmatched.csv")
    return ap.parse_args()

# ---------- Picks I/O ----------
def load_picks_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_fixed_2025_picks_path(mw: int, base_dir: Path = DEFAULT_PICKS_DIR) -> Path:
    if mw < 1 or mw > 38:
        raise ValueError(f"matchweek must be 1..38, got {mw}")
    p = base_dir / f"picks_next_epl_{SEASON_FOR_NAME}_mw{mw:02d}.json"
    if not p.exists():
        raise FileNotFoundError(f"Picks file not found: {p}")
    return p

# ---------- Utilities ----------
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
                "matchweek": None,  # ESPN doesn't publish MW
            })

    out = pd.DataFrame(rows)
    if debug:
        print(f"[debug] ESPN: {'no rows' if out.empty else f'{len(out)} rows'}")
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

# ---------- Main ----------
def main():
    args = parse_args()

    # Picks path based on fixed 2025 scheme
    picks_path = resolve_fixed_2025_picks_path(args.mw)

    # Load picks
    picks = load_picks_json(picks_path)
    singles = picks.get("singles") or picks.get("picks") or []
    accas   = picks.get("accas") or []

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

    # Date window for ESPN fetch
    if s["kickoff_uk_dt"].notna().any():
        dmin = s["kickoff_uk_dt"].min() - pd.Timedelta(days=2)
        dmax = s["kickoff_uk_dt"].max() + pd.Timedelta(days=2)
    else:
        now = pd.Timestamp.now(tz="Europe/London")
        dmin, dmax = now - pd.Timedelta(days=7), now + pd.Timedelta(days=7)

    # ESPN ONLY
    espn_df = fetch_espn_results(dmin, dmax, debug=args.debug)
    if args.debug:
        print(f"[debug] ESPN: {'no rows' if espn_df.empty else f'{len(espn_df)} rows'}")

    sources: List[Tuple[str, pd.DataFrame]] = [("ESPN", espn_df)]

    # Working join frame
    j = s.copy()
    for col in ["home_goals","away_goals","kickoff_uk_res","matchweek"]:
        if col not in j.columns:
            j[col] = pd.NA

    # Matching passes
    for src_name, src_df in sources:
        if src_df is None or src_df.empty:
            continue
        src = src_df.copy()
        src["kick_date_uk"] = pd.to_datetime(src["kick_date_uk"], errors="coerce").dt.date
        src["pair_key"] = src.apply(lambda r: _build_pair_key(r.get("home_key"), r.get("away_key")), axis=1)

        before = j["home_goals"].notna().sum()

        # Pass 1: exact (home, away, date)
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
                    j.loc[fill_idx, "matchweek"] = j.loc[fill_idx, "matchweek"].fillna(m1["matchweek"])

        # Pass 2: swapped (handle potential inversions)
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
                    j.loc[fill_idx, "matchweek"] = j.loc[fill_idx, "matchweek"].fillna(m2["matchweek"])

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
                    j.at[idx, "matchweek"] = j.at[idx, "matchweek"] if pd.notna(j.at[idx, "matchweek"]) else best.get("matchweek")

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
                    j.at[idx, "matchweek"] = j.at[idx, "matchweek"] if pd.notna(j.at[idx, "matchweek"]) else best.get("matchweek")

        after = j["home_goals"].notna().sum()
        if args.debug:
            print(f"[debug] {src_name}: matched +{after - before}, now matched {after}/{len(j)}")
        if after == len(j):
            break  # all done

    # Build final labels (attach the CLI MW for traceability)
    finals = []
    for _, r in j.iterrows():
        hg = r.get("home_goals"); ag = r.get("away_goals")
        finals.append({
            "home_goals": int(hg) if pd.notna(hg) else None,
            "away_goals": int(ag) if pd.notna(ag) else None,
            "matchweek": int(args.mw),   # propagate requested MW (ESPN doesn't provide one)
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
    for a in (picks.get("accas") or []):
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

    # ---------- Output path ----------
    if args.out is not None:
        out_path = args.out
    else:
        out_path = DEFAULT_OUT_DIR / f"epl_{SEASON_FOR_NAME}_mw{args.mw:02d}_pick_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Write payload ----------
    payload = {
        "source": {
            "picks": str(picks_path),
            "espn": {"enabled": True},
        },
        "summary": {
            "singles": {
                "count_total": n_s_total,
                "count_graded": n_s,
                "wins": n_sw,
                "hit_rate": hit_s,
                "profit (Total £s won or lost)": pnl_s,
                "roi_per_bet (average profit per wager)": roi_s
            },
            "accas":   {
                "count_total": len(ga_df_all),
                "count_graded": n_a,
                "wins": n_aw,
                "hit_rate": hit_a,
                "profit (Total £s won or lost)": pnl_a,
                "roi_per_bet (average profit per wager)": roi_a
            },
            "matchweek": int(args.mw),
        },
        "singles": graded_singles,
        "accas": graded_accas,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    unmatched = int((pd.Series([d.get("won") for d in graded_singles]).isna()).sum())
    print(f"[done] MW={args.mw:02d} | singles total={n_s_total} graded={n_s} (wins={n_sw}, unmatched={unmatched}), "
          f"accas total={len(ga_df_all)} graded={n_a} (wins={n_aw}) → {out_path}")
    if n_s:
        print(f"Singles ROI: {roi_s:+.3f}  |  Accas ROI: {roi_a:+.3f}" if n_a else f"Singles ROI: {roi_s:+.3f}")

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
