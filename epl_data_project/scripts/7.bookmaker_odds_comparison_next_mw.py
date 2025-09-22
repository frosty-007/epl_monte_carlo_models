#!/usr/bin/env python3
"""
odds_next_mw_to_json.py  —  BEST PRICES per match (robust parsing)

Reads a saved EPL odds parquet (e.g. epl_odds_2025_MW04.parquet) and outputs JSON with
the best available odds for these markets:

- Match Result (h2h / h2h_3_way): best {home, draw, away}
- BTTS (Yes/No): best {yes, no}
- Totals (Over/Under): best per line, e.g. {"2.5": {"over": {...}, "under": {...}}, ...}
- Asian Handicap: best per line, e.g. {"+0.25": {"home": {...}, "away": {...}}, ...}
- Correct Score: top-K (default 10) best scorelines overall

The script is tolerant to naming/format differences:
- Detects markets by regex (e.g., "btts" or "both teams to score").
- If `point` is missing, it tries to parse the line from outcome / outcome_name / market strings.
- Normalises team-name outcomes to "home"/"away".
- Normalises scoreline text to "H-A".

Output file:
  ../data/output/odds/bookmaker_odds_compared_<SEASON>_MWnn.json

CLI:
  --odds-file PATH      read this parquet (recommended)
  --odds-dir  DIR       if odds-file omitted, auto-pick newest epl_odds_YYYY_MWnn.parquet in DIR
  --outdir    DIR       output directory (default ../data/output/odds)
  --cs-topk   N         top-K correct scores to include (default 10)
  --debug               add diagnostics per match (markets seen) and print extra logs
"""

from __future__ import annotations
import argparse, re, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

FILENAME_RX = re.compile(r"epl_odds_(\d{4})_MW(\d{2})(?:-MW(\d{2}))?\.parquet$", re.IGNORECASE)

# ---------- time / path helpers ----------
def to_uk(ts):
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if isinstance(t, pd.Series):
        return t.dt.tz_convert("Europe/London")
    return t.tz_convert("Europe/London")

def build_output_path(outdir: Path, season: int, mws: List[int], prefix="bookmaker_odds_compared") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if mws:
        slug = f"MW{mws[0]:02d}" if len(mws) == 1 else f"MW{mws[0]:02d}-MW{mws[-1]:02d}"
    else:
        slug = "MW"
    return outdir / f"{prefix}_{season}_{slug}.json"

def pick_latest_odds_file(odds_dir: Path) -> Path:
    cand = []
    for p in odds_dir.glob("epl_odds_*_MW*.parquet"):
        m = FILENAME_RX.search(p.name)
        if not m: continue
        season = int(m.group(1))
        mw1 = int(m.group(2))
        mw2 = int(m.group(3)) if m.group(3) else mw1
        cand.append((season, mw2, p.stat().st_mtime, p))
    if not cand:
        raise FileNotFoundError(f"No files like epl_odds_YYYY_MWnn.parquet found in {odds_dir}")
    cand.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return cand[0][3]

def parse_season_mw_from_filename(path: Path) -> tuple[int, list[int]]:
    m = FILENAME_RX.search(path.name)
    if not m:
        raise ValueError(f"Filename doesn't match epl_odds_YYYY_MWnn.parquet: {path.name}")
    season = int(m.group(1))
    mw1 = int(m.group(2))
    mw2 = int(m.group(3)) if m.group(3) else mw1
    mws = list(range(mw1, mw2+1))
    return season, mws

# ---------- robust normalisation ----------
RX_BETWEEN_NUM = re.compile(r"([+-]?\d+(?:\.\d+)?)")
RX_SCORE = re.compile(r"(\d+)\D+(\d+)")

def _norm_lower(s) -> str:
    return str(s).strip().lower() if pd.notna(s) else ""

def _market_bucket(market: str) -> Optional[str]:
    m = _norm_lower(market)
    if m in ("h2h","h2h_3_way","h2h 3 way"): return "h2h"
    if "both teams" in m and "score" in m:    return "btts"
    if "btts" in m:                            return "btts"
    if "total" in m:                           return "totals"
    if "asian" in m or "handicap" in m or "spread" in m or m == "ah": return "asian_handicap"
    if "correct" in m and "score" in m:        return "correct_score"
    if m in ("correct_score","cs"):            return "correct_score"
    return None

def _parse_line_from_text(*txts) -> Optional[float]:
    """
    Try to extract a numeric line from any of the provided strings
    (e.g., 'Over 2.5', 'Arsenal -0.5', 'AH +0.25').
    """
    for t in txts:
        s = _norm_lower(t)
        if not s: continue
        m = RX_BETWEEN_NUM.search(s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

def _norm_side(outcome: str, home: str, away: str) -> Optional[str]:
    o = _norm_lower(outcome)
    h = _norm_lower(home)
    a = _norm_lower(away)
    if o in ("home","away"): return o
    if h and (o == h or h in o): return "home"
    if a and (o == a or a in o): return "away"
    return None

def _norm_yesno(outcome: str) -> Optional[str]:
    o = _norm_lower(outcome)
    if o in ("yes","y"): return "yes"
    if o in ("no","n"):  return "no"
    return None

def _norm_ou(outcome: str) -> Optional[str]:
    o = _norm_lower(outcome)
    if o.startswith("over"):  return "over"
    if o.startswith("under"): return "under"
    return None

def _norm_score(outcome: str, outcome_name: str) -> Optional[str]:
    # Prefer any "H-A" numeric inside either field
    for s in (outcome, outcome_name):
        text = str(s) if pd.notna(s) else ""
        m = RX_SCORE.search(text)
        if m:
            return f"{int(m.group(1))}-{int(m.group(2))}"
    return None

def normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - mkt_bucket          in {"h2h","btts","totals","asian_handicap","correct_score"}
      - line_val            float (from point or parsed text), if applicable
      - line_str            string: totals -> "2.5", AH -> "+0.25"
      - outcome_norm        category per bucket:
          * h2h:   home/draw/away (map team names to home/away if needed)
          * btts:  yes/no
          * totals: over/under
          * AH:    home/away
          * CS:    "H-A"
    """
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna() & (df["price"] > 1.0)]
    for c in ("market","outcome","outcome_name","bookmaker_title","home_team","away_team"):
        if c not in df.columns:
            df[c] = None

    df["mkt_bucket"] = df["market"].apply(_market_bucket)

    # Compute line for totals & AH (prefer 'point', else parse)
    has_point = pd.to_numeric(df.get("point"), errors="coerce")
    df["line_val"] = has_point
    # Fill missing from text
    fill_mask = df["line_val"].isna()
    if fill_mask.any():
        df.loc[fill_mask, "line_val"] = df.loc[fill_mask].apply(
            lambda r: _parse_line_from_text(r.get("outcome"), r.get("outcome_name"), r.get("market")), axis=1
        )
    # Strings: totals plain, AH signed
    df["line_str"] = df.apply(
        lambda r: (None if pd.isna(r["line_val"]) else (f"{r['line_val']:+g}" if r["mkt_bucket"]=="asian_handicap" else f"{r['line_val']:g}")),
        axis=1
    )

    # outcome_norm per bucket
    out_norm = []
    for _, r in df.iterrows():
        bkt = r["mkt_bucket"]
        if bkt == "h2h":
            o = _norm_lower(r["outcome"])
            if o not in ("home","draw","away"):
                o = _norm_side(r["outcome"], r["home_team"], r["away_team"]) or o
            out_norm.append(o if o in ("home","draw","away") else None)
        elif bkt == "btts":
            out_norm.append(_norm_yesno(r["outcome"]))
        elif bkt == "totals":
            out_norm.append(_norm_ou(r["outcome"]))
        elif bkt == "asian_handicap":
            out_norm.append(_norm_side(r["outcome"], r["home_team"], r["away_team"]))
        elif bkt == "correct_score":
            out_norm.append(_norm_score(r["outcome"], r["outcome_name"]))
        else:
            out_norm.append(None)
    df["outcome_norm"] = out_norm

    return df

# ---------- best-pick helpers ----------
def _best_row(rows: pd.DataFrame, price_col="price") -> Optional[dict]:
    if rows.empty: return None
    r = rows.loc[rows[price_col].astype(float).idxmax()]
    return {"price": float(r[price_col]), "bookmaker": r.get("bookmaker_title")}

def _best_by_outcome(df: pd.DataFrame, outcomes: List[str]) -> Dict[str, Optional[dict]]:
    out: Dict[str, Optional[dict]] = {}
    for oc in outcomes:
        sub = df[df["outcome_norm"] == oc]
        out[oc] = _best_row(sub)
    # drop keys that have no data at all
    return {k: v for k, v in out.items() if v is not None}

def _best_by_line_two_way(df: pd.DataFrame, outcomes=("over","under")) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if df.empty: return out
    df2 = df.dropna(subset=["line_str"])
    for line, sub in df2.groupby("line_str", dropna=False):
        bests = {}
        for oc in outcomes:
            ss = sub[sub["outcome_norm"] == oc]
            b = _best_row(ss)
            if b is not None: bests[oc] = b
        if bests:
            out[str(line)] = bests
    return out

def collect_best_markets_for_match(df_norm: pd.DataFrame, cs_topk: int, debug: bool=False) -> Tuple[dict, dict]:
    """
    Returns (markets_best, diag)
      markets_best: dict with h2h, btts, totals, asian_handicap, correct_score
      diag: diagnostics (unique raw market labels present) for debugging
    """
    diag = {}
    if debug:
        diag["markets_seen"] = (
            df_norm["market"].dropna().astype(str).str.lower().value_counts().to_dict()
        )

    out = {}

    # H2H
    h2h = df_norm[df_norm["mkt_bucket"]=="h2h"]
    if not h2h.empty:
        bests = _best_by_outcome(h2h, ["home","draw","away"])
        if bests: out["h2h"] = bests

    # BTTS
    btts = df_norm[df_norm["mkt_bucket"]=="btts"]
    if not btts.empty:
        bests = _best_by_outcome(btts, ["yes","no"])
        if bests: out["btts"] = bests

    # Totals
    tot = df_norm[(df_norm["mkt_bucket"]=="totals") & df_norm["outcome_norm"].isin(["over","under"])]
    if not tot.empty:
        tb = _best_by_line_two_way(tot, outcomes=("over","under"))
        if tb: out["totals"] = tb

    # Asian Handicap
    ah = df_norm[(df_norm["mkt_bucket"]=="asian_handicap") & df_norm["outcome_norm"].isin(["home","away"])]
    if not ah.empty:
        ab = _best_by_line_two_way(ah, outcomes=("home","away"))
        if ab: out["asian_handicap"] = ab

    # Correct Score (top-K)
    cs = df_norm[(df_norm["mkt_bucket"]=="correct_score") & df_norm["outcome_norm"].notna()]
    if not cs.empty:
        # best price per score
        cs_best = (cs.groupby("outcome_norm")["price"].max().reset_index())
        cs_best = cs_best.rename(columns={"outcome_norm":"score","price":"price"}).sort_values("price", ascending=False)
        merged = cs.merge(cs_best, left_on=["outcome_norm","price"], right_on=["score","price"], how="inner")
        merged = merged.sort_values(["price"], ascending=False).drop_duplicates(subset=["score"])
        merged = merged[["score","price","bookmaker_title"]].head(cs_topk)
        if len(merged):
            out["correct_score"] = [
                {"score": str(r["score"]), "price": float(r["price"]), "bookmaker": r["bookmaker_title"]}
                for _, r in merged.iterrows()
            ]

    return out, diag

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds-file", default="", help="Exact parquet to load (e.g. ../data/raw/odds/epl_odds_2025_MW04.parquet)")
    ap.add_argument("--odds-dir",  default="../data/raw/odds", help="Directory to auto-detect newest epl_odds_YYYY_MWnn.parquet")
    ap.add_argument("--outdir",    default="../data/output/odds", help="Output directory")
    ap.add_argument("--cs-topk",   type=int, default=10, help="Top-K correct score selections per match")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # choose file
    if args.odds_file:
        odds_path = Path(args.odds_file)
        if not odds_path.exists():
            raise FileNotFoundError(f"{odds_path} not found")
    else:
        odds_path = pick_latest_odds_file(Path(args.odds_dir))

    season, mws = parse_season_mw_from_filename(odds_path)
    if args.debug:
        print(f"[info] using odds file: {odds_path.name} (season={season}, MW={mws})")

    # load
    df = pd.read_parquet(odds_path)
    need = {"match_id","commence_time","home_team","away_team","market","outcome","bookmaker_title","price"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    # time
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")

    # normalise once
    df_norm = normalise_df(df)

    # fixtures
    fixtures = (df[["match_id","commence_time","home_team","away_team"]]
                .drop_duplicates("match_id")
                .rename(columns={"commence_time":"kickoff_utc"}))
    fixtures["kickoff_uk"] = to_uk(fixtures["kickoff_utc"])

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "fixtures_count": int(len(fixtures)),
        "matchweeks": mws,
        "window_uk": {
            "start": fixtures["kickoff_uk"].min().isoformat() if len(fixtures) else None,
            "end":   fixtures["kickoff_uk"].max().isoformat() if len(fixtures) else None,
        },
        "fixtures": []
    }

    by_match = dict(tuple(df_norm.groupby("match_id", sort=False)))

    for _, fx in fixtures.sort_values("kickoff_utc").iterrows():
        mid = fx["match_id"]
        sub = by_match.get(mid, pd.DataFrame(columns=df_norm.columns))
        markets_best, diag = collect_best_markets_for_match(sub, cs_topk=args.cs_topk, debug=args.debug)
        md = {
            "match_id": int(mid) if pd.notna(mid) and str(mid).isdigit() else (None if pd.isna(mid) else str(mid)),
            "kickoff_utc": fx["kickoff_utc"].isoformat() if pd.notna(fx["kickoff_utc"]) else None,
            "kickoff_uk": fx["kickoff_uk"].isoformat() if pd.notna(fx["kickoff_uk"]) else None,
            "home_team": fx.get("home_team"),
            "away_team": fx.get("away_team"),
            "markets_best": markets_best
        }
        if args.debug:
            md["debug"] = diag
        payload["fixtures"].append(md)

    out_path = build_output_path(Path(args.outdir), season, mws)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote {out_path}")
    if args.debug:
        print(f"[debug] fixtures={payload['fixtures_count']} | season={season} | MW={mws} | "
              f"window_uk={payload['window_uk']['start']} → {payload['window_uk']['end']}")

if __name__ == "__main__":
    main()
