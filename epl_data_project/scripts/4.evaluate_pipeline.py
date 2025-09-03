#!/usr/bin/env python3
"""
Football Prediction Evaluation Pipeline — auto names (season + NEXT MW)
======================================================================

Reads per-team lambdas (parquet) → builds per-match Poisson 1X2 → evaluates
historical rows (Brier, LogLoss, ECE). Output filenames are *automatically*
stamped with <season> and the **next** matchweek inferred from dates (UK).

Run (no args needed):
    python 4.evaluate_pipeline.py

Defaults:
- input parquet : ../data/calibrated/team_match_lambdas.parquet
- out directory : ../data/output/evaluated
- calibration   : off (use --calibrate multinomial to add in-script diagnostic calibration)

If you prefer manual override, you can still pass:
  --season 2025  --mw 03
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# Optional sklearn for in-script calibration
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore

EPS = 1e-15
RESULT_MAP = {"H": 0, "D": 1, "A": 2}

# ----------------------------
# Date & naming helpers (UK)
# ----------------------------
def _season_year_now_uk() -> int:
    d = pd.Timestamp.now(tz="Europe/London").date()
    return d.year if d.month >= 8 else d.year - 1

def _to_uk(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return t.tz_convert("Europe/London")

def _monday_of_week(d: pd.Timestamp | pd.Timestamp.date) -> pd.Timestamp.date:
    if isinstance(d, pd.Timestamp):
        d = d.date()
    return d - pd.Timedelta(days=pd.Timestamp(d).weekday())

def _season_year_for_date_uk(d: pd.Timestamp | pd.Timestamp.date) -> int:
    if isinstance(d, pd.Timestamp):
        d = d.tz_convert("Europe/London").date()
    return d.year if d.month >= 8 else d.year - 1

def _season_start_monday(season_year: int) -> pd.Timestamp.date:
    aug1 = pd.Timestamp(year=season_year, month=8, day=1).date()
    offset = (7 - pd.Timestamp(aug1).weekday()) % 7
    return aug1 + pd.Timedelta(days=offset)

def _mw_for_date_uk(dt_uk: pd.Timestamp) -> Tuple[int, int]:
    """Return (season_year, mw>=1) for a single UK datetime."""
    d = dt_uk.date()
    season = _season_year_for_date_uk(dt_uk)
    s0 = _season_start_monday(season)
    mw = int((( _monday_of_week(d) - s0).days // 7) + 1)
    return season, max(1, mw)

def infer_next_mw_and_season(match_df: pd.DataFrame) -> Tuple[int, int]:
    """
    Infer <season, NEXT mw> from match_df dates (UK):
      - If future matches exist (no goals), use the earliest upcoming kickoff date.
      - Else use last historical date's mw + 1 (capped to 38).
    """
    if match_df.empty:
        return _season_year_now_uk(), 1

    # UK times
    kt = pd.to_datetime(match_df["kickoff_time"], utc=True, errors="coerce").dt.tz_convert("Europe/London")
    df = match_df.copy()
    df["kt_uk"] = kt

    # Split historical vs upcoming
    hist = df[df["home_goals"].notna() & df["away_goals"].notna()]
    upc  = df[~(df["home_goals"].notna() & df["away_goals"].notna())]

    if not upc.empty:
        ref = upc["kt_uk"].min()
        season, mw = _mw_for_date_uk(ref)
        return season, min(38, max(1, mw))
    elif not hist.empty:
        last_dt = hist["kt_uk"].max()
        season, mw = _mw_for_date_uk(last_dt)
        return season, min(38, max(1, mw + 1))
    else:
        # No clear signals; fall back to current UK season
        return _season_year_now_uk(), 1

# ----------------------------
# Scoring utilities
# ----------------------------
def _one_hot(y: np.ndarray, K: int) -> np.ndarray:
    y = y.astype(int)
    out = np.zeros((y.shape[0], K), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

def _clamp_probs(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s

def brier_multiclass(y_true: np.ndarray, p: np.ndarray) -> float:
    K = p.shape[1]
    oh = _one_hot(y_true, K)
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))

def logloss_multiclass(y_true: np.ndarray, p: np.ndarray) -> float:
    p = _clamp_probs(p)
    return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))

def ece_maxprob(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    conf = p.max(axis=1)
    preds = p.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if not np.any(m):
            continue
        acc = np.mean(y_true[m] == preds[m])
        avg_conf = np.mean(conf[m])
        ece += (np.sum(m) / n) * abs(acc - avg_conf)
    return float(ece)

# ----------------------------
# Poisson conversion
# ----------------------------
def _poisson_pmf(lam: float, max_g: int) -> np.ndarray:
    g = np.arange(0, max_g + 1)
    pmf = np.zeros_like(g, dtype=float)
    pmf[0] = math.exp(-lam)
    for i in range(1, len(g)):
        pmf[i] = pmf[i - 1] * lam / i
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf

def poisson_1x2(lambda_home: float, lambda_away: float, max_goals: int = 12) -> Tuple[float, float, float]:
    ph = _poisson_pmf(lambda_home, max_goals)
    pa = _poisson_pmf(lambda_away, max_goals)
    M = np.outer(ph, pa)
    p_H = float(np.tril(M, k=-1).sum())
    p_D = float(np.trace(M))
    p_A = float(np.triu(M, k=1).sum())
    s = p_H + p_D + p_A
    if s <= 0:
        return 1/3, 1/3, 1/3
    return p_H / s, p_D / s, p_A / s

# ----------------------------
# Build per-match from team lambdas
# ----------------------------
def build_from_team_lambdas(parquet_path: str, max_goals: int = 12) -> pd.DataFrame:
    """
    Expected input (per-team rows): match_id, date, team, opp, home/is_home/etc, goals (optional),
    and at least one lambda* column (e.g. lambda_glm).
    """
    df = pd.read_parquet(parquet_path).copy()

    # kickoff_time from `date` (epoch ms OR datetime OR string)
    if "date" not in df.columns:
        raise KeyError("Missing 'date' column in lambdas parquet")

    s = df["date"]
    if is_numeric_dtype(s):
        df["kickoff_time"] = pd.to_datetime(s, unit="ms", utc=True)
    elif is_datetime64_any_dtype(s):
        df["kickoff_time"] = pd.to_datetime(s, utc=True)
    else:
        df["kickoff_time"] = pd.to_datetime(s, utc=True, errors="coerce")

    # find lambda column
    cand_lcols = [c for c in df.columns if "lambda" in c.lower()]
    if not cand_lcols:
        raise KeyError("No lambda* column found (e.g. 'lambda_glm')")
    lcol = cand_lcols[0]

    def _get_home_away_rows(grp: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        cand_cols = [c for c in ["home", "is_home", "home_flag", "home_away", "side"] if c in grp.columns]
        if cand_cols:
            c = cand_cols[0]
            vals = grp[c].astype(str).str.lower().str.strip()
            is_home = vals.isin(["1", "true", "t", "yes", "y", "home", "h"])
            if is_home.sum() == 1:
                return grp.loc[is_home].iloc[0], grp.loc[~is_home].iloc[0]
        return grp.iloc[0], grp.iloc[1]  # deterministic fallback

    rows: List[Dict] = []
    for mid, grp in df.groupby("match_id", sort=False):
        if grp.shape[0] < 2:
            continue
        home_row, away_row = _get_home_away_rows(grp)

        hl = float(home_row.get(lcol)) if not pd.isna(home_row.get(lcol)) else np.nan
        al = float(away_row.get(lcol)) if not pd.isna(away_row.get(lcol)) else np.nan

        if not pd.isna(hl) and not pd.isna(al):
            p_H, p_D, p_A = poisson_1x2(hl, al, max_goals=max_goals)
        else:
            p_H = p_D = p_A = np.nan

        hg = home_row.get("goals", pd.NA)
        ag = away_row.get("goals", pd.NA)
        hg = None if pd.isna(hg) else int(hg)
        ag = None if pd.isna(ag) else int(ag)

        if hg is not None and ag is not None:
            res = "H" if hg > ag else ("D" if hg == ag else "A")
        else:
            res = None

        rows.append({
            "match_id": mid,
            "kickoff_time": home_row.get("kickoff_time"),
            "home_team": home_row.get("team"),
            "away_team": away_row.get("team"),
            "home_goals": hg,
            "away_goals": ag,
            "home_lambda": hl,
            "away_lambda": al,
            "p_H": p_H,
            "p_D": p_D,
            "p_A": p_A,
            "result": res,
        })

    match_df = pd.DataFrame(rows)
    match_df = match_df.dropna(subset=["p_H", "p_D", "p_A"], how="any").reset_index(drop=True)
    match_df["is_historical"] = match_df["home_goals"].notna() & match_df["away_goals"].notna()
    return match_df

# ----------------------------
# Evaluator
# ----------------------------
@dataclass
class EvalConfig:
    K: int = 3
    n_bins: int = 10
    group_cols: Sequence[str] = ("home_team", "away_team")
    calibrate: str = "none"           # "none" | "multinomial"
    include_segments_in_json: bool = False
    max_segments: Optional[int] = 50
    summary_precision: int = 6
    include_ece_in_skill: bool = False

class Evaluator:
    def __init__(self, y: np.ndarray, P_raw: np.ndarray, cfg: EvalConfig):
        self.y = y.astype(int)
        self.P_raw = _clamp_probs(P_raw)
        self.cfg = cfg
        self.P_cal: Optional[np.ndarray] = None

    def _calibrate(self) -> Optional[np.ndarray]:
        if self.cfg.calibrate == "none":
            return None
        if self.cfg.calibrate == "multinomial":
            if LogisticRegression is None:
                return None
            X = np.log(_clamp_probs(self.P_raw))
            lr = LogisticRegression(multi_class="multinomial", max_iter=1000)
            lr.fit(X, self.y)
            return _clamp_probs(lr.predict_proba(X))
        return None

    def _metrics(self, y: np.ndarray, P: np.ndarray) -> Dict[str, float]:
        return {
            "brier": brier_multiclass(y, P),
            "logloss": logloss_multiclass(y, P),
            "ece": ece_maxprob(y, P, n_bins=self.cfg.n_bins),
        }

    def _segment_metrics(self, y: np.ndarray, P: np.ndarray, groups: pd.DataFrame) -> pd.DataFrame:
        K = P.shape[1]
        df = groups.copy().reset_index(drop=True)
        df["y"] = y
        for i, lab in enumerate(["H", "D", "A"][:K]):
            df[f"p_{lab}"] = P[:, i]
        seg = (
            df.groupby(list(self.cfg.group_cols))
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "brier": brier_multiclass(g["y"].to_numpy(), g[["p_H","p_D","p_A"]].to_numpy()),
                  "logloss": logloss_multiclass(g["y"].to_numpy(), g[["p_H","p_D","p_A"]].to_numpy()),
                  "ece": ece_maxprob(g["y"].to_numpy(), g[["p_H","p_D","p_A"]].to_numpy(), n_bins=self.cfg.n_bins),
              }))
              .reset_index()
              .sort_values("n", ascending=False)
        )
        if self.cfg.max_segments is not None and len(seg) > self.cfg.max_segments:
            seg = seg.head(self.cfg.max_segments)
        return seg

    def _skill_scores(self, base: Dict[str, float], sys: Dict[str, float]) -> Dict[str, float]:
        skills: Dict[str, float] = {}
        for k in sys.keys():
            if k == "ece" and not self.cfg.include_ece_in_skill:
                continue
            denom = base.get(k, np.nan)
            num = sys.get(k, np.nan)
            if denom and denom > 0 and not np.isnan(denom) and not np.isnan(num):
                skills[f"skill_{k}"] = 1.0 - (num / denom)
            else:
                skills[f"skill_{k}"] = float("nan")
        return skills

    def evaluate(self, groups: Optional[pd.DataFrame] = None) -> Dict:
        P_cal = self._calibrate()
        self.P_cal = P_cal

        overall_raw = self._metrics(self.y, self.P_raw)
        overall_cal = self._metrics(self.y, P_cal) if P_cal is not None else None

        # Baseline: uniform
        K = self.P_raw.shape[1]
        base = np.full_like(self.P_raw, 1.0 / K)
        base_metrics = self._metrics(self.y, base)

        result: Dict = {
            "n": int(len(self.y)),
            "K": int(K),
            "overall_raw": overall_raw,
            "skill_raw": self._skill_scores(base_metrics, overall_raw),
        }
        if overall_cal is not None:
            result["overall_calibrated"] = overall_cal
            result["skill_calibrated"] = self._skill_scores(base_metrics, overall_cal)

        # Segments (kept out of JSON by default; always available for CSV)
        if groups is not None and not groups.empty:
            seg_raw = self._segment_metrics(self.y, self.P_raw, groups)
            result["_segments_raw_df"] = seg_raw
            if P_cal is not None:
                seg_cal = self._segment_metrics(self.y, P_cal, groups)
                result["_segments_cal_df"] = seg_cal
            if self.cfg.include_segments_in_json:
                result["by_segment_raw"] = seg_raw.to_dict(orient="records")
                if P_cal is not None:
                    result["by_segment_calibrated"] = seg_cal.to_dict(orient="records")

        # Round floats for JSON (exclude private keys)
        def _round_vals(obj):
            if isinstance(obj, dict):
                return {k: _round_vals(v) for k, v in obj.items() if not k.startswith("_")}
            if isinstance(obj, list):
                return [_round_vals(v) for v in obj]
            if isinstance(obj, float):
                return float(round(obj, self.cfg.summary_precision))
            return obj

        return _round_vals(result)

# ----------------------------
# I/O helpers
# ----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _maybe_write_csv(df: Optional[pd.DataFrame], path: str) -> None:
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(path, index=False)

# ----------------------------
# CLI & main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Poisson 1X2 forecasts from team lambdas")
    p.add_argument("--from-team-lambdas", dest="from_team_lambdas",
                   default="../data/calibrated/team_match_lambdas.parquet",
                   help="Path to parquet with per-team lambdas")
    p.add_argument("--outdir", default="../data/output/evaluated", help="Output directory")
    p.add_argument("--max-goals", type=int, default=12, help="Max goals for Poisson truncation")
    p.add_argument("--calibrate", choices=["none", "multinomial"], default="none",
                   help="Optional in-script calibration (diagnostic only)")
    p.add_argument("--n-bins", type=int, default=10, help="Bins for ECE")
    p.add_argument("--include-segments-in-json", action="store_true",
                   help="Embed by-segment tables inside summary.json (off by default)")
    p.add_argument("--max-segments", type=int, default=50,
                   help="When including segments in JSON, cap to this many rows")
    p.add_argument("--precision", type=int, default=6, help="Rounding precision in JSON")
    p.add_argument("--skill-include-ece", action="store_true",
                   help="Also compute skill for ECE (often not meaningful)")
    # Optional manual overrides (auto inference used if omitted)
    p.add_argument("--season", type=int, default=None, help="Override season (e.g., 2025)")
    p.add_argument("--mw", type=int, default=None, help="Override matchweek (e.g., 3)")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    _ensure_dir(args.outdir)

    # Build dataset
    match_df = build_from_team_lambdas(args.from_team_lambdas, max_goals=args.max_goals)

    # Historical subset (we evaluate only where result is known)
    hist = match_df.loc[match_df["is_historical"]].copy()
    if hist.empty:
        raise ValueError("No historical matches with goals found to evaluate.")

    # Labels & probs
    def _label(row) -> int:
        if row["home_goals"] > row["away_goals"]:
            return RESULT_MAP["H"]
        if row["home_goals"] == row["away_goals"]:
            return RESULT_MAP["D"]
        return RESULT_MAP["A"]

    y = hist.apply(_label, axis=1).to_numpy()
    P = hist[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)

    # Config / evaluator
    cfg = EvalConfig(
        K=3,
        n_bins=args.n_bins,
        group_cols=("home_team", "away_team"),
        calibrate=args.calibrate,
        include_segments_in_json=args.include_segments_in_json,
        max_segments=args.max_segments,
        summary_precision=args.precision,
        include_ece_in_skill=args.skill_include_ece,
    )
    ev = Evaluator(y, P, cfg)
    result = ev.evaluate(groups=hist[list(cfg.group_cols)].reset_index(drop=True))

    # ------- Filename stamping (AUTO unless overridden) -------
    auto_season, auto_mw = infer_next_mw_and_season(match_df)
    season_for_name = args.season if args.season is not None else auto_season
    mw_for_name = args.mw if args.mw is not None else auto_mw
    mw_for_name = min(38, max(1, int(mw_for_name)))  # clamp

    suffix = f"_epl_{season_for_name}_mw{mw_for_name:02d}"
    json_name    = f"summary{suffix}.json"
    seg_raw_name = f"segments_raw{suffix}.csv"
    seg_cal_name = f"segments_calibrated{suffix}.csv"

    # Write outputs
    json_path = os.path.join(args.outdir, json_name)
    _write_json(json_path, result)

    seg_raw_df = ev._segment_metrics(y, ev.P_raw, hist[list(cfg.group_cols)].reset_index(drop=True))
    _maybe_write_csv(seg_raw_df, os.path.join(args.outdir, seg_raw_name))

    seg_cal_df = None
    if ev.P_cal is not None:
        seg_cal_df = ev._segment_metrics(y, ev.P_cal, hist[list(cfg.group_cols)].reset_index(drop=True))
        _maybe_write_csv(seg_cal_df, os.path.join(args.outdir, seg_cal_name))

    cal_on = ev.P_cal is not None
    print(
        f"Wrote {json_path}\n"
        f"Auto-inferred: season={auto_season}, next_mw={auto_mw:02d}  |  Used: season={season_for_name}, mw={mw_for_name:02d}\n"
        f"Rows evaluated: {len(y)}  |  segments_raw present: {seg_raw_df is not None and not seg_raw_df.empty}\n"
        f"Calibrated (here): {cal_on}  |  segments_calibrated present: {cal_on and seg_cal_df is not None and not seg_cal_df.empty}"
    )

if __name__ == "__main__":
    main()
