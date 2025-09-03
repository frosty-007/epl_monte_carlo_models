#!/usr/bin/env python3
# calibrate_lambda.py
"""
End-to-end Lambda Calibration:
- Reads all_matches.parquet, all_shots.parquet, understat_epl_players_since2020.parquet
- Builds features (form, low_rest, finishing_bias, cards_burden)
- Adds team/opponent fixed effects with ridge regularization (or a simple beginner GLM)
- Fits Poisson GLM with log(xG) offset
- Writes team_match_lambdas.parquet with calibrated lambdas (historical + future)

Dependencies: pandas, numpy, statsmodels
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="../data/raw/game_stats/all_matches.parquet",
                    help="Path to all_matches.parquet")
    ap.add_argument("--shots", default="../data/raw/game_stats/all_shots.parquet",
                    help="Path to all_shots.parquet")
    ap.add_argument("--players", default="../data/raw/player_stats/understat_epl_players_since2020.parquet",
                    help="Path to understat player parquet")
    ap.add_argument("--out", default="../data/calibrated/team_match_lambdas.parquet",
                    help="Output parquet for per-team lambdas")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge strength for fit_regularized (full model)")
    ap.add_argument("--ema_half_life", type=float, default=6.0,
                    help="Half-life (matches) for finishing bias EWMA")
    ap.add_argument("--anchor_team", type=str, default=None,
                    help="Team slug to drop from FE (optional)")
    ap.add_argument("--min_xg", type=float, default=1e-8,
                    help="Lower clip for xG before log offset")
    ap.add_argument("--sanity", action="store_true",
                    help="Print/save sanity checks after fitting")
    ap.add_argument("--simple", action="store_true",
                    help="Beginner mode: tiny GLM with home + form, xG offset")
    return ap.parse_args()


# ------------------------------
# Helpers
# ------------------------------
def slugify_title(name: str) -> str:
    """Understat title → slug (e.g., 'Manchester United' -> 'Manchester_United')."""
    return (
        str(name or "").strip()
        .replace(" ", "_")
        .replace("&", "and")
        .replace("'", "")
        .replace(".", "")
    )


def season_from_dt(dt: pd.Series) -> pd.Series:
    y = dt.dt.year.astype(int)
    return np.where(dt.dt.month >= 7, y, y - 1).astype(int)


def _coerce_int(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return np.nan


# ------------------------------
# Feature builders
# ------------------------------
def build_team_match(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()

    # Date/time
    dt_col = "datetime" if "datetime" in m.columns else ("date" if "date" in m.columns else None)
    if dt_col is None:
        raise ValueError("matches parquet must have 'datetime' or 'date' column")
    m["date"] = pd.to_datetime(m[dt_col], utc=True, errors="coerce")

    # team/opponent slugs & home flag
    for c in ("team", "opponent", "is_home"):
        if c not in m.columns:
            raise ValueError(f"matches parquet missing '{c}'")
    m["team"] = m["team"].astype(str)
    m["opp"] = m["opponent"].astype(str)
    m["home"] = m["is_home"].astype("boolean").fillna(False).astype(int)
    m["side"] = m["home"].map({1: "h", 0: "a"})

    # goals (support dict or flattened cols)
    def extract_ha(row):
        gh = ga = None
        g = row.get("goals", None)
        if isinstance(g, dict):
            gh = g.get("h", None)
            ga = g.get("a", None)
        if gh is None:
            for k in ("home_goals", "h_goals", "goals_h"):
                if k in row and pd.notna(row[k]):
                    gh = row[k]; break
        if ga is None:
            for k in ("away_goals", "a_goals", "goals_a"):
                if k in row and pd.notna(row[k]):
                    ga = row[k]; break
        return _coerce_int(gh), _coerce_int(ga)

    ha = m.apply(extract_ha, axis=1, result_type="expand")
    ha.columns = ["_gh", "_ga"]
    m = pd.concat([m, ha], axis=1)

    m["_gh"] = pd.to_numeric(m["_gh"], errors="coerce")
    m["_ga"] = pd.to_numeric(m["_ga"], errors="coerce")

    # team-perspective goals
    m["goals"] = m["_gh"].where(m["home"].eq(1), m["_ga"]).astype("Int64")
    m["opp_goals"] = m["_ga"].where(m["home"].eq(1), m["_gh"]).astype("Int64")

    # xG/xGA numeric (clip to avoid log(0))
    for col_src, col_dst in (("xG_num", "xg_raw"), ("xGA_num", "xga_raw")):
        if col_src not in m.columns:
            raise ValueError(f"matches parquet missing '{col_src}'")
        m[col_dst] = pd.to_numeric(m[col_src], errors="coerce").fillna(0).clip(lower=1e-12)

    # rolling form (optional in raw; default 0)
    m["form_att"] = pd.to_numeric(m.get("rolling_xg", 0), errors="coerce").fillna(0.0)
    m["form_def"] = -pd.to_numeric(m.get("rolling_xga", 0), errors="coerce").fillna(0.0)

    # fatigue/flags
    m["rest_days"] = pd.to_numeric(m.get("rest_days", 0), errors="coerce").fillna(0).clip(lower=0)
    m["low_rest"] = ((3 - m["rest_days"]).clip(lower=0)) / 3.0
    m["is_derby"] = m.get("is_derby", False).astype(bool).fillna(False).astype(int)
    m["euro_travel"] = m.get("euro_travel", False).astype(bool).fillna(False).astype(int)

    # season
    m["season"] = season_from_dt(m["date"])

    keep = ["id", "date", "season", "team", "opp", "home", "side",
            "goals", "opp_goals", "xg_raw", "xga_raw",
            "form_att", "form_def", "low_rest", "is_derby", "euro_travel"]
    m = m[keep].sort_values(["team", "date"]).reset_index(drop=True)
    # normalize id to string for merges with shots (which may come as str)
    m["id"] = m["id"].astype("Int64").astype(str)
    return m


def add_finishing_bias(tm: pd.DataFrame, shots: pd.DataFrame, ema_half_life: float) -> pd.DataFrame:
    if shots is None or len(shots) == 0:
        tm2 = tm.copy()
        tm2["finish_bias"] = 0.0
        return tm2

    s = shots.copy()
    # date column name in shots is usually "date"
    s["date"] = pd.to_datetime(s.get("date"), utc=True, errors="coerce")
    s["is_goal"] = (s.get("result") == "Goal").astype(int)
    s["xG"] = pd.to_numeric(s.get("xG"), errors="coerce").fillna(0.0)
    # side fields can be 'h_a' or 'team_side'
    if "h_a" in s.columns:
        s["side"] = s["h_a"].map({"h": "h", "a": "a"})
    elif "team_side" in s.columns:
        s["side"] = s["team_side"].map({"h": "h", "a": "a"})
    else:
        s["side"] = pd.NA
    s["match_id"] = s.get("match_id").astype(str)

    tm2 = tm.copy()
    tm2["id"] = tm2["id"].astype(str)

    # aggregate shot xG and goals by match+side
    agg = (s.groupby(["match_id", "side"], as_index=False)
             .agg(shots_xg=("xG", "sum"), shots_goals=("is_goal", "sum")))

    tm2 = tm2.merge(agg, left_on=["id", "side"], right_on=["match_id", "side"], how="left")
    tm2["shots_xg"] = pd.to_numeric(tm2["shots_xg"], errors="coerce").fillna(0.0)
    tm2["shots_goals"] = pd.to_numeric(tm2["shots_goals"], errors="coerce").fillna(0.0)
    tm2["fin_bias_match"] = tm2["shots_goals"] - tm2["shots_xg"]

    # EWMA of finishing bias (previous matches only)
    tm2 = tm2.sort_values(["team", "date"]).copy()
    alpha = 1.0 - 2.0 ** (-1.0 / max(ema_half_life, 1e-6))
    tm2["finish_bias"] = (
        tm2.groupby("team", group_keys=False)["fin_bias_match"]
           .apply(lambda s_: s_.shift().ewm(alpha=alpha, adjust=False).mean())
           .fillna(0.0)
    )

    tm2.drop(columns=["match_id"], inplace=True, errors="ignore")
    return tm2


def add_cards_burden(tm: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    # If players not available, default to 0
    if players is None or len(players) == 0:
        tm2 = tm.copy()
        tm2["cards_burden"] = 0.0
        return tm2

    p = players.copy()
    # build slug for players team_title to join with tm.team (slug)
    if "team_title" not in p.columns or "season" not in p.columns:
        tm2 = tm.copy()
        tm2["cards_burden"] = 0.0
        return tm2

    p["team_slug"] = p["team_title"].astype(str).map(slugify_title)
    # aggregate disciplinary (reds per game) by team-season
    disc = (p.groupby(["team_slug", "season"], as_index=False)
              .agg(reds=("red_cards", "sum"),
                   games=("games", "max")))
    disc["red_rate"] = (disc["reds"] / disc["games"]).replace([np.inf, np.nan], 0.0).clip(lower=0.0)
    # crude minutes-down scaling
    disc["cards_burden_season"] = (disc["red_rate"] * 35.0) / 90.0

    tm2 = tm.merge(disc[["team_slug", "season", "cards_burden_season"]],
                   left_on=["team", "season"], right_on=["team_slug", "season"], how="left")
    tm2["cards_burden"] = tm2["cards_burden_season"].fillna(0.0)
    return tm2.drop(columns=["team_slug", "cards_burden_season"])


def add_fixed_effects(tm: pd.DataFrame, anchor_team: str | None):
    team_fe = pd.get_dummies(tm["team"], prefix="t", dtype=float)
    opp_fe  = pd.get_dummies(tm["opp"],  prefix="o", dtype=float)

    if anchor_team is None:
        anchor_team = sorted(tm["team"].unique())[0]
    t_anchor = f"t_{anchor_team}"
    o_anchor = f"o_{anchor_team}"

    if t_anchor in team_fe.columns:
        team_fe = team_fe.drop(columns=[t_anchor])
    if o_anchor in opp_fe.columns:
        opp_fe = opp_fe.drop(columns=[o_anchor])

    return team_fe, opp_fe, anchor_team


# ------------------------------
# Models
# ------------------------------
def fit_glm_and_predict(tm: pd.DataFrame,
                        team_fe: pd.DataFrame,
                        opp_fe: pd.DataFrame,
                        alpha: float,
                        min_xg: float):
    feature_cols = ["home","form_att","form_def","finish_bias","cards_burden",
                    "low_rest","is_derby","euro_travel"]

    X_core = tm[feature_cols].fillna(0.0)
    X = pd.concat([sm.add_constant(X_core), team_fe, opp_fe], axis=1)

    tm["goals"] = pd.to_numeric(tm["goals"], errors="coerce")
    tm["xg_raw"] = pd.to_numeric(tm["xg_raw"], errors="coerce")

    # offset = log(xG)
    eps = max(1e-9, float(min_xg))
    offset_all = np.log(tm["xg_raw"].fillna(eps).clip(lower=eps).to_numpy())

    # train only where goals observed
    mask_train = tm["goals"].notna()
    y = tm.loc[mask_train, "goals"].astype(int).to_numpy()
    X_train = X.loc[mask_train]
    offset_train = offset_all[mask_train.values]

    glm = sm.GLM(y, X_train, family=sm.families.Poisson(), offset=offset_train)
    res = glm.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=200)

    # predict λ for all (historical + future)
    tm["lambda_glm"] = res.predict(X, offset=offset_all)
    return res, tm["lambda_glm"]


def fit_simple_glm(tm: pd.DataFrame):
    """
    Beginner GLM:
      log(lambda) = const + b_home*home + b1*form_att + b2*form_def + log(xg_raw)
      -> lambda = xg_raw * exp(const + ...)
    """
    df = tm.dropna(subset=["goals", "xg_raw"]).copy()
    df["log_xg"] = np.log(df["xg_raw"].clip(1e-12))

    model = smf.glm(
        formula="goals ~ home + form_att + form_def",
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_xg"],
    )
    res = model.fit()
    coef = res.params
    print("\n=== Core coefficients (log-scale) ===")
    print(coef)
    print("home % effect ~", float((np.exp(coef.get("home", 0.0)) - 1) * 100))

    tm.loc[df.index, "lambda_glm"] = res.predict(df)
    return res, tm["lambda_glm"]


# ------------------------------
# Diagnostics
# ------------------------------
def run_sanity_checks(tm: pd.DataFrame, out_path: str) -> None:
    print("\n=== Sanity checks ===")
    print({
        "mean_goals": float(tm["goals"].mean()),
        "mean_lambda": float(tm["lambda_glm"].mean()),
        "rows": int(len(tm)),
        "dupe_team_match_rows": int(tm.duplicated(["id","team"]).sum())
    })
    q = tm["lambda_glm"].quantile([.01, .05, .50, .95, .99]).to_dict()
    print({"lambda_quantiles": {str(k): float(v) for k, v in q.items()}})
    print({
        "home_mean_lambda": float(tm.loc[tm.home==1, "lambda_glm"].mean()),
        "away_mean_lambda": float(tm.loc[tm.home==0, "lambda_glm"].mean())
    })

    # decile calibration (historical only)
    hist = tm[tm["goals"].notna()].copy()
    if not hist.empty:
        hist["rank"] = hist["lambda_glm"].rank(method="first")
        hist["decile"] = pd.qcut(hist["rank"], 10, labels=False, duplicates="drop")
        cal = (hist.groupby("decile", as_index=False)
                    .agg(mean_lambda=("lambda_glm","mean"),
                         mean_goals=("goals","mean"),
                         n=("goals","size")))
        print("\nDecile calibration (mean λ vs mean goals):")
        print(cal.to_string(index=False))

        out_p = Path(out_path)
        cal_csv = out_p.with_name(out_p.stem + "_decile_calibration.csv")
        cal.to_csv(cal_csv, index=False)
        print(f"[saved] {cal_csv}")


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()

    # Load inputs (handle missing files safely)
    matches_path = Path(args.matches)
    shots_path   = Path(args.shots)
    players_path = Path(args.players)

    if not matches_path.exists():
        raise FileNotFoundError(f"matches not found: {matches_path}")
    matches = pd.read_parquet(matches_path)

    shots = pd.read_parquet(shots_path) if shots_path.exists() else pd.DataFrame()
    players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()

    # Build features
    tm = build_team_match(matches)
    tm = add_finishing_bias(tm, shots, args.ema_half_life)
    tm = add_cards_burden(tm, players)

    # Fixed effects (full model) or simple GLM
    if args.simple:
        print("[info] Running SIMPLE GLM (home + form; xG offset)…")
        res, _ = fit_simple_glm(tm)
        anchor = None
    else:
        team_fe, opp_fe, anchor = add_fixed_effects(tm, args.anchor_team)
        print(f"[info] Anchor FE team dropped: {anchor}")
        res, _ = fit_glm_and_predict(tm, team_fe, opp_fe, alpha=args.alpha, min_xg=args.min_xg)

    print(f"[info] Mean goals={tm['goals'].mean():.3f} | Mean lambda={tm['lambda_glm'].mean():.3f}")

    # Output (per-team rows, both historical & future)
    out = (tm[["id","date","team","opp","home","goals","xg_raw","lambda_glm"]]
           .rename(columns={"id":"match_id","xg_raw":"xg"}))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[done] Wrote {out_path} with {len(out):,} rows.")

    if args.sanity:
        run_sanity_checks(tm, str(out_path))
        print("[info] Sanity checks completed.")


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
