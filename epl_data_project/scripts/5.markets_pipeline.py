#!/usr/bin/env python3
# markets_pipeline.py
# Commands:
#   prices   -> build fair markets from λ (Skellam 1X2 by default; optional Dixon–Coles; optional blend with market)
#   backtest -> exact-score log-likelihood + 1X2 Brier (builds prices if missing; prints model & blended if present)
#   simulate -> Monte Carlo per match
#
# Defaults assume your repo layout:
#   lambdas: ../data/callibrated/team_match_lambdas.parquet
#   prices:  ../data/callibrated/market_prices.parquet
#   sims:    ../data/callibrated/mc_results.parquet

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
from datetime import timedelta
import re
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from scipy.stats import poisson, skellam


# ---------- Core helpers ----------

def pair_lambdas(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per match with both lambdas (and goals/kickoff if present)."""
    need = {"match_id", "team", "opp", "home", "lambda_glm"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"team_match_lambdas missing columns: {missing}")

    # try to carry a kickoff time across if present
    extra_time_cols = [c for c in ["datetime", "commence_time", "kickoff", "date"] if c in df.columns]

    cols = ["match_id", "team", "opp", "lambda_glm"] + extra_time_cols
    add_goals = "goals" in df.columns
    if add_goals:
        cols += ["goals"]

    home = df[df.home == 1][cols].rename(
        columns={"team": "home_team", "opp": "away_team", "lambda_glm": "lambda_home", "goals": "home_goals"}
    )
    away = df[df.home == 0][cols].rename(
        columns={"team": "away_team_chk", "opp": "home_team_chk", "lambda_glm": "lambda_away", "goals": "away_goals"}
    )
    m = home.merge(away, on="match_id", how="inner", suffixes=("_h", "_a"))

    # sanity: names must align
    if not ((m["home_team"] == m["home_team_chk"]) & (m["away_team"] == m["away_team_chk"])).all():
        bad = m.loc[
            (m["home_team"] != m["home_team_chk"]) | (m["away_team"] != m["away_team_chk"]),
            ["match_id", "home_team", "home_team_chk", "away_team", "away_team_chk"],
        ]
        raise ValueError("Home/away team names do not align:\n" + bad.head().to_string())

    m = m.drop(columns=["home_team_chk", "away_team_chk"])

    # unify kickoff time if possible
    kickoff = None
    for c in ["datetime_h", "commence_time_h", "kickoff_h", "date_h",
              "datetime_a", "commence_time_a", "kickoff_a", "date_a"]:
        if c in m.columns:
            kickoff = m[c]
            break
    if kickoff is not None:
        # parse numbers (epoch ms) or ISO strings into UTC
        try:
            if str(kickoff.dtype).startswith(("int", "float")):
                m["commence_time"] = pd.to_datetime(kickoff, unit="ms", utc=True, errors="coerce")
            else:
                m["commence_time"] = pd.to_datetime(kickoff, utc=True, errors="coerce")
        except Exception:
            m["commence_time"] = pd.NaT

    return m


def over_under_probs(mu: float, lines: Sequence[float]) -> Dict[str, float]:
    """Return dict like {'over_2_5': p, 'under_2_5': 1-p, ...} for any k.5 lines."""
    out: Dict[str, float] = {}
    for k in lines:
        k_floor = int(np.floor(k))
        p_over = 1.0 - poisson.cdf(k_floor, mu)  # P(total >= k_floor+1) -> Over k.5
        key = str(k).replace(".", "_")
        out[f"over_{key}"] = float(p_over)
        out[f"under_{key}"] = float(1.0 - p_over)
    return out


def probs_1x2_dixon_coles(lh: float, la: float, rho: float = 0.0, max_goals: int = 10) -> Tuple[float, float, float]:
    """Dixon–Coles low-score adjustment for 1X2 using a finite correct-score grid."""
    i = np.arange(0, max_goals + 1)
    ph = poisson.pmf(i, lh)
    pa = poisson.pmf(i, la)
    cs = np.outer(ph, pa)  # independent correct-score

    # DC tau adjustments (Dixon & Coles, 1997) on four low-score cells
    tau00 = 1 - lh * la * rho
    tau10 = 1 + la * rho
    tau01 = 1 + lh * rho
    tau11 = 1 - rho

    cs[0, 0] *= tau00
    if max_goals >= 1:
        cs[1, 0] *= tau10
        cs[0, 1] *= tau01
        cs[1, 1] *= tau11

    cs = cs / cs.sum()  # renormalize
    pH = float(np.tril(cs, -1).sum())
    pD = float(np.trace(cs))
    pA = float(np.triu(cs, 1).sum())
    return pH, pD, pA


def parse_ou_lines(s: Optional[str]) -> Tuple[float, ...]:
    """Parse comma-separated OU lines; default to (1.5,2.5,3.5) if None."""
    if not s:
        return (1.5, 2.5, 3.5)
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    vals = sorted(set(vals))
    return tuple(vals or (1.5, 2.5, 3.5))


def fair_decimal(p: float) -> float:
    return float("inf") if p <= 0 else float(1.0 / p)


# ---------- Market blending + keys ----------

def _canon_team(s: str) -> str:
    s0 = (s or "").lower()
    s1 = re.sub(r"[^a-z0-9]+", "", s0)  # drop spaces/underscores/punct
    s1 = re.sub(r"fc$", "", s1)
    syn = {
        "westhamunited": "westham", "westham": "westham",
        "manchesterunited": "manutd", "manutd": "manutd",
        "manchestercity": "mancity", "mancity": "mancity",
        "tottenhamhotspur": "tottenham", "tottenham": "tottenham",
        "newcastleunited": "newcastle", "newcastle": "newcastle",
        "brightonandhovealbion": "brighton", "brightonhovealbion": "brighton", "brighton": "brighton",
        "wolverhamptonwanderers": "wolves", "wolverhampton": "wolves", "wolves": "wolves",
        "nottinghamforest": "forest", "nottmforest": "forest", "forest": "forest",
        "leicestercity": "leicester", "leicester": "leicester",
        "sheffieldunited": "sheffieldutd", "sheffieldutd": "sheffieldutd",
        "westbromwichalbion": "westbrom", "westbrom": "westbrom",
        "everton": "everton", "liverpool": "liverpool", "arsenal": "arsenal", "chelsea": "chelsea",
        "astonvilla": "astonvilla", "crystalpalace": "crystalpalace", "fulham": "fulham",
        "brentford": "brentford", "bournemouth": "bournemouth", "burnley": "burnley",
        "lutontown": "luton", "luton": "luton", "ipswichtown": "ipswich", "ipswich": "ipswich",
    }
    return syn.get(s1, s1)


def top_of_market_1x2_from_odds(odds_path: Path) -> pd.DataFrame:
    """
    From long-form odds (Odds API style), compute top-of-market 1X2 decimal odds
    per (home_key, away_key, kick_date).
    Expects columns: market in {'h2h','h2h_3_way'}, outcome in {'home','draw','away'},
    price, commence_time, home_team, away_team.
    """
    odds = pd.read_parquet(odds_path)
    x = odds[odds["market"].isin(["h2h", "h2h_3_way"]) & odds["outcome"].isin(["home", "draw", "away"])].copy()
    if x.empty:
        return pd.DataFrame(columns=["home_key", "away_key", "kick_date", "bk_home", "bk_draw", "bk_away"])
    x["home_key"] = x["home_team"].apply(_canon_team)
    x["away_key"] = x["away_team"].apply(_canon_team)
    x["kick_date"] = pd.to_datetime(x["commence_time"], utc=True, errors="coerce").dt.date
    idx = x.groupby(["home_key", "away_key", "kick_date", "outcome"])["price"].idxmax()
    best = x.loc[idx, ["home_key", "away_key", "kick_date", "outcome", "price"]]
    best = best.pivot(index=["home_key", "away_key", "kick_date"], columns="outcome", values="price").reset_index()
    best.columns.name = None
    best = best.rename(columns={"home": "bk_home", "draw": "bk_draw", "away": "bk_away"})
    return best


def norm_implied_from_decimal(oH: pd.Series, oD: pd.Series, oA: pd.Series) -> np.ndarray:
    """Row-wise normalized implied probabilities from decimal odds (remove overround)."""
    raw = np.vstack([1 / np.asarray(oH, float), 1 / np.asarray(oD, float), 1 / np.asarray(oA, float)]).T
    s = raw.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = raw / s
    return out


# ---------- Recalibration (optional) ----------

def fit_recalibration_from_history(lambdas_df: pd.DataFrame, out_json: str) -> Dict[str, float]:
    """
    Fit λ' = exp(a + b*log(λ)) using historical exact goals.
    Saves {"a": a, "b": b} to out_json and returns the dict.
    """
    m = pair_lambdas(lambdas_df)
    need = {"home_goals", "away_goals", "lambda_home", "lambda_away"}
    if not need.issubset(m.columns):
        raise ValueError("Need goals + lambdas in lambdas parquet to fit recalibration.")

    import statsmodels.api as sm  # optional; only needed if you call this

    lam = np.concatenate([m["lambda_home"].to_numpy(), m["lambda_away"].to_numpy()]).clip(1e-12)
    goals = np.concatenate([m["home_goals"].to_numpy(), m["away_goals"].to_numpy()]).astype(int)

    X = sm.add_constant(np.log(lam))
    res = sm.GLM(goals, X, family=sm.families.Poisson()).fit()
    a, b = map(float, res.params)
    params = {"a": a, "b": b}
    with open(out_json, "w") as f:
        json.dump(params, f)
    print(f"[recal] saved intercept/slope to {out_json}: a={a:.4f}, b={b:.4f}")
    return params


def apply_recalibration_inplace(df: pd.DataFrame, recal_json: str) -> None:
    """Modify df's lambda_home/away in place using params from recal_json."""
    with open(recal_json, "r") as f:
        params = json.load(f)
    a, b = float(params["a"]), float(params["b"])
    for col in ("lambda_home", "lambda_away"):
        lam = df[col].to_numpy().clip(1e-12)
        df[col] = np.exp(a + b * np.log(lam))
    df["mu_total"] = df["lambda_home"] + df["lambda_away"]


# ---------- Probabilities ----------

def probs_markets(
    lh: float,
    la: float,
    ou_lines: Sequence[float],
    dc_rho: Optional[float] = None,
    dc_max_goals: int = 10,
) -> Dict[str, float]:
    """
    Fair probs from independent Poisson with exact 1X2 via Skellam,
    unless Dixon–Coles rho is provided (then use DC-corrected 1X2).
    """
    lh = float(max(lh, 1e-12))
    la = float(max(la, 1e-12))

    # 1X2
    if dc_rho is not None and abs(dc_rho) > 0:
        pH, pD, pA = probs_1x2_dixon_coles(lh, la, rho=dc_rho, max_goals=dc_max_goals)
    else:
        pD = float(skellam.pmf(0, lh, la))
        pH = float(1.0 - skellam.cdf(0, lh, la))  # P(H-A > 0)
        pA = float(skellam.cdf(-1, lh, la))       # P(H-A < 0)

    # BTTS (independent Poisson)
    p_btts = float(1.0 - np.exp(-lh) - np.exp(-la) + np.exp(-(lh + la)))

    # Totals
    mu_tot = lh + la
    ou = over_under_probs(mu_tot, ou_lines)
    if "over_2_5" in ou:
        ou["over2.5"] = ou["over_2_5"]
        ou["under2.5"] = ou["under_2_5"]
    else:
        p_over25 = 1.0 - poisson.cdf(2, mu_tot)
        ou["over2.5"] = float(p_over25)
        ou["under2.5"] = float(1.0 - p_over25)

    return {"pH": pH, "pD": pD, "pA": pA, "BTTS": p_btts, "mu_total": float(mu_tot), **ou}


# ---------- NEW: ratings + future fixtures ----------

def _discover_fixtures_from_odds(odds_path: Path, days: int) -> pd.DataFrame:
    """Use odds as source of truth for upcoming fixtures (next N days)."""
    o = pd.read_parquet(odds_path).copy()
    o["commence_time"] = pd.to_datetime(o["commence_time"], utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC"); end_utc = now_utc + timedelta(days=days)
    o = o[(o["commence_time"] >= now_utc) & (o["commence_time"] <= end_utc)]
    fixtures = (o[["match_id", "commence_time", "home_team", "away_team"]]
                .drop_duplicates("match_id")
                .rename(columns={"commence_time": "kickoff_utc"}))
    if fixtures.empty:
        return fixtures
    fixtures["kickoff_uk"] = fixtures["kickoff_utc"].dt.tz_convert("Europe/London")
    fixtures["home_key"] = fixtures["home_team"].apply(_canon_team)
    fixtures["away_key"] = fixtures["away_team"].apply(_canon_team)
    return fixtures


def _build_team_ratings(lam_df: pd.DataFrame):
    """Create team attack/defence means + home-field factor + global scale from historical paired lambdas."""
    hist = pair_lambdas(lam_df)
    if hist.empty:
        raise ValueError("No historical matches available to build ratings.")
    hist["home_key"] = hist["home_team"].apply(_canon_team)
    hist["away_key"] = hist["away_team"].apply(_canon_team)

    # attack = avg λ_for; defence = avg λ_against
    att_for = pd.concat([
        hist[["home_key", "lambda_home"]].rename(columns={"home_key": "team", "lambda_home": "lam_for"}),
        hist[["away_key", "lambda_away"]].rename(columns={"away_key": "team", "lambda_away": "lam_for"}),
    ], ignore_index=True)
    def_against = pd.concat([
        hist[["home_key", "lambda_away"]].rename(columns={"home_key": "team", "lambda_away": "lam_against"}),
        hist[["away_key", "lambda_home"]].rename(columns={"away_key": "team", "lambda_home": "lam_against"}),
    ], ignore_index=True)

    team_att = att_for.groupby("team")["lam_for"].mean()
    team_def = def_against.groupby("team")["lam_against"].mean()

    hfa = hist["lambda_home"].mean() / max(hist["lambda_away"].mean(), 1e-9)

    # global scale so predicted totals match historical totals
    num = (hist["lambda_home"] + hist["lambda_away"]).sum()
    den = (team_att.reindex(hist["home_key"]).to_numpy() *
           team_def.reindex(hist["away_key"]).to_numpy() * hfa).sum() + \
          (team_att.reindex(hist["away_key"]).to_numpy() *
           team_def.reindex(hist["home_key"]).to_numpy()).sum()
    c = float(num / max(den, 1e-9))
    return team_att, team_def, hfa, c


def _make_future_lambda_rows(fixtures: pd.DataFrame, team_att, team_def, hfa, c) -> pd.DataFrame:
    """Two rows per fixture with λ, matching team_match_lambdas schema."""
    default_att = float(team_att.mean())
    default_def = float(team_def.mean())

    def pred_pair(hk, ak):
        ah = team_att.get(hk, default_att)
        da = team_def.get(ak, default_def)
        aa = team_att.get(ak, default_att)
        dh = team_def.get(hk, default_def)
        lh = c * ah * da * hfa
        la = c * aa * dh
        return max(lh, 1e-6), max(la, 1e-6)

    rows = []
    for _, r in fixtures.iterrows():
        lh, la = pred_pair(r["home_key"], r["away_key"])
        rows.append({
            "match_id": r["match_id"], "team": r["home_team"], "opp": r["away_team"],
            "home": 1, "lambda_glm": lh, "commence_time": r["kickoff_utc"], "is_future": True
        })
        rows.append({
            "match_id": r["match_id"], "team": r["away_team"], "opp": r["home_team"],
            "home": 0, "lambda_glm": la, "commence_time": r["kickoff_utc"], "is_future": True
        })
    return pd.DataFrame(rows)


# ---------- Build prices (with optional blending) ----------

def build_prices_df_from_lambdas(
    lambdas: Union[Path, pd.DataFrame],
    ou_lines: Sequence[float],
    recal_json: Optional[Path],
    dc_rho: Optional[float],
    dc_max_goals: int,
    odds_path: Optional[Path],
    blend_w: float,
) -> pd.DataFrame:
    if isinstance(lambdas, (str, Path)):
        df = pd.read_parquet(lambdas)
    else:
        df = lambdas
    matches = pair_lambdas(df)

    # keys for potential odds join
    matches["home_key"] = matches["home_team"].apply(_canon_team)
    matches["away_key"] = matches["away_team"].apply(_canon_team)
    if "commence_time" in matches.columns:
        matches["kick_date"] = pd.to_datetime(matches["commence_time"], utc=True, errors="coerce").dt.date
    else:
        matches["kick_date"] = pd.NaT

    # compute model markets
    rows: List[Dict] = []
    for _, r in matches.iterrows():
        lh, la = float(r.lambda_home), float(r.lambda_away)
        res = probs_markets(lh, la, ou_lines=ou_lines, dc_rho=dc_rho, dc_max_goals=dc_max_goals)
        row = {
            "match_id": r.match_id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_key": r.home_key,
            "away_key": r.away_key,
            "kick_date": r.kick_date,
            "commence_time": r.get("commence_time"),
            "lambda_home": lh,
            "lambda_away": la,
            "mu_total": res["mu_total"],
            # keep model-only 1X2 for diagnostics
            "pH_model": res["pH"], "pD_model": res["pD"], "pA_model": res["pA"],
            "BTTS_yes": res["BTTS"],
        }
        # default (may get blended below)
        row["pH"], row["pD"], row["pA"] = row["pH_model"], row["pD_model"], row["pA_model"]
        # add OU lines (model-only; blending only applies to 1X2)
        for k, v in res.items():
            if k.startswith("over") or k.startswith("under_") or k in ("over2.5", "under2.5"):
                row[k] = v
        rows.append(row)

    prices = pd.DataFrame(rows)

    # optional recalibration: adjust lambdas then recompute markets/odds (pre-blend)
    if recal_json:
        apply_recalibration_inplace(prices, str(recal_json))
        # recompute model probs with recalibrated λ
        for idx, r in prices.iterrows():
            res = probs_markets(float(r.lambda_home), float(r.lambda_away), ou_lines=ou_lines, dc_rho=dc_rho, dc_max_goals=dc_max_goals)
            prices.at[idx, "mu_total"] = res["mu_total"]
            prices.at[idx, "pH_model"] = res["pH"]; prices.at[idx, "pD_model"] = res["pD"]; prices.at[idx, "pA_model"] = res["pA"]
            prices.at[idx, "BTTS_yes"] = res["BTTS"]
            for k, v in res.items():
                if k.startswith("over") or k.startswith("under_") or k in ("over2.5", "under2.5"):
                    prices.at[idx, k] = v
        prices[["pH", "pD", "pA"]] = prices[["pH_model", "pD_model", "pA_model"]]

    # optional blend with market implied 1X2
    if odds_path is not None and blend_w > 0:
        tom = top_of_market_1x2_from_odds(odds_path)
        prices = prices.merge(tom, on=["home_key", "away_key", "kick_date"], how="left")
        mask = prices[["bk_home", "bk_draw", "bk_away"]].notna().all(axis=1)
        if mask.any():
            imp = norm_implied_from_decimal(prices.loc[mask, "bk_home"],
                                            prices.loc[mask, "bk_draw"],
                                            prices.loc[mask, "bk_away"])
            prices.loc[mask, "pH_mkt"] = imp[:, 0]
            prices.loc[mask, "pD_mkt"] = imp[:, 1]
            prices.loc[mask, "pA_mkt"] = imp[:, 2]

            w = float(blend_w)
            prices.loc[mask, "pH"] = (1 - w) * prices.loc[mask, "pH_model"] + w * prices.loc[mask, "pH_mkt"]
            prices.loc[mask, "pD"] = (1 - w) * prices.loc[mask, "pD_model"] + w * prices.loc[mask, "pD_mkt"]
            prices.loc[mask, "pA"] = (1 - w) * prices.loc[mask, "pA_model"] + w * prices.loc[mask, "pA_mkt"]

    # finally compute odds from current pH/pD/pA and OU/BTTS
    prices["odds_H"] = prices["pH"].apply(fair_decimal)
    prices["odds_D"] = prices["pD"].apply(fair_decimal)
    prices["odds_A"] = prices["pA"].apply(fair_decimal)
    prices["odds_BTTS_yes"] = prices["BTTS_yes"].apply(fair_decimal)
    for col in list(prices.columns):
        if col.startswith("over") or col.startswith("under_") or col in ("over2.5", "under2.5"):
            prices[f"odds_{col}"] = prices[col].apply(fair_decimal)

    # nice tidy ordering
    order = ["match_id", "home_team", "away_team", "lambda_home", "lambda_away", "mu_total",
             "pH_model", "pD_model", "pA_model", "pH", "pD", "pA",
             "odds_H", "odds_D", "odds_A", "BTTS_yes", "odds_BTTS_yes"]
    ou_cols = [c for c in prices.columns if c.startswith("over") or c.startswith("under_")
               or c in ("over2.5","under2.5","odds_over2.5","odds_under2.5")
               or c.startswith("odds_over_") or c.startswith("odds_under_")]
    extra = [c for c in ["bk_home","bk_draw","bk_away","pH_mkt","pD_mkt","pA_mkt",
                         "home_key","away_key","kick_date","commence_time"] if c in prices.columns]
    keep = [c for c in order if c in prices.columns] + extra + ou_cols
    prices = prices[keep]

    return prices


# ---------- Subcommands ----------

def cmd_prices(args):
    # load historical lambdas
    lam_hist = pd.read_parquet(args.lambdas).copy()
    lam_hist["is_future"] = False

    # optionally append future fixtures (DEFAULT: ON)
    if args.include_future and args.odds and Path(args.odds).exists():
        fixtures = _discover_fixtures_from_odds(args.odds, args.future_days)
        if not fixtures.empty:
            team_att, team_def, hfa, c = _build_team_ratings(lam_hist)
            fut = _make_future_lambda_rows(fixtures, team_att, team_def, hfa, c)
            lam_all = pd.concat([lam_hist, fut], ignore_index=True)
            print(f"[info] appended {len(fut)//2} future fixtures (next {args.future_days}d) from odds to lambdas")
        else:
            print("[warn] no upcoming fixtures discovered in odds; continuing with historical only")
            lam_all = lam_hist
    else:
        lam_all = lam_hist

    ou_lines = parse_ou_lines(args.ou_lines)
    prices = build_prices_df_from_lambdas(
        lambdas=lam_all,
        ou_lines=ou_lines,
        recal_json=args.recal,
        dc_rho=args.dc_rho,
        dc_max_goals=args.dc_max_goals,
        odds_path=args.odds,
        blend_w=args.blend_w,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(args.out, index=False)
    print(f"[done] prices → {args.out} ({len(prices)} matches)")
    print(prices.head(5).to_string(index=False))


def _brier_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - actual) ** 2, axis=1)))


def cmd_backtest(args):
    ou_lines = parse_ou_lines(args.ou_lines)

    # Build prices on the fly if not provided
    if args.prices and Path(args.prices).exists():
        prices = pd.read_parquet(args.prices)
    else:
        print("[info] building prices from lambdas for backtest…")
        prices = build_prices_df_from_lambdas(
            lambdas=args.lambdas,
            ou_lines=ou_lines,
            recal_json=args.recal,
            dc_rho=args.dc_rho,
            dc_max_goals=args.dc_max_goals,
            odds_path=args.odds,
            blend_w=args.blend_w,
        )

    lambdas = pd.read_parquet(args.lambdas)
    m = pair_lambdas(lambdas)

    need_goals = {"home_goals", "away_goals"}
    if not need_goals.issubset(m.columns):
        raise ValueError("Backtest needs actual goals in lambdas parquet (home_goals/away_goals).")

    # merge ONLY goals to avoid lambda column collisions
    eval_df = prices.merge(m[["match_id", "home_goals", "away_goals"]], on="match_id", how="inner")
    if not {"lambda_home", "lambda_away"}.issubset(eval_df.columns):
        # prices was from elsewhere? take lambdas from m
        eval_df = eval_df.merge(m[["match_id", "lambda_home", "lambda_away"]], on="match_id", how="left")

    # ---- Mask to completed matches BEFORE metrics (avoids NA issues) ----
    hg = pd.to_numeric(eval_df["home_goals"], errors="coerce")
    ag = pd.to_numeric(eval_df["away_goals"], errors="coerce")
    mask = hg.notna() & ag.notna()
    eval_df = eval_df.loc[mask].reset_index(drop=True)

    if eval_df.empty:
        raise ValueError("Backtest requires at least one completed match (home_goals/away_goals present).")

    # exact-score log-likelihood (independent Poisson on λ), on masked rows
    ll = poisson.logpmf(eval_df["home_goals"], eval_df["lambda_home"]) + \
         poisson.logpmf(eval_df["away_goals"], eval_df["lambda_away"])
    eval_df["loglik"] = ll

    # Vectorized one-hot outcome matrix [home, draw, away]
    diff = (eval_df["home_goals"] - eval_df["away_goals"]).to_numpy(dtype=float)
    cls_idx = (np.sign(diff) + 1).astype(int)  # {-1,0,1} -> {0,1,2}
    oh = np.eye(3)[cls_idx]                    # columns [away, draw, home]
    actual = oh[:, [2, 1, 0]]                  # reorder → [home, draw, away]

    out_stats = {
        "matches": int(len(eval_df)),
        "total_loglik": float(np.nansum(ll)),
        "avg_loglik": float(np.nanmean(ll)),
    }

    # model-only probs (if available)
    if {"pH_model", "pD_model", "pA_model"}.issubset(eval_df.columns):
        probs_model = eval_df[["pH_model", "pD_model", "pA_model"]].to_numpy()
        out_stats["brier_1x2_model"] = _brier_1x2(probs_model, actual)

    # current probs (possibly blended)
    if {"pH", "pD", "pA"}.issubset(eval_df.columns):
        probs_cur = eval_df[["pH", "pD", "pA"]].to_numpy()
        out_stats["brier_1x2_current"] = _brier_1x2(probs_cur, actual)

    # Baselines
    uniform = np.tile(np.array([1/3, 1/3, 1/3]), (len(eval_df), 1))
    out_stats["brier_1x2_uniform"] = _brier_1x2(uniform, actual)
    ph_emp = float((eval_df["home_goals"] > eval_df["away_goals"]).mean())
    pd_emp = float((eval_df["home_goals"] == eval_df["away_goals"]).mean())
    pa_emp = float(1.0 - ph_emp - pd_emp)
    league_const = np.tile(np.array([ph_emp, pd_emp, pa_emp]), (len(eval_df), 1))
    out_stats["brier_1x2_league"] = _brier_1x2(league_const, actual)
    out_stats["league_rates"] = {"H": ph_emp, "D": pd_emp, "A": pa_emp}

    print(out_stats)

    # always write backtest details
    args.out.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(args.out, index=False)
    print(f"[done] backtest details → {args.out}")


def cmd_simulate(args):
    prices = pd.read_parquet(args.prices)
    rng = np.random.default_rng(args.seed)
    rows = []
    for _, r in prices.iterrows():
        lh, la = float(r["lambda_home"]), float(r["lambda_away"])
        h = rng.poisson(lam=lh, size=args.n)
        a = rng.poisson(lam=la, size=args.n)
        rows.append({
            "match_id": r.match_id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "pH_mc": float((h > a).mean()),
            "pD_mc": float((h == a).mean()),
            "pA_mc": float((h < a).mean()),
            "avg_total_mc": float((h + a).mean()),
        })
    sims = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sims.to_parquet(args.out, index=False)
    print(f"[done] simulate → {args.out} ({len(sims)} matches, n={args.n} per match)")
    if {"pH", "pD", "pA"}.issubset(prices.columns):
        merged = prices.merge(sims, on=["match_id", "home_team", "away_team"])
        merged["diff_H"] = merged["pH_mc"] - merged["pH"]
        print(merged[["match_id", "home_team", "away_team", "pH", "pH_mc", "diff_H"]].head().to_string(index=False))


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Markets pipeline: prices | backtest | simulate (Skellam 1X2; optional Dixon–Coles; optional blend with market; optional recalibration)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prices
    p = sub.add_parser("prices", help="Compute fair markets from lambdas (historical + future).")
    p.add_argument("--lambdas", type=Path, default=Path("../data/callibrated/team_match_lambdas.parquet"))
    p.add_argument("--out",     type=Path, default=Path("../data/callibrated/market_prices.parquet"))
    p.add_argument("--ou-lines", type=str, default=None, help="Comma-separated O/U lines, e.g. '1.5,2.5,3.5'")
    p.add_argument("--recal",    type=Path, default=None, help="Path to recalibration JSON with {'a':..., 'b':...}")
    # Dixon–Coles options
    p.add_argument("--dc-rho", type=float, default=-0.05, help="Dixon–Coles rho (e.g., -0.05). If omitted, use Skellam.")
    p.add_argument("--dc-max-goals", type=int, default=10, help="Goal cap for DC grid (default 10)")
    # Odds / blending + FUTURE FIXTURES (DEFAULT ON)
    p.add_argument("--odds", type=Path, default=Path("../data/raw/odds/odds.parquet"),
                   help="Odds parquet (long-form). Used to discover upcoming fixtures and optional blending.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--include-future", dest="include_future", action="store_true",
                     help="Include next N days fixtures (default ON).")
    grp.add_argument("--no-include-future", dest="include_future", action="store_false",
                     help="Disable including future fixtures.")
    p.set_defaults(include_future=True)
    p.add_argument("--future-days", type=int, default=7, help="How many days ahead to include from odds (default 7).")
    p.add_argument("--blend-w", type=float, default=0.0, help="Blend weight with market implied probs (0..1). 0 = model only.")
    p.set_defaults(func=cmd_prices)

    # backtest
    b = sub.add_parser("backtest", help="Backtest log-likelihood & 1X2 Brier on history; prints model & current (blended) if present.")
    b.add_argument("--prices",  type=Path, default=None, help="If omitted, prices are built from lambdas")
    b.add_argument("--lambdas", type=Path, default=Path("../data/callibrated/team_match_lambdas.parquet"))
    b.add_argument("--ou-lines", type=str, default=None, help="Comma-separated O/U lines (when building prices)")
    b.add_argument("--recal",    type=Path, default=None, help="Apply recalibration when building prices")
    b.add_argument("--dc-rho", type=float, default=None, help="Dixon–Coles rho when building prices")
    b.add_argument("--dc-max-goals", type=int, default=10, help="Goal cap for DC grid")
    b.add_argument("--odds", type=Path, default=None, help="Odds parquet to enable blending during backtest build")
    b.add_argument("--blend-w", type=float, default=0.0, help="Blend weight with market probs during backtest build")
    b.add_argument("--out", type=Path, default=Path("../data/callibrated/backtest_detail.parquet"),
                   help="Detailed backtest parquet output (always written)")
    b.set_defaults(func=cmd_backtest)

    # simulate
    s = sub.add_parser("simulate", help="Monte Carlo per match")
    s.add_argument("--prices", type=Path, default=Path("../data/callibrated/market_prices.parquet"))
    s.add_argument("--out",    type=Path, default=Path("../data/callibrated/mc_results.parquet"))
    s.add_argument("-n", type=int, default=500000)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_simulate)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
