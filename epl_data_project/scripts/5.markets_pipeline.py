#!/usr/bin/env python3
# markets_pipeline.py — auto naming uses (a) earliest future cluster, else (b) TODAY(UK) season+week
from __future__ import annotations

import argparse
from pathlib import Path
import json
from datetime import date, timedelta
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple, Union
from scipy.stats import poisson, skellam

# ---------- Season / MW helpers ----------

def _season_year_for_date_uk(d: date) -> int:
    # EPL season rolls over in August
    return d.year if d.month >= 8 else d.year - 1

def _mw_for_date_uk(d: date, season_year: int) -> int:
    # Week counting from the first Monday on/after Aug 1 of season-year
    aug1 = date(season_year, 8, 1)
    start_monday = aug1 + timedelta(days=(7 - aug1.weekday()) % 7)
    monday_of = d - timedelta(days=d.weekday())
    return max(1, ((monday_of - start_monday).days // 7) + 1)

def annotate_season_mw(df: pd.DataFrame, time_col: str = "commence_time") -> pd.DataFrame:
    out = df.copy()
    if time_col in out.columns:
        t = pd.to_datetime(out[time_col], utc=True, errors="coerce")
        out["date_uk"] = t.dt.tz_convert("Europe/London").dt.date
    elif "kick_date" in out.columns:
        out["date_uk"] = pd.to_datetime(out["kick_date"], errors="coerce").dt.date
    else:
        out["date_uk"] = pd.NaT

    def _season_safe(x):
        return _season_year_for_date_uk(x) if pd.notna(pd.Timestamp(x)) else np.nan

    def _mw_safe(x):
        if pd.isna(x): return np.nan
        sy = _season_year_for_date_uk(x)
        return _mw_for_date_uk(x, sy)

    out["season"] = out["date_uk"].apply(_season_safe)
    out["mw_num"] = out["date_uk"].apply(_mw_safe)
    return out

def _mode_int(s: pd.Series) -> Optional[int]:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(int)
    if x.empty: return None
    return int(x.mode().iloc[0])

def choose_filename_tokens(
    df_in: pd.DataFrame,
    prefer_future: bool = True,
    force_season: Optional[int] = None,
    force_mw: Optional[int] = None,
) -> tuple[int, str]:
    """
    Pick (season, 'mwNN') with this priority:
      1) Overrides -> use them.
      2) If future rows exist and prefer_future -> use EARLIEST future cluster (min date_uk ± 4d),
         prefer 'round'/'matchweek' if present else computed mw.
      3) Otherwise anchor on TODAY(UK): use same-season rows within today ± 4d; prefer 'round'/'matchweek';
         if none, fall back to today's season + computed mw.
    """
    # Manual overrides
    if force_season is not None and force_mw is not None:
        return int(force_season), f"mw{int(force_mw):02d}"

    df = annotate_season_mw(df_in, time_col="commence_time")

    # If we have any future entries: name for the *upcoming* cluster
    if prefer_future and "is_future" in df.columns and df["is_future"].any():
        fut = df[df["is_future"] == True].copy()
        fut = fut.dropna(subset=["date_uk"])
        if not fut.empty:
            dmin = pd.Series(fut["date_uk"]).min()
            start = dmin - timedelta(days=4)
            end = dmin + timedelta(days=4)
            cluster = fut[(fut["date_uk"] >= start) & (fut["date_uk"] <= end)].copy()

            # Prefer official round/matchweek if present
            for round_col in ("round", "matchweek"):
                if round_col in cluster.columns:
                    rnd = _mode_int(cluster[round_col])
                    if rnd is not None:
                        seas = _mode_int(cluster["season"])
                        if seas is None:
                            # if missing, infer from earliest future date
                            seas = _season_year_for_date_uk(dmin)
                        return int(seas), f"mw{rnd:02d}"

            seas = _mode_int(cluster["season"]) or _season_year_for_date_uk(dmin)
            mw = _mode_int(cluster["mw_num"]) or _mw_for_date_uk(dmin, seas)
            return int(seas), f"mw{mw:02d}"

    # No future rows or prefer_future=False: anchor on TODAY (UK)
    today = pd.Timestamp.now(tz="Europe/London").date()
    season_today = _season_year_for_date_uk(today)
    start = today - timedelta(days=4)
    end = today + timedelta(days=4)

    ann = df.dropna(subset=["date_uk"]).copy()
    # constrain to SAME SEASON as today to avoid drifting to old seasons
    ann_same_season = ann[ann["season"] == season_today].copy()

    # Use cluster around today in current season if available
    cluster = ann_same_season[(ann_same_season["date_uk"] >= start) & (ann_same_season["date_uk"] <= end)].copy()
    if not cluster.empty:
        for round_col in ("round", "matchweek"):
            if round_col in cluster.columns:
                rnd = _mode_int(cluster[round_col])
                if rnd is not None:
                    return int(season_today), f"mw{rnd:02d}"
        mw = _mode_int(cluster["mw_num"])
        if mw is not None:
            return int(season_today), f"mw{mw:02d}"

    # Fallback: purely from today's calendar
    return int(season_today), f"mw{_mw_for_date_uk(today, season_today):02d}"

def _auto_out_path(anchor: Path, stem: str, season: int, mw_tag: str, ext: str = ".parquet") -> Path:
    outdir = anchor if (anchor.suffix == "" or anchor.is_dir()) else anchor.parent
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{stem}_epl_{season}_{mw_tag}{ext}"

# ---------- Core helpers (pairing, markets, blending, etc.) ----------

def pair_lambdas(df: pd.DataFrame) -> pd.DataFrame:
    need = {"match_id", "team", "opp", "home", "lambda_glm"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"team_match_lambdas missing columns: {missing}")

    # carry a kickoff time if present (your calibrated file has 'date')
    extra_time_cols = [c for c in ["datetime", "commence_time", "kickoff", "date"] if c in df.columns]

    cols = ["match_id", "team", "opp", "lambda_glm"] + extra_time_cols
    if "goals" in df.columns: cols += ["goals"]
    if "is_future" in df.columns: cols += ["is_future"]

    home = df[df.home == 1][cols].rename(
        columns={"team":"home_team","opp":"away_team","lambda_glm":"lambda_home","goals":"home_goals"}
    )
    away = df[df.home == 0][cols].rename(
        columns={"team":"away_team_chk","opp":"home_team_chk","lambda_glm":"lambda_away","goals":"away_goals"}
    )
    m = home.merge(away, on="match_id", how="inner", suffixes=("_h","_a"))

    if not ((m["home_team"] == m["home_team_chk"]) & (m["away_team"] == m["away_team_chk"])).all():
        bad = m.loc[(m["home_team"] != m["home_team_chk"]) | (m["away_team"] != m["away_team_chk"]),
                    ["match_id","home_team","home_team_chk","away_team","away_team_chk"]]
        raise ValueError("Home/away team names do not align:\n" + bad.head().to_string())

    # unify commence_time from any available time col (prefers 'date_h' from calibrator)
    kickoff = None
    for c in ["datetime_h","commence_time_h","kickoff_h","date_h","datetime_a","commence_time_a","kickoff_a","date_a"]:
        if c in m.columns:
            kickoff = m[c]; break
    if kickoff is not None:
        try:
            if str(kickoff.dtype).startswith(("int","float")):
                m["commence_time"] = pd.to_datetime(kickoff, unit="ms", utc=True, errors="coerce")
            else:
                m["commence_time"] = pd.to_datetime(kickoff, utc=True, errors="coerce")
        except Exception:
            m["commence_time"] = pd.NaT

    if "is_future_h" in m.columns or "is_future_a" in m.columns:
        m["is_future"] = m.get("is_future_h")
        if "is_future_a" in m.columns:
            m["is_future"] = m["is_future"].fillna(m["is_future_a"])
    elif "is_future" in m.columns:
        pass
    else:
        m["is_future"] = False

    m = m.drop(columns=[c for c in ["home_team_chk","away_team_chk","is_future_h","is_future_a"] if c in m.columns], errors="ignore")
    return m

def over_under_probs(mu: float, lines: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in lines:
        k_floor = int(np.floor(k))
        p_over = 1.0 - poisson.cdf(k_floor, mu)
        key = str(k).replace(".","_")
        out[f"over_{key}"] = float(p_over)
        out[f"under_{key}"] = float(1.0 - p_over)
    return out

def probs_1x2_dixon_coles(lh: float, la: float, rho: float = 0.0, max_goals: int = 10) -> Tuple[float,float,float]:
    i = np.arange(0, max_goals + 1)
    ph = poisson.pmf(i, lh); pa = poisson.pmf(i, la)
    cs = np.outer(ph, pa)
    tau00 = 1 - lh*la*rho; tau10 = 1 + la*rho; tau01 = 1 + lh*rho; tau11 = 1 - rho
    cs[0,0] *= tau00
    if max_goals >= 1: cs[1,0] *= tau10; cs[0,1] *= tau01; cs[1,1] *= tau11
    cs /= cs.sum()
    return float(np.tril(cs,-1).sum()), float(np.trace(cs)), float(np.triu(cs,1).sum())

def parse_ou_lines(s: Optional[str]) -> Tuple[float,...]:
    if not s: return (1.5, 2.5, 3.5)
    vals = [float(tok.strip()) for tok in s.split(",") if tok.strip()]
    vals = sorted(set(vals))
    return tuple(vals or (1.5, 2.5, 3.5))

def fair_decimal(p: float) -> float:
    return float("inf") if p <= 0 else float(1.0 / p)

def _canon_team(s: str) -> str:
    s0 = (s or "").lower()
    s1 = re.sub(r"[^a-z0-9]+","", s0); s1 = re.sub(r"fc$","", s1)
    syn = {
        "westhamunited":"westham","westham":"westham",
        "manchesterunited":"manutd","manutd":"manutd",
        "manchestercity":"mancity","mancity":"mancity",
        "tottenhamhotspur":"tottenham","tottenham":"tottenham",
        "newcastleunited":"newcastle","newcastle":"newcastle",
        "brightonandhovealbion":"brighton","brightonhovealbion":"brighton","brighton":"brighton",
        "wolverhamptonwanderers":"wolves","wolverhampton":"wolves","wolves":"wolves",
        "nottinghamforest":"forest","nottmforest":"forest","forest":"forest",
        "leicestercity":"leicester","leicester":"leicester",
        "sheffieldunited":"sheffieldutd","sheffieldutd":"sheffieldutd",
        "westbromwichalbion":"westbrom","westbrom":"westbrom",
        "everton":"everton","liverpool":"liverpool","arsenal":"arsenal","chelsea":"chelsea",
        "astonvilla":"astonvilla","crystalpalace":"crystalpalace","fulham":"fulham",
        "brentford":"brentford","bournemouth":"bournemouth","burnley":"burnley",
        "lutontown":"luton","luton":"luton","ipswichtown":"ipswich","ipswich":"ipswich",
    }
    return syn.get(s1, s1)

def top_of_market_1x2_from_odds(odds_path: Path) -> pd.DataFrame:
    odds = pd.read_parquet(odds_path)
    x = odds[odds["market"].isin(["h2h","h2h_3_way"]) & odds["outcome"].isin(["home","draw","away"])].copy()
    if x.empty:
        return pd.DataFrame(columns=["home_key","away_key","kick_date","bk_home","bk_draw","bk_away"])
    x["home_key"] = x["home_team"].apply(_canon_team)
    x["away_key"] = x["away_team"].apply(_canon_team)
    x["kick_date"] = pd.to_datetime(x["commence_time"], utc=True, errors="coerce").dt.date
    idx = x.groupby(["home_key","away_key","kick_date","outcome"])["price"].idxmax()
    best = x.loc[idx, ["home_key","away_key","kick_date","outcome","price"]]
    best = best.pivot(index=["home_key","away_key","kick_date"], columns="outcome", values="price").reset_index()
    best.columns.name = None
    best = best.rename(columns={"home":"bk_home","draw":"bk_draw","away":"bk_away"})
    return best

def norm_implied_from_decimal(oH: pd.Series, oD: pd.Series, oA: pd.Series) -> np.ndarray:
    raw = np.vstack([1/np.asarray(oH,float), 1/np.asarray(oD,float), 1/np.asarray(oA,float)]).T
    s = raw.sum(axis=1, keepdims=True); s[s==0]=1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return raw / s

def apply_recalibration_inplace(df: pd.DataFrame, recal_json: str) -> None:
    with open(recal_json, "r") as f:
        params = json.load(f)
    a, b = float(params["a"]), float(params["b"])
    for col in ("lambda_home","lambda_away"):
        lam = df[col].to_numpy().clip(1e-12)
        df[col] = np.exp(a + b * np.log(lam))
    df["mu_total"] = df["lambda_home"] + df["lambda_away"]

def probs_markets(lh: float, la: float, ou_lines: Sequence[float],
                  dc_rho: Optional[float] = None, dc_max_goals: int = 10) -> Dict[str, float]:
    lh = float(max(lh, 1e-12)); la = float(max(la, 1e-12))
    if dc_rho is not None and abs(dc_rho) > 0:
        pH,pD,pA = probs_1x2_dixon_coles(lh, la, rho=dc_rho, max_goals=dc_max_goals)
    else:
        pD = float(skellam.pmf(0, lh, la))
        pH = float(1.0 - skellam.cdf(0, lh, la))
        pA = float(skellam.cdf(-1, lh, la))
    p_btts = float(1.0 - np.exp(-lh) - np.exp(-la) + np.exp(-(lh + la)))
    mu_tot = lh + la
    ou = over_under_probs(mu_tot, ou_lines)
    if "over_2_5" in ou:
        ou["over2.5"] = ou["over_2_5"]; ou["under2.5"] = ou["under_2_5"]
    else:
        p_over25 = 1.0 - poisson.cdf(2, mu_tot)
        ou["over2.5"] = float(p_over25); ou["under2.5"] = float(1.0 - p_over25)
    return {"pH": pH, "pD": pD, "pA": pA, "BTTS": p_btts, "mu_total": float(mu_tot), **ou}

def _discover_fixtures_from_odds(odds_path: Path, days: int) -> pd.DataFrame:
    o = pd.read_parquet(odds_path).copy()
    o["commence_time"] = pd.to_datetime(o["commence_time"], utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC"); end_utc = now_utc + timedelta(days=days)
    o = o[(o["commence_time"] >= now_utc) & (o["commence_time"] <= end_utc)]
    fixtures = (o[["match_id","commence_time","home_team","away_team"]]
                .drop_duplicates("match_id")
                .rename(columns={"commence_time":"kickoff_utc"}))
    if fixtures.empty:
        return fixtures
    fixtures["kickoff_uk"] = fixtures["kickoff_utc"].dt.tz_convert("Europe/London")
    fixtures["home_key"] = fixtures["home_team"].apply(_canon_team)
    fixtures["away_key"] = fixtures["away_team"].apply(_canon_team)
    return fixtures

def _build_team_ratings(lam_df: pd.DataFrame):
    hist = pair_lambdas(lam_df)
    if hist.empty: raise ValueError("No historical matches available to build ratings.")
    hist["home_key"] = hist["home_team"].apply(_canon_team)
    hist["away_key"] = hist["away_team"].apply(_canon_team)
    att_for = pd.concat([
        hist[["home_key","lambda_home"]].rename(columns={"home_key":"team","lambda_home":"lam_for"}),
        hist[["away_key","lambda_away"]].rename(columns={"away_key":"team","lambda_away":"lam_for"}),
    ], ignore_index=True)
    def_against = pd.concat([
        hist[["home_key","lambda_away"]].rename(columns={"home_key":"team","lambda_away":"lam_against"}),
        hist[["away_key","lambda_home"]].rename(columns={"away_key":"team","lambda_home":"lam_against"}),
    ], ignore_index=True)
    team_att = att_for.groupby("team")["lam_for"].mean()
    team_def = def_against.groupby("team")["lam_against"].mean()
    hfa = hist["lambda_home"].mean() / max(hist["lambda_away"].mean(), 1e-9)
    num = (hist["lambda_home"] + hist["lambda_away"]).sum()
    den = (team_att.reindex(hist["home_key"]).to_numpy() * team_def.reindex(hist["away_key"]).to_numpy() * hfa).sum() + \
          (team_att.reindex(hist["away_key"]).to_numpy() * team_def.reindex(hist["home_key"]).to_numpy()).sum()
    c = float(num / max(den, 1e-9))
    return team_att, team_def, hfa, c

def _make_future_lambda_rows(fixtures: pd.DataFrame, team_att, team_def, hfa, c) -> pd.DataFrame:
    default_att = float(team_att.mean()); default_def = float(team_def.mean())
    def pred_pair(hk, ak):
        ah = team_att.get(hk, default_att); da = team_def.get(ak, default_def)
        aa = team_att.get(ak, default_att); dh = team_def.get(hk, default_def)
        lh = c * ah * da * hfa; la = c * aa * dh
        return max(lh, 1e-6), max(la, 1e-6)
    rows = []
    for _, r in fixtures.iterrows():
        lh, la = pred_pair(r["home_key"], r["away_key"])
        rows.append({"match_id": r["match_id"], "team": r["home_team"], "opp": r["away_team"],
                     "home": 1, "lambda_glm": lh, "commence_time": r["kickoff_utc"], "is_future": True})
        rows.append({"match_id": r["match_id"], "team": r["away_team"], "opp": r["home_team"],
                     "home": 0, "lambda_glm": la, "commence_time": r["kickoff_utc"], "is_future": True})
    return pd.DataFrame(rows)

def build_prices_df_from_lambdas(
    lambdas: Union[Path, pd.DataFrame],
    ou_lines: Sequence[float],
    recal_json: Optional[Path],
    dc_rho: Optional[float],
    dc_max_goals: int,
    odds_path: Optional[Path],
    blend_w: float,
) -> pd.DataFrame:
    df = pd.read_parquet(lambdas) if isinstance(lambdas, (str, Path)) else lambdas
    matches = pair_lambdas(df)

    matches["home_key"] = matches["home_team"].apply(_canon_team)
    matches["away_key"] = matches["away_team"].apply(_canon_team)
    matches["kick_date"] = pd.to_datetime(matches.get("commence_time"), utc=True, errors="coerce").dt.date

    # normalize ou lines
    if isinstance(ou_lines, str) or ou_lines is None:
        ou_lines = parse_ou_lines(ou_lines)
    else:
        ou_lines = tuple(ou_lines)

    rows: List[Dict] = []
    for _, r in matches.iterrows():
        lh, la = float(r.lambda_home), float(r.lambda_away)
        res = probs_markets(lh, la, ou_lines=ou_lines, dc_rho=dc_rho, dc_max_goals=dc_max_goals)
        row = {
            "match_id": r.match_id, "home_team": r.home_team, "away_team": r.away_team,
            "home_key": r.home_key, "away_key": r.away_key, "kick_date": r.kick_date,
            "commence_time": r.get("commence_time"), "lambda_home": lh, "lambda_away": la,
            "mu_total": res["mu_total"], "pH_model": res["pH"], "pD_model": res["pD"], "pA_model": res["pA"],
            "BTTS_yes": res["BTTS"], "is_future": bool(r.get("is_future", False)),
        }
        row["pH"], row["pD"], row["pA"] = row["pH_model"], row["pD_model"], row["pA_model"]
        for k, v in res.items():
            if k.startswith("over") or k.startswith("under_") or k in ("over2.5","under2.5"):
                row[k] = v
        rows.append(row)

    prices = pd.DataFrame(rows)

    if recal_json:
        apply_recalibration_inplace(prices, str(recal_json))
        for idx, r in prices.iterrows():
            res = probs_markets(float(r["lambda_home"]), float(r["lambda_away"]), ou_lines=ou_lines, dc_rho=dc_rho, dc_max_goals=dc_max_goals)
            prices.at[idx,"mu_total"] = res["mu_total"]
            prices.at[idx,"pH_model"] = res["pH"]; prices.at[idx,"pD_model"] = res["pD"]; prices.at[idx,"pA_model"] = res["pA"]
            prices.at[idx,"BTTS_yes"] = res["BTTS"]
            for k, v in res.items():
                if k.startswith("over") or k.startswith("under_") or k in ("over2.5","under2.5"):
                    prices.at[idx, k] = v
        prices[["pH","pD","pA"]] = prices[["pH_model","pD_model","pA_model"]]

    if odds_path is not None and blend_w > 0:
        tom = top_of_market_1x2_from_odds(odds_path)
        prices = prices.merge(tom, on=["home_key","away_key","kick_date"], how="left")
        mask = prices[["bk_home","bk_draw","bk_away"]].notna().all(axis=1)
        if mask.any():
            imp = norm_implied_from_decimal(prices.loc[mask,"bk_home"],
                                            prices.loc[mask,"bk_draw"],
                                            prices.loc[mask,"bk_away"])
            w = float(blend_w)
            prices.loc[mask,["pH_mkt","pD_mkt","pA_mkt"]] = imp
            prices.loc[mask,"pH"] = (1 - w) * prices.loc[mask,"pH_model"] + w * prices.loc[mask,"pH_mkt"]
            prices.loc[mask,"pD"] = (1 - w) * prices.loc[mask,"pD_model"] + w * prices.loc[mask,"pD_mkt"]
            prices.loc[mask,"pA"] = (1 - w) * prices.loc[mask,"pA_model"] + w * prices.loc[mask,"pA_mkt"]

    prices["odds_H"] = prices["pH"].apply(fair_decimal)
    prices["odds_D"] = prices["pD"].apply(fair_decimal)
    prices["odds_A"] = prices["pA"].apply(fair_decimal)
    prices["odds_BTTS_yes"] = prices["BTTS_yes"].apply(fair_decimal)
    for col in list(prices.columns):
        if col.startswith("over") or col.startswith("under_") or col in ("over2.5","under2.5"):
            prices[f"odds_{col}"] = prices[col].apply(fair_decimal)

    order = ["match_id","home_team","away_team","lambda_home","lambda_away","mu_total",
             "pH_model","pD_model","pA_model","pH","pD","pA",
             "odds_H","odds_D","odds_A","BTTS_yes","odds_BTTS_yes","is_future"]
    ou_cols = [c for c in prices.columns if c.startswith("over") or c.startswith("under_")
               or c in ("over2.5","under2.5","odds_over2.5","odds_under2.5")
               or c.startswith("odds_over_") or c.startswith("odds_under_")]
    extra = [c for c in ["bk_home","bk_draw","bk_away","pH_mkt","pD_mkt","pA_mkt",
                         "home_key","away_key","kick_date","commence_time"] if c in prices.columns]
    keep = [c for c in order if c in prices.columns] + extra + ou_cols
    return prices[keep]

# ---------- Subcommands ----------

def cmd_prices(args):
    lam_hist = pd.read_parquet(args.lambdas).copy()
    lam_hist["is_future"] = False

    if args.include_future and args.odds and Path(args.odds).exists():
        fixtures = _discover_fixtures_from_odds(args.odds, args.future_days)
        if not fixtures.empty:
            team_att, team_def, hfa, c = _build_team_ratings(lam_hist)
            fut = _make_future_lambda_rows(fixtures, team_att, team_def, hfa, c)
            lam_all = pd.concat([lam_hist, fut], ignore_index=True)
            print(f"[info] appended {len(fut)//2} future fixtures (next {args.future_days}d)")
        else:
            print("[warn] no upcoming fixtures discovered in odds; continuing with historical only")
            lam_all = lam_hist
    else:
        lam_all = lam_hist

    prices = build_prices_df_from_lambdas(
        lambdas=lam_all,
        ou_lines=args.ou_lines,
        recal_json=args.recal,
        dc_rho=args.dc_rho,
        dc_max_goals=args.dc_max_goals,
        odds_path=args.odds,
        blend_w=args.blend_w,
    )

    season_for_name, mw_tag = choose_filename_tokens(
        prices, prefer_future=True, force_season=args.force_season, force_mw=args.force_mw
    )
    out_path = _auto_out_path(args.out, stem="market_prices", season=season_for_name, mw_tag=mw_tag, ext=".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path, index=False)
    print(f"[done] prices → {out_path} ({len(prices)} matches)")
    print(prices.head(5).to_string(index=False))

def _brier_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - actual) ** 2, axis=1)))

def cmd_backtest(args):
    if args.prices and Path(args.prices).exists():
        prices = pd.read_parquet(args.prices)
    else:
        print("[info] building prices from lambdas for backtest…")
        prices = build_prices_df_from_lambdas(
            lambdas=args.lambdas,
            ou_lines=args.ou_lines,
            recal_json=args.recal,
            dc_rho=args.dc_rho,
            dc_max_goals=args.dc_max_goals,
            odds_path=args.odds,
            blend_w=args.blend_w,
        )

    lambdas = pd.read_parquet(args.lambdas)
    m = pair_lambdas(lambdas)
    if not {"home_goals","away_goals"}.issubset(m.columns):
        raise ValueError("Backtest needs actual goals in lambdas parquet (home_goals/away_goals).")

    eval_df = prices.merge(m[["match_id","home_goals","away_goals","lambda_home","lambda_away"]],
                           on="match_id", how="inner")

    hg = pd.to_numeric(eval_df["home_goals"], errors="coerce")
    ag = pd.to_numeric(eval_df["away_goals"], errors="coerce")
    eval_df = eval_df.loc[hg.notna() & ag.notna()].reset_index(drop=True)
    if eval_df.empty:
        raise ValueError("Backtest requires at least one completed match.")

    ll = poisson.logpmf(eval_df["home_goals"], eval_df["lambda_home"]) + \
         poisson.logpmf(eval_df["away_goals"], eval_df["lambda_away"])
    eval_df["loglik"] = ll

    diff = (eval_df["home_goals"] - eval_df["away_goals"]).to_numpy(dtype=float)
    cls_idx = (np.sign(diff) + 1).astype(int)  # {-1,0,1}->{0,1,2}
    oh = np.eye(3)[cls_idx]                    # [away,draw,home]
    actual = oh[:, [2,1,0]]                    # -> [home,draw,away]

    out_stats = {"matches": int(len(eval_df)),
                 "total_loglik": float(np.nansum(ll)),
                 "avg_loglik": float(np.nanmean(ll))}
    if {"pH_model","pD_model","pA_model"}.issubset(eval_df.columns):
        out_stats["brier_1x2_model"] = _brier_1x2(eval_df[["pH_model","pD_model","pA_model"]].to_numpy(), actual)
    if {"pH","pD","pA"}.issubset(eval_df.columns):
        out_stats["brier_1x2_current"] = _brier_1x2(eval_df[["pH","pD","pA"]].to_numpy(), actual)

    print(out_stats)

    season_for_name, mw_tag = choose_filename_tokens(prices, prefer_future=False,
                                                     force_season=args.force_season, force_mw=args.force_mw)
    out_path = _auto_out_path(args.out, stem="backtest_detail", season=season_for_name, mw_tag=mw_tag, ext=".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(out_path, index=False)
    print(f"[done] backtest details → {out_path}")

def cmd_simulate(args):
    prices = pd.read_parquet(args.prices)
    rng = np.random.default_rng(args.seed)
    rows = []
    for _, r in prices.iterrows():
        lh, la = float(r["lambda_home"]), float(r["lambda_away"])
        h = rng.poisson(lam=lh, size=args.n); a = rng.poisson(lam=la, size=args.n)
        rows.append({"match_id": r.match_id, "home_team": r.home_team, "away_team": r.away_team,
                     "pH_mc": float((h > a).mean()), "pD_mc": float((h == a).mean()), "pA_mc": float((h < a).mean()),
                     "avg_total_mc": float((h + a).mean())})
    sims = pd.DataFrame(rows)

    season_for_name, mw_tag = choose_filename_tokens(prices, prefer_future=True,
                                                     force_season=args.force_season, force_mw=args.force_mw)
    out_path = _auto_out_path(args.out, stem="mc_results", season=season_for_name, mw_tag=mw_tag, ext=".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sims.to_parquet(out_path, index=False)
    print(f"[done] simulate → {out_path} ({len(sims)} matches, n={args.n} per match)")

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Markets pipeline: prices | backtest | simulate (auto SEASON+MW naming anchored to TODAY or upcoming fixtures)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prices
    p = sub.add_parser("prices", help="Compute fair markets from lambdas (historical + optional future).")
    p.add_argument("--lambdas", type=Path, default=Path("../data/calibrated/team_match_lambdas.parquet"))
    p.add_argument("--out",     type=Path, default=Path("../data/calibrated/"),
                   help="Directory anchor; filename auto e.g. market_prices_epl_2025_mw03.parquet")
    p.add_argument("--ou-lines", type=str, default=None, help="Comma-separated O/U lines, e.g. '1.5,2.5,3.5'")
    p.add_argument("--recal",    type=Path, default=None, help="Path to recalibration JSON with {'a':...,'b':...}")
    p.add_argument("--dc-rho", type=float, default=-0.05, help="Dixon–Coles rho (omit or 0 for Skellam)")
    p.add_argument("--dc-max-goals", type=int, default=10)
    p.add_argument("--odds", type=Path, default=Path("../data/raw/odds/odds.parquet"),
                   help="Odds parquet; used for future fixture discovery and (optional) blending")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--include-future", dest="include_future", action="store_true")
    grp.add_argument("--no-include-future", dest="include_future", action="store_false")
    p.set_defaults(include_future=True)
    p.add_argument("--future-days", type=int, default=7)
    p.add_argument("--blend-w", type=float, default=0.0, help="Blend weight with market implied probs (0..1)")
    p.add_argument("--force-season", type=int, default=None)
    p.add_argument("--force-mw", type=int, default=None)
    p.set_defaults(func=cmd_prices)

    # backtest
    b = sub.add_parser("backtest", help="Backtest log-likelihood & 1X2 Brier (history only).")
    b.add_argument("--prices",  type=Path, default=None)
    b.add_argument("--lambdas", type=Path, default=Path("../data/calibrated/team_match_lambdas.parquet"))
    b.add_argument("--ou-lines", type=str, default=None)
    b.add_argument("--recal",    type=Path, default=None)
    b.add_argument("--dc-rho", type=float, default=None)
    b.add_argument("--dc-max-goals", type=int, default=10)
    b.add_argument("--odds", type=Path, default=None)
    b.add_argument("--blend-w", type=float, default=0.0)
    b.add_argument("--out", type=Path, default=Path("../data/calibrated/"),
                   help="Directory anchor; filename auto e.g. backtest_detail_epl_2025_mw03.parquet")
    b.add_argument("--force-season", type=int, default=None)
    b.add_argument("--force-mw", type=int, default=None)
    b.set_defaults(func=cmd_backtest)

    # simulate
    s = sub.add_parser("simulate", help="Monte Carlo per match.")
    s.add_argument("--prices", type=Path, default=Path("../data/calibrated/market_prices.parquet"))
    s.add_argument("--out",    type=Path, default=Path("../data/calibrated/"),
                   help="Directory anchor; filename auto e.g. mc_results_epl_2025_mw03.parquet")
    s.add_argument("-n", type=int, default=500000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--force-season", type=int, default=None)
    s.add_argument("--force-mw", type=int, default=None)
    s.set_defaults(func=cmd_simulate)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.func(args)
