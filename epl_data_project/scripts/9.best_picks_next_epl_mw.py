#!/usr/bin/env python3
# 9.best_picks_next_epl_mw.py
# Top 10 single picks + Top 5 accas for the next 7 days (UK time)
# Inputs (defaults):
#   ../data/raw/odds/odds.parquet
#   ../data/calibrated/market_prices.parquet       (optional λ for future fixtures)
#   ../data/calibrated/team_match_lambdas.parquet  (fallback λ ratings from history)
# Output:
#   ../data/output/picks/picks_next_epl_<season>_mw<NN>.json

import json, re, math, itertools
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd

# -------- paths / settings --------
PRICES_PARQUET = Path("../data/calibrated/market_prices_epl_2025_mw03.parquet")
ODDS_PARQUET   = Path("../data/raw/odds/epl_odds_2025_MW03.parquet")
LAMBDAS_PARQ   = Path("../data/calibrated/team_match_lambdas.parquet")
OUT_DIR        = Path("../data/output/picks")  # dynamic filename written here

DAYS_LOOKAHEAD = 7
CS_K_MAX       = 6         # correct-score grid (0..K)
DC_RHO         = -0.05     # Dixon–Coles tweak on CS grid
MIN_ACC_LEGS   = 2         # if few fixtures, allow doubles
PREF_ACC_LEGS  = 3         # prefer trebles; consider 4-folds if enough matches

# Canonical EPL teams (lowercase, canonicalized)
EPL_CANON = {
    "arsenal","astonvilla","bournemouth","brentford","brighton","chelsea","crystalpalace",
    "everton","fulham","ipswich","leicester","liverpool","manutd","mancity","newcastle",
    "southampton","tottenham","westham","wolves","forest","luton","sheffieldutd","westbrom"
}

# -------- helpers --------
def canon(s: str) -> str:
    s0 = (s or "").lower()
    s1 = re.sub(r'[^a-z0-9]+', '', s0)
    s1 = re.sub(r'fc$', '', s1)
    syn = {
        "manchesterunited":"manutd","manutd":"manutd","manchesterutd":"manutd",
        "manchestercity":"mancity","mancity":"mancity",
        "tottenhamhotspur":"tottenham","tottenham":"tottenham","spurs":"tottenham",
        "westhamunited":"westham","westham":"westham",
        "newcastleunited":"newcastle","newcastle":"newcastle",
        "brightonandhovealbion":"brighton","brightonhovealbion":"brighton","brighton":"brighton",
        "wolverhamptonwanderers":"wolves","wolverhampton":"wolves","wolves":"wolves",
        "nottinghamforest":"forest","nottmforest":"forest","forest":"forest",
        "sheffieldunited":"sheffieldutd","sheffieldutd":"sheffieldutd",
        "westbromwichalbion":"westbrom","westbrom":"westbrom",
        "lutontown":"luton","luton":"luton",
        "ipswichtown":"ipswich","ipswich":"ipswich",
        "leicestercity":"leicester","leicester":"leicester",
        "astonvillaa":"astonvilla","astonvilla":"astonvilla",
    }
    return syn.get(s1, s1)

def as_utc(ts):
    return pd.to_datetime(ts, utc=True, errors="coerce")

def iso_or_none(ts):
    if ts is None or pd.isna(ts): return None
    try: return pd.Timestamp(ts).isoformat()
    except Exception: return None

def pois_pmf_vec(k_max: int, lam: float) -> np.ndarray:
    out = np.zeros(k_max+1, dtype=float)
    out[0] = math.exp(-lam)
    for k in range(1, k_max+1):
        out[k] = out[k-1] * lam / k
    return out

def dc_tau(lh, la, rho, x, y):
    if x==0 and y==0: return 1 - (lh*la*rho)
    if x==1 and y==0: return 1 + (la*rho)
    if x==0 and y==1: return 1 + (lh*rho)
    if x==1 and y==1: return 1 - rho
    return 1.0

def cs_grid(lh: float, la: float, kmax: int, rho: float|None=None) -> np.ndarray:
    ph = pois_pmf_vec(kmax, lh)
    pa = pois_pmf_vec(kmax, la)
    cs = np.outer(ph, pa)
    if rho is not None and abs(rho) > 0:
        for (x,y) in [(0,0),(1,0),(0,1),(1,1)]:
            cs[x,y] *= dc_tau(lh, la, rho, x, y)
        s = cs.sum()
        if s > 0: cs = cs / s
    return cs

def totals_over(mu: float, line: float) -> float:
    k = int(math.floor(line))
    p0 = math.exp(-mu); cdf = p0; pk = p0
    for i in range(1, k+1):
        pk = pk * mu / i
        cdf += pk
    return max(0.0, min(1.0, 1 - cdf))

def prob_btts_yes(lh, la):
    return 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh+la))

def norm_1x2(oH, oD, oA):
    raw = np.vstack([1/np.asarray(oH,float), 1/np.asarray(oD,float), 1/np.asarray(oA,float)]).T
    s = raw.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return raw/s

def two_way_dejuice(o1, o2):
    a = 1/np.asarray(o1, float); b = 1/np.asarray(o2, float)
    s = a+b
    with np.errstate(invalid="ignore", divide="ignore"):
        return a/s, b/s

def kelly(odds, p):
    b = odds - 1.0; q = 1.0 - p
    f = (b*p - q) / b
    return float(f) if np.isfinite(f) and f > 0 else 0.0

# ---- ratings fallback for λ (from team_match_lambdas) ----
def build_team_ratings(lam_df: pd.DataFrame):
    home = lam_df[lam_df["home"]==1][["match_id","team","opp","lambda_glm"]].rename(
        columns={"team":"home_team","opp":"away_team","lambda_glm":"lambda_home"})
    away = lam_df[lam_df["home"]==0][["match_id","team","opp","lambda_glm"]].rename(
        columns={"team":"away_team_chk","opp":"home_team_chk","lambda_glm":"lambda_away"})
    hist = home.merge(away, on="match_id", how="inner")
    ok = (hist["home_team"]==hist["home_team_chk"]) & (hist["away_team"]==hist["away_team_chk"])
    hist = hist[ok].drop(columns=["home_team_chk","away_team_chk"]).copy()

    hist["home_key"] = hist["home_team"].apply(canon)
    hist["away_key"] = hist["away_team"].apply(canon)

    att_for = pd.concat([
        hist[["home_key","lambda_home"]].rename(columns={"home_key":"team","lambda_home":"lam_for"}),
        hist[["away_key","lambda_away"]].rename(columns={"away_key":"team","lambda_away":"lam_for"})
    ], ignore_index=True)
    def_against = pd.concat([
        hist[["home_key","lambda_away"]].rename(columns={"home_key":"team","lambda_away":"lam_against"}),
        hist[["away_key","lambda_home"]].rename(columns={"away_key":"team","lambda_home":"lam_against"})
    ], ignore_index=True)

    team_att = att_for.groupby("team")["lam_for"].mean()
    team_def = def_against.groupby("team")["lam_against"].mean()
    hfa = hist["lambda_home"].mean() / max(hist["lambda_away"].mean(), 1e-9)

    num = (hist["lambda_home"] + hist["lambda_away"]).sum()
    den = (team_att.reindex(hist["home_key"]).to_numpy() *
           team_def.reindex(hist["away_key"]).to_numpy() * hfa).sum() + \
          (team_att.reindex(hist["away_key"]).to_numpy() *
           team_def.reindex(hist["home_key"]).to_numpy()).sum()
    c = float(num / max(den, 1e-9))
    return team_att, team_def, hfa, c

def predict_lambdas(fixtures: pd.DataFrame, team_att, team_def, hfa, c):
    default_att = float(team_att.mean()); default_def = float(team_def.mean())
    def pred_pair(hk, ak):
        ah = team_att.get(hk, default_att); da = team_def.get(ak, default_def)
        aa = team_att.get(ak, default_att); dh = team_def.get(hk, default_def)
        lh = c * ah * da * hfa; la = c * aa * dh
        return max(lh, 1e-6), max(la, 1e-6)
    lams = fixtures.copy()
    lams[["lambda_home","lambda_away"]] = lams.apply(
        lambda r: pd.Series(pred_pair(r["home_key"], r["away_key"])), axis=1
    )
    return lams

def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if not s2.empty else None

# -------- matchweek inference (date-only, no APIs) --------
def _monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday-based week

def _season_year_for_date_uk(d: date) -> int:
    # Aug..Dec -> season = year; Jan..May -> year-1; (Jun/Jul assumed previous season)
    return d.year if d.month >= 8 else d.year - 1

def _season_start_monday(season_year: int) -> date:
    # MW1 = Monday on/after Aug 1 in the season year
    aug1 = date(season_year, 8, 1)
    return aug1 + timedelta(days=(7 - aug1.weekday()) % 7)

def infer_matchweek_from_dates(kickoffs_uk: pd.Series):
    """
    Returns (season_year, mw_mode, mw_min, mw_max).
    """
    dt = pd.to_datetime(kickoffs_uk, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("Europe/London")
    else:
        dt = dt.dt.tz_convert("Europe/London")
    dt = dt.dropna()
    if dt.empty:
        return None, None, None, None
    dates = dt.dt.date
    seasons = pd.Series(dates).map(_season_year_for_date_uk)
    season_year = int(seasons.mode().iloc[0])
    s0 = _season_start_monday(season_year)
    week_starts = pd.Series(dates).map(_monday_of_week)
    mw = ((week_starts - s0).map(lambda td: td.days // 7) + 1).astype(int)
    return season_year, int(mw.mode().iloc[0]), int(mw.min()), int(mw.max())

def main():
    if not ODDS_PARQUET.exists():
        raise FileNotFoundError(f"{ODDS_PARQUET} not found. Fetch odds first.")

    # Load odds and filter to next 7 days (UK time) & EPL teams
    odds = pd.read_parquet(ODDS_PARQUET).copy()
    odds["commence_time"] = as_utc(odds["commence_time"])
    odds["home_key"] = odds["home_team"].apply(canon)
    odds["away_key"] = odds["away_team"].apply(canon)

    # EPL filter: both teams recognized as EPL
    epl_mask = odds["home_key"].isin(EPL_CANON) & odds["away_key"].isin(EPL_CANON)
    odds = odds[epl_mask].copy()

    uk_now = pd.Timestamp.now(tz="Europe/London")
    start_uk = uk_now.normalize()
    end_uk   = (uk_now + pd.Timedelta(days=DAYS_LOOKAHEAD)).normalize() + pd.Timedelta(days=1)

    odds["kickoff_uk"] = odds["commence_time"].dt.tz_convert("Europe/London")
    in_window = (odds["kickoff_uk"] >= start_uk) & (odds["kickoff_uk"] < end_uk)
    nxt = odds[in_window].copy()

    # Robust names/KO per match_id from all rows in window
    names_by_match = (
        nxt.groupby("match_id", as_index=False)
           .agg(
               home_team=("home_team", first_non_null),
               away_team=("away_team", first_non_null),
               kickoff_utc=("commence_time", "min"),
               kickoff_uk=("kickoff_uk", "min"),
           )
    )
    keys_by_match = (
        nxt.groupby("match_id", as_index=False)
           .agg(
               home_key=("home_key", first_non_null),
               away_key=("away_key", first_non_null),
           )
    )
    fixtures = names_by_match.merge(keys_by_match, on="match_id", how="left")

    # ---------- infer EPL matchweek (for output naming) ----------
    season_year = mw_mode = mw_min = mw_max = None
    if not fixtures.empty and "kickoff_uk" in fixtures:
        season_year, mw_mode, mw_min, mw_max = infer_matchweek_from_dates(fixtures["kickoff_uk"])
    # dynamic output filename
    if season_year is not None and mw_mode is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_json = OUT_DIR / f"picks_next_epl_{season_year}_mw{mw_mode:02d}.json"
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_json = OUT_DIR / "picks_next_epl_mw.json"  # fallback

    if fixtures.empty:
        with open(out_json, "w") as f:
            json.dump({
                "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "window_uk": {"start": iso_or_none(start_uk), "end": iso_or_none(end_uk)},
                "epl_matchweek": {
                    "season": season_year, "mw_mode": mw_mode, "mw_range": [mw_min, mw_max] if mw_min is not None else None
                },
                "singles_count": 0, "accas_count": 0, "singles": [], "accas": []
            }, f, indent=2)
        print(f"[warn] No EPL fixtures in next {DAYS_LOOKAHEAD} days. Wrote empty {out_json}")
        return

    # λ: try market_prices (already contains future match_id rows) else fallback ratings from historical lambdas
    if PRICES_PARQUET.exists():
        prices = pd.read_parquet(PRICES_PARQUET).copy()
        have = {"match_id","lambda_home","lambda_away"}
        model = fixtures.merge(prices[list(have)], on="match_id", how="left")
        model = model.dropna(subset=["lambda_home","lambda_away"])
    else:
        model = pd.DataFrame()

    if model.empty:
        if not LAMBDAS_PARQ.exists():
            raise RuntimeError("No λ for upcoming fixtures (market_prices absent/empty; no team_match_lambdas fallback).")
        lam_hist = pd.read_parquet(LAMBDAS_PARQ).copy()
        team_att, team_def, hfa, c = build_team_ratings(lam_hist)
        # -------- FIXED LINE: use a LIST, not a set, for column selection --------
        model = predict_lambdas(fixtures[["match_id","home_key","away_key"]], team_att, team_def, hfa, c) \
                  .merge(fixtures.drop(columns=["home_key","away_key"]), on="match_id", how="left")

    # DIAG & ensure names/KO
    n_total = len(model)
    n_named = model["home_team"].notna().sum()
    print(f"[diag] upcoming matches with λ: {n_total} | with names: {n_named}")

    # ---------- top-of-market odds (in window only) ----------
    o = nxt  # shorthand

    # 1X2
    h2h = o[o["market"].isin(["h2h","h2h_3_way"]) & o["outcome"].isin(["home","draw","away"])].copy()
    if not h2h.empty:
        idx = h2h.groupby(["match_id","outcome"])["price"].idxmax()
        h2h_best = h2h.loc[idx, ["match_id","outcome","bookmaker_title","price"]]
        h2h_p = h2h_best.pivot(index="match_id", columns="outcome", values="price").rename(
            columns={"home":"bk_home","draw":"bk_draw","away":"bk_away"}).reset_index()
        h2h_b = h2h_best.pivot(index="match_id", columns="outcome", values="bookmaker_title").rename(
            columns={"home":"bk_home_book","draw":"bk_draw_book","away":"bk_away_book"}).reset_index()
        h2h_top = h2h_p.merge(h2h_b, on="match_id", how="inner")
    else:
        h2h_top = pd.DataFrame(columns=["match_id","bk_home","bk_draw","bk_away","bk_home_book","bk_draw_book","bk_away_book"])

    # BTTS
    btts_odds = o[o["market"].astype(str).str.lower().eq("btts") & o["outcome"].str.lower().isin(["yes","no"])].copy()
    btts_odds["outcome"] = btts_odds["outcome"].str.lower()
    if not btts_odds.empty:
        idx = btts_odds.groupby(["match_id","outcome"])["price"].idxmax()
        btts_best = btts_odds.loc[idx, ["match_id","outcome","bookmaker_title","price"]]
        btts_yes_df = btts_best[btts_best["outcome"]=="yes"].rename(
            columns={"price":"bk_btts_yes","bookmaker_title":"bk_btts_yes_book"})
        btts_no_df  = btts_best[btts_best["outcome"]=="no" ].rename(
            columns={"price":"bk_btts_no","bookmaker_title":"bk_btts_no_book"})
        btts_top = btts_yes_df.merge(btts_no_df, on="match_id", how="outer")[["match_id","bk_btts_yes","bk_btts_yes_book","bk_btts_no","bk_btts_no_book"]]
    else:
        btts_top = pd.DataFrame(columns=["match_id","bk_btts_yes","bk_btts_yes_book","bk_btts_no","bk_btts_no_book"])

    # Totals (all lines)
    tot = o[o["market"].astype(str).str.lower().eq("totals") & o["outcome"].isin(["over","under"])].copy()
    if not tot.empty:
        tot["point_f"] = pd.to_numeric(tot["point"], errors="coerce")
        tot = tot.dropna(subset=["point_f"])
        idx = tot.groupby(["match_id","point_f","outcome"])["price"].idxmax()
        tot_best = tot.loc[idx, ["match_id","point_f","outcome","bookmaker_title","price"]]
        tot_over = tot_best[tot_best["outcome"]=="over"].rename(columns={"price":"bk_over","bookmaker_title":"bk_over_book"})
        tot_under= tot_best[tot_best["outcome"]=="under"].rename(columns={"price":"bk_under","bookmaker_title":"bk_under_book"})
        tot_top  = tot_over.merge(tot_under, on=["match_id","point_f"], how="outer")
    else:
        tot_top = pd.DataFrame(columns=["match_id","point_f","bk_over","bk_over_book","bk_under","bk_under_book"])

    # Correct Score (market names vary; catch "correct score" / "correct_score" / "cs")
    mkt = o["market"].astype(str).str.lower()
    cs = o[mkt.str.contains(r"\b(?:correct[\s_]?score|cs)\b", regex=True, na=False)].copy()
    if not cs.empty:
        sc = cs["outcome"].astype(str).str.extract(r"(\d+)\D+(\d+)")
        sc.columns = ["sc_h", "sc_a"]
        cs = pd.concat([cs, sc], axis=1).dropna(subset=["sc_h","sc_a"]).copy()
        cs["sc_h"] = cs["sc_h"].astype(int); cs["sc_a"] = cs["sc_a"].astype(int)
        cs = cs[(cs["sc_h"] <= CS_K_MAX) & (cs["sc_a"] <= CS_K_MAX)]
        if cs.empty:
            cs_top = pd.DataFrame(columns=["match_id","sc_h","sc_a","bk_cs_book","bk_cs"])
        else:
            idx = cs.groupby(["match_id","sc_h","sc_a"])["price"].idxmax()
            cs_top = cs.loc[idx, ["match_id","sc_h","sc_a","bookmaker_title","price"]].rename(
                columns={"price":"bk_cs","bookmaker_title":"bk_cs_book"})
    else:
        cs_top = pd.DataFrame(columns=["match_id","sc_h","sc_a","bk_cs_book","bk_cs"])

    # ---------- assemble model + odds (names first), drop nameless ----------
    base = (model
            .merge(h2h_top, on="match_id", how="left")
            .merge(btts_top, on="match_id", how="left"))
    base = base.dropna(subset=["home_team","away_team","kickoff_utc","kickoff_uk"])

    # ---------- build candidates ----------
    candidates = []

    # 1X2 de-juice (if any)
    if not base.empty and {"bk_home","bk_draw","bk_away"}.issubset(base.columns):
        imp = norm_1x2(base["bk_home"], base["bk_draw"], base["bk_away"])
        base["impH_mkt"], base["impD_mkt"], base["impA_mkt"] = imp[:,0], imp[:,1], imp[:,2]

    for _, r in base.iterrows():
        lh, la = float(r["lambda_home"]), float(r["lambda_away"])
        grid = cs_grid(lh, la, CS_K_MAX, rho=DC_RHO)
        pD = float(np.trace(grid)); pH = float(np.triu(grid, 1).sum()); pA = float(np.tril(grid, -1).sum())
        p_btts = prob_btts_yes(lh, la)

        # 1X2
        for sel, p, o, book, imp_p in [
            ("Home", pH, r.get("bk_home"), r.get("bk_home_book"), r.get("impH_mkt")),
            ("Draw", pD, r.get("bk_draw"), r.get("bk_draw_book"), r.get("impD_mkt")),
            ("Away", pA, r.get("bk_away"), r.get("bk_away_book"), r.get("impA_mkt")),
        ]:
            if pd.notna(o) and float(o) > 1.0 and p is not None:
                o = float(o)
                candidates.append({
                    "match_id": r["match_id"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                    "market": "1X2", "selection": sel,
                    "model_prob": float(p),
                    "market_imp_prob": float(imp_p) if imp_p is not None and not pd.isna(imp_p) else None,
                    "edge": float(p - imp_p) if imp_p is not None and not pd.isna(imp_p) else None,
                    "kelly": kelly(o, float(p)),
                    "best_odds": o, "best_bookmaker": book,
                    "score": float(p*o - 1.0),
                })

        # BTTS
        y_odds, n_odds = r.get("bk_btts_yes"), r.get("bk_btts_no")
        y_book, n_book = r.get("bk_btts_yes_book"), r.get("bk_btts_no_book")
        imp_y = imp_n = None
        if pd.notna(y_odds) and pd.notna(n_odds):
            imp_y, imp_n = two_way_dejuice(y_odds, n_odds)
        if pd.notna(y_odds) and float(y_odds) > 1.0:
            o = float(y_odds)
            candidates.append({
                "match_id": r["match_id"],
                "home_team": r["home_team"], "away_team": r["away_team"],
                "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                "market": "BTTS", "selection": "Yes",
                "model_prob": float(p_btts),
                "market_imp_prob": float(imp_y) if imp_y is not None else None,
                "edge": float(p_btts - imp_y) if imp_y is not None else None,
                "kelly": kelly(o, float(p_btts)),
                "best_odds": o, "best_bookmaker": y_book,
                "score": float(p_btts*o - 1.0),
            })
        if pd.notna(n_odds) and float(n_odds) > 1.0:
            o = float(n_odds)
            p_no = 1 - p_btts
            candidates.append({
                "match_id": r["match_id"],
                "home_team": r["home_team"], "away_team": r["away_team"],
                "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                "market": "BTTS", "selection": "No",
                "model_prob": float(p_no),
                "market_imp_prob": float(imp_n) if imp_n is not None else None,
                "edge": float(p_no - imp_n) if imp_n is not None else None,
                "kelly": kelly(o, float(p_no)),
                "best_odds": o, "best_bookmaker": n_book,
                "score": float(p_no*o - 1.0),
            })

    # Totals (by line)
    if not tot_top.empty:
        tot_join = tot_top.merge(model[["match_id","home_team","away_team","kickoff_utc","kickoff_uk","lambda_home","lambda_away"]],
                                 on="match_id", how="inner")
        for _, r in tot_join.iterrows():
            lh, la = float(r["lambda_home"]), float(r["lambda_away"])
            mu = lh + la
            line = float(r["point_f"])
            p_over = totals_over(mu, line); p_under = 1 - p_over
            o_over, o_under = r.get("bk_over"), r.get("bk_under")
            b_over, b_under = r.get("bk_over_book"), r.get("bk_under_book")
            imp_over = imp_under = None
            if pd.notna(o_over) and pd.notna(o_under):
                imp_over, imp_under = two_way_dejuice(o_over, o_under)
            if pd.notna(o_over) and float(o_over) > 1.0:
                o = float(o_over)
                candidates.append({
                    "match_id": r["match_id"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                    "market": f"Totals {line}", "selection": f"Over {line}",
                    "model_prob": float(p_over),
                    "market_imp_prob": float(imp_over) if imp_over is not None else None,
                    "edge": float(p_over - imp_over) if imp_over is not None else None,
                    "kelly": kelly(o, float(p_over)),
                    "best_odds": o, "best_bookmaker": b_over,
                    "score": float(p_over*o - 1.0),
                })
            if pd.notna(o_under) and float(o_under) > 1.0:
                o = float(o_under)
                candidates.append({
                    "match_id": r["match_id"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                    "market": f"Totals {line}", "selection": f"Under {line}",
                    "model_prob": float(p_under),
                    "market_imp_prob": float(imp_under) if imp_under is not None else None,
                    "edge": float(p_under - imp_under) if imp_under is not None else None,
                    "kelly": kelly(o, float(p_under)),
                    "best_odds": o, "best_bookmaker": b_under,
                    "score": float(p_under*o - 1.0),
                })

    # Correct Score
    if not cs_top.empty:
        cs_join = cs_top.merge(model[["match_id","home_team","away_team","kickoff_utc","kickoff_uk","lambda_home","lambda_away"]],
                               on="match_id", how="inner")
        for _, r in cs_join.iterrows():
            lh, la = float(r["lambda_home"]), float(r["lambda_away"])
            grid = cs_grid(lh, la, CS_K_MAX, rho=DC_RHO)
            h, a = int(r["sc_h"]), int(r["sc_a"])
            if h <= CS_K_MAX and a <= CS_K_MAX:
                p = float(grid[h, a]); o = float(r["bk_cs"])
                if o > 1.0 and p > 0:
                    candidates.append({
                        "match_id": r["match_id"],
                        "home_team": r["home_team"], "away_team": r["away_team"],
                        "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                        "market": "Correct Score", "selection": f"{h}-{a}",
                        "model_prob": p,
                        "market_imp_prob": None, "edge": None,
                        "kelly": kelly(o, p),
                        "best_odds": o, "best_bookmaker": r.get("bk_cs_book"),
                        "score": p*o - 1.0,
                    })

    cand_df = pd.DataFrame(candidates)
    if not cand_df.empty:
        cand_df = cand_df.dropna(subset=["home_team","away_team","best_odds","model_prob"])
    cand_df = cand_df.sort_values(["score","kelly"], ascending=[False, False]).reset_index(drop=True)

    # -------- singles: top 10 --------
    top10 = cand_df.head(10).copy()

    # -------- accas: best leg per match, then combos --------
    best_by_match = (cand_df.sort_values(["match_id","score","kelly"], ascending=[True, False, False])
                            .groupby("match_id", as_index=False).first())
    legs = best_by_match.to_dict(orient="records")
    M = len(legs)

    accas = []
    if M >= MIN_ACC_LEGS:
        sizes = []
        if M >= PREF_ACC_LEGS: sizes.append(PREF_ACC_LEGS)
        if M >= 4: sizes.append(4)
        if not sizes: sizes = [min(M, MIN_ACC_LEGS)]

        combos = []
        for r in sizes:
            for combo in itertools.combinations(legs, r):
                probs = [c["model_prob"] for c in combo]
                odds  = [c["best_odds"]  for c in combo]
                if any(pd.isna(p) or pd.isna(o) or o <= 1.0 for p,o in zip(probs,odds)):
                    continue
                prod_p = float(np.prod(probs))
                prod_o = float(np.prod(odds))
                ev = prod_p * prod_o - 1.0
                combos.append({
                    "legs": combo, "legs_count": r,
                    "acca_prob": prod_p, "acca_odds": prod_o,
                    "score": ev
                })
        combos = sorted(combos, key=lambda x: (x["score"], x["acca_prob"]), reverse=True)
        for c in combos[:5]:
            accas.append({
                "legs_count": c["legs_count"],
                "acca_prob": c["acca_prob"],
                "acca_odds": c["acca_odds"],
                "score": c["score"],
                "legs": [
                    {
                        "home_team": l["home_team"], "away_team": l["away_team"],
                        "kickoff_uk": l["kickoff_uk"],
                        "market": l["market"], "selection": l["selection"],
                        "odds": l["best_odds"], "p": l["model_prob"],
                        "bookmaker": l["best_bookmaker"]
                    } for l in c["legs"]
                ]
            })

    # -------- write JSON --------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "window_uk": {"start": iso_or_none(start_uk), "end": iso_or_none(end_uk)},
        "epl_matchweek": {
            "season": season_year,
            "mw_mode": mw_mode,
            "mw_range": [mw_min, mw_max] if (mw_min is not None and mw_max is not None) else None
        },
        "singles_count": int(len(top10)),
        "accas_count": int(len(accas)),
        "singles": top10.to_dict(orient="records"),
        "accas": accas,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # -------- preview --------
    print(f"[done] singles={len(top10)} accas={len(accas)} → {out_json}")
    if len(top10):
        cols = ["home_team","away_team","market","selection","best_odds","model_prob","score","kelly","kickoff_uk"]
        print("\nTop 10 singles:")
        print(top10[cols].head(10).to_string(index=False))
    if len(accas):
        print("\nTop 5 accas (best EV):")
        for i, a in enumerate(accas, 1):
            legs_str = " | ".join([f'{l["home_team"]}–{l["away_team"]} {l["market"]} {l["selection"]} @{l["odds"]:.2f}'
                                   for l in a["legs"]])
            print(f"{i}. {a['legs_count']}-fold @ {a['acca_odds']:.2f} | EV={a['score']:+.3f} | {legs_str}")

if __name__ == "__main__":
    main()
