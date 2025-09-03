#!/usr/bin/env python3
# 9.best_picks_next_epl_mw.py
# Top 10 single picks + Top 5 accas for the *next EPL matchweek* (UK time)
# Markets output: 1X2, BTTS, Totals (O/U), Correct Score, Asian Handicap
#
# Inputs (defaults):
#   ../data/raw/odds/epl_odds.parquet                   (or update path below)
#   ../data/calibrated/market_prices.parquet           (optional λ for future fixtures)
#   ../data/calibrated/team_match_lambdas.parquet      (fallback λ ratings from history)
# Output:
#   ../data/output/picks/picks_next_epl_<season>_mw<NN>.json

import json, re, math, itertools, argparse
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd

# -------- paths / settings --------
PRICES_PARQUET = Path("../data/calibrated/market_prices.parquet")       # update if needed
ODDS_PARQUET   = Path("../data/raw/odds/epl_odds.parquet")              # update if needed
LAMBDAS_PARQ   = Path("../data/calibrated/team_match_lambdas.parquet")
OUT_DIR        = Path("../data/output/picks")  # dynamic filename written here

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

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser(description="Build picks for the NEXT EPL matchweek (or a specific one).")
    ap.add_argument("--mw", type=int, default=None, help="Matchweek number to force (1..38). If omitted, auto-detect next MW.")
    ap.add_argument("--season", type=int, default=None, help="Season start year (e.g., 2025). If omitted, inferred from odds.")
    ap.add_argument("--odds", type=Path, default=ODDS_PARQUET, help="Path to EPL odds parquet.")
    ap.add_argument("--market-prices", type=Path, default=PRICES_PARQUET, help="Path to market prices parquet (optional).")
    ap.add_argument("--lambdas", type=Path, default=LAMBDAS_PARQ, help="Fallback historical team lambdas parquet.")
    ap.add_argument("--outdir", type=Path, default=OUT_DIR, help="Output directory.")
    return ap.parse_args()

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

def gd_pmf_from_grid(grid: np.ndarray) -> dict[int,float]:
    k = grid.shape[0]-1
    pmf = {}
    for h in range(k+1):
        for a in range(k+1):
            gd = h - a
            pmf[gd] = pmf.get(gd, 0.0) + float(grid[h, a])
    return pmf

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

# -------- matchweek helpers (date-only, no external APIs) --------
def _monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday-based week

def _season_year_for_date_uk(d: date) -> int:
    # Aug..Dec -> season = year; Jan..May -> year-1; (Jun/Jul assumed previous season)
    return d.year if d.month >= 8 else d.year - 1

def _season_start_monday(season_year: int) -> date:
    # MW1 = Monday on/after Aug 1 in the season year
    aug1 = date(season_year, 8, 1)
    return aug1 + timedelta(days=(7 - aug1.weekday()) % 7)

def _mw_for_dt_uk(dt_uk: pd.Timestamp) -> tuple[int,int]:
    """Return (season_year, matchweek) for a UK-aware timestamp."""
    d = dt_uk.date()
    season_year = _season_year_for_date_uk(d)
    s0 = _season_start_monday(season_year)
    mw = (( _monday_of_week(d) - s0).days // 7) + 1
    return season_year, int(mw)

def infer_next_epl_matchweek_from_odds(odds_df: pd.DataFrame):
    """
    From all FUTURE EPL rows, compute (season_year, mw) per row and pick the earliest (season, mw)
    that has at least one fixture with kickoff >= now (UK).
    """
    now_uk = pd.Timestamp.now(tz="Europe/London")
    fut = odds_df[odds_df["kickoff_uk"] >= now_uk].copy()
    if fut.empty:
        return None, None
    tmp = fut[["match_id","kickoff_uk"]].drop_duplicates().copy()
    tmp[["season","mw"]] = tmp["kickoff_uk"].apply(lambda t: pd.Series(_mw_for_dt_uk(t)))
    # sort by kickoff then pick the earliest (season, mw)
    order = tmp.sort_values("kickoff_uk")
    first_season, first_mw = int(order.iloc[0]["season"]), int(order.iloc[0]["mw"])
    # But ensure we pick the minimum (season, mw) pair among future fixtures
    grouped = tmp.groupby(["season","mw"], as_index=False)["match_id"].nunique()
    grouped = grouped.sort_values(["season","mw"])
    for _, r in grouped.iterrows():
        season_i, mw_i = int(r["season"]), int(r["mw"])
        # accept the first (season, mw) that has any future fixture
        subset = tmp[(tmp["season"]==season_i) & (tmp["mw"]==mw_i)]
        if not subset.empty:
            return season_i, mw_i
    return first_season, first_mw

# ---------- AH math helpers ----------
def _prob_win_push_loss_for_line(pmf_gd: dict[int,float], line: float) -> tuple[float,float,float]:
    eps = 1e-12
    p_win = 0.0
    p_push = 0.0
    for gd, p in pmf_gd.items():
        v = gd + line
        if v > eps:
            p_win += p
        elif abs(v) <= eps and float(line).is_integer():
            p_push += p
    p_loss = max(0.0, 1.0 - p_win - p_push)
    return p_win, p_push, p_loss

def _quarter_to_halves(line: float):
    # treat x.25/x.75 as half to adjacent 0.0/0.5 steps
    if abs(line*2 - round(line*2)) > 1e-9 and abs(line*4 - round(line*4)) < 1e-9:
        sgn = 1 if line >= 0 else -1
        base = math.floor(abs(line)*2)/2 * sgn   # nearest towards zero at 0.5 step
        return base, base + 0.5*sgn
    return None

def _ah_ev_and_probs(pmf_gd: dict[int,float], line: float, odds: float) -> dict:
    halves = _quarter_to_halves(line)
    if halves is None:
        pW, pP, pL = _prob_win_push_loss_for_line(pmf_gd, line)
        ev = pW*(odds-1.0) - pL
    else:
        pW = pP = pL = ev = 0.0
        for L in halves:
            w, p, l = _prob_win_push_loss_for_line(pmf_gd, L)
            pW += 0.5*w; pP += 0.5*p; pL += 0.5*l
            ev += 0.5*(w*(odds-1.0) - l)
    p_equiv = max(0.0, min(1.0, (ev + 1.0) / max(odds, 1e-9)))
    return {"ev": ev, "p_full_win": pW, "p_push": pP, "p_loss": pL, "p_equiv": p_equiv}

# ================================
# ============ MAIN ==============
# ================================
def main():
    args = parse_args()

    if not args.odds.exists():
        raise FileNotFoundError(f"{args.odds} not found. Provide EPL odds parquet with markets.")

    # Load odds, canonicalize, keep EPL teams only
    odds = pd.read_parquet(args.odds).copy()
    odds["commence_time"] = as_utc(odds["commence_time"])
    odds["home_key"] = odds["home_team"].apply(canon)
    odds["away_key"] = odds["away_team"].apply(canon)
    odds["kickoff_uk"] = odds["commence_time"].dt.tz_convert("Europe/London")

    epl_mask = odds["home_key"].isin(EPL_CANON) & odds["away_key"].isin(EPL_CANON)
    odds = odds[epl_mask].copy()

    # Identify target (season, mw)
    if args.mw is not None:
        # Force season/mw (infer season if missing)
        if args.season is None:
            # Pick the soonest fixture >= now and compute its season
            now_uk = pd.Timestamp.now(tz="Europe/London")
            fut = odds[odds["kickoff_uk"] >= now_uk].copy()
            if fut.empty:
                raise RuntimeError("No future EPL fixtures found to infer season.")
            first_season = _mw_for_dt_uk(fut.sort_values("kickoff_uk").iloc[0]["kickoff_uk"])[0]
            season_year = int(first_season)
        else:
            season_year = int(args.season)
        target_season, target_mw = season_year, int(args.mw)
    else:
        # Auto-detect next MW from odds
        season_year, mw_auto = infer_next_epl_matchweek_from_odds(odds)
        if season_year is None or mw_auto is None:
            raise RuntimeError("Unable to infer next EPL matchweek from odds.")
        target_season, target_mw = int(season_year), int(mw_auto)

    # Slice odds to the target matchweek only
    meta = odds[["match_id","kickoff_uk","home_team","away_team","home_key","away_key","commence_time"]].drop_duplicates("match_id")
    # Compute (season, mw) per match_id
    tmp = meta.copy()
    tmp[["season","mw"]] = tmp["kickoff_uk"].apply(lambda t: pd.Series(_mw_for_dt_uk(t)))
    pick_ids = tmp[(tmp["season"]==target_season) & (tmp["mw"]==target_mw)]["match_id"].unique().tolist()

    if not pick_ids:
        raise RuntimeError(f"No EPL fixtures found for season={target_season}, mw={target_mw:02d} in odds file.")

    nxt = odds[odds["match_id"].isin(pick_ids)].copy()

    # Names/KO per match_id
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

    # λ: try market_prices (future rows by match_id) else fallback ratings from historical lambdas
    PRICES = args.market_prices
    if PRICES.exists():
        prices = pd.read_parquet(PRICES).copy()
        have = {"match_id","lambda_home","lambda_away"}
        model = fixtures.merge(prices[list(have)], on="match_id", how="left")
        model = model.dropna(subset=["lambda_home","lambda_away"])
    else:
        model = pd.DataFrame()

    if model.empty:
        LAMB = args.lambdas
        if not LAMB.exists():
            raise RuntimeError("No λ for upcoming fixtures (market_prices absent/empty; no team_match_lambdas fallback).")
        lam_hist = pd.read_parquet(LAMB).copy()
        team_att, team_def, hfa, c = build_team_ratings(lam_hist)
        model = predict_lambdas(fixtures[["match_id","home_key","away_key"]], team_att, team_def, hfa, c) \
                  .merge(fixtures.drop(columns=["home_key","away_key"]), on="match_id", how="left")

    # ---------- gather best odds per market ----------
    o = nxt  # shorthand
    mkt_lower = o["market"].astype(str).str.lower()

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
    btts_odds = o[mkt_lower.eq("btts") & o["outcome"].str.lower().isin(["yes","no"])].copy()
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

    # Totals
    tot = o[mkt_lower.eq("totals") & o["outcome"].isin(["over","under"])].copy()
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

    # Correct Score
    cs_m = mkt_lower.str.contains(r"\b(?:correct[\s_]?score|cs)\b", regex=True, na=False)
    cs = o[cs_m].copy()
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

    # Asian Handicap
    ah_mask = mkt_lower.isin(["spreads","asian_handicap","asian handicap","handicap","ah","asian lines","asian"])
    ah = o[ah_mask & o["outcome"].isin(["home","away"])].copy()
    if not ah.empty:
        ah["point_f"] = pd.to_numeric(ah["point"], errors="coerce")
        ah = ah.dropna(subset=["point_f"])
        idx = ah.groupby(["match_id","point_f","outcome"])["price"].idxmax()
        ah_top = ah.loc[idx, ["match_id","point_f","outcome","bookmaker_title","price"]]
    else:
        ah_top = pd.DataFrame(columns=["match_id","point_f","outcome","bookmaker_title","price"])

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
        pmf_gd = gd_pmf_from_grid(grid)
        pD = float(np.trace(grid)); pH = float(np.triu(grid, 1).sum()); pA = float(np.tril(grid, -1).sum())
        p_btts = prob_btts_yes(lh, la)

        # 1X2
        for sel, p, o_, book, imp_p in [
            ("Home", pH, r.get("bk_home"), r.get("bk_home_book"), r.get("impH_mkt")),
            ("Draw", pD, r.get("bk_draw"), r.get("bk_draw_book"), r.get("impD_mkt")),
            ("Away", pA, r.get("bk_away"), r.get("bk_away_book"), r.get("impA_mkt")),
        ]:
            if pd.notna(o_) and float(o_) > 1.0 and p is not None:
                o = float(o_)
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

    # Asian Handicap
    if not ah_top.empty:
        ah_join = ah_top.merge(model[["match_id","home_team","away_team","kickoff_utc","kickoff_uk","lambda_home","lambda_away"]],
                               on="match_id", how="inner")
        for _, r in ah_join.iterrows():
            lh, la = float(r["lambda_home"]), float(r["lambda_away"])
            grid = cs_grid(lh, la, CS_K_MAX, rho=DC_RHO)
            pmf_gd = gd_pmf_from_grid(grid)

            line = float(r["point_f"])
            outcome = str(r["outcome"]).lower()  # "home" or "away"
            odds_ = float(r["price"])
            book = r.get("bookmaker_title")

            # For away side, reflect GD distribution
            pmf = pmf_gd if outcome=="home" else {k: pmf_gd.get(-k, 0.0) for k in pmf_gd}
            res = _ah_ev_and_probs(pmf, line, odds_)
            ev = res["ev"]; p_equiv = res["p_equiv"]
            sel_txt = f'{"Home" if outcome=="home" else "Away"} {line:+g}'
            candidates.append({
                "match_id": r["match_id"],
                "home_team": r["home_team"], "away_team": r["away_team"],
                "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                "market": "Asian Handicap",
                "selection": sel_txt,
                "line": line,
                "model_prob": None,
                "p_full_win": res["p_full_win"],
                "p_push": res["p_push"],
                "p_loss": res["p_loss"],
                "kelly": kelly(odds_, p_equiv),
                "best_odds": odds_,
                "best_bookmaker": book,
                "score": ev,
            })

    # Collect candidates
    cand_df = pd.DataFrame(candidates)
    if not cand_df.empty:
        cand_df = cand_df.dropna(subset=["home_team","away_team","best_odds"])
    cand_df = cand_df.sort_values(["score","kelly"], ascending=[False, False]).reset_index(drop=True)

    # -------- PER-GAME MARKET PICKS (1X2, BTTS, Totals, Correct Score, Asian Handicap) --------
    def _as_float(x):
        try: return float(x)
        except Exception: return None

    def _mk_sel_dict(row, include_line=False, ah=False):
        d = {
            "market": row.get("market"),
            "selection": row.get("selection"),
            "odds": _as_float(row.get("best_odds")),
            "p": _as_float(row.get("model_prob")),
            "edge": _as_float(row.get("edge")),
            "kelly": _as_float(row.get("kelly")),
            "ev": _as_float(row.get("score")),
            "bookmaker": row.get("best_bookmaker"),
        }
        if include_line:
            d["line"] = _as_float(row.get("line"))
        if ah:
            d["p_full_win"] = _as_float(row.get("p_full_win"))
            d["p_push"] = _as_float(row.get("p_push"))
            d["p_loss"] = _as_float(row.get("p_loss"))
        return d

    def _top_by_ev(df, n, include_line=False, ah=False):
        if df is None or df.empty: return []
        use = df.sort_values(["score","kelly"], ascending=[False, False]).head(n)
        return [_mk_sel_dict(r, include_line=include_line, ah=ah) for _, r in use.iterrows()]

    def _select_totals_lines(df, center=2.5, max_lines=3):
        if df is None or df.empty: return []
        df = df.copy()
        df["line"] = df["market"].astype(str).str.extract(r"Totals\s+([0-9.]+)", expand=False).astype(float)
        df = df.dropna(subset=["line"])
        keep_lines = (df.assign(dist=(df["line"] - center).abs())
                        .sort_values("dist")
                        .drop_duplicates(subset=["match_id","line"])
                        .head(max_lines)["line"].tolist())
        out = []
        for ln in keep_lines:
            side = df[df["line"] == ln]
            over_pick  = _top_by_ev(side[side["selection"].str.startswith("Over")],  1, include_line=True)
            under_pick = _top_by_ev(side[side["selection"].str.startswith("Under")], 1, include_line=True)
            if over_pick:  out.extend(over_pick)
            if under_pick: out.extend(under_pick)
        return out

    def _select_ah_lines(df, center=0.0, max_lines=4):
        if df is None or df.empty: return []
        df = df.copy()
        df["line"] = df["selection"].str.extract(r"([+-]?\d+(?:\.\d+)?)", expand=False).astype(float)
        df = df.dropna(subset=["line"])
        keep_lines = (df.assign(dist=(df["line"] - center).abs())
                        .sort_values("dist")
                        .drop_duplicates(subset=["match_id","line"])
                        .head(max_lines)["line"].tolist())
        out = []
        for ln in keep_lines:
            side = df[df["line"] == ln]
            home_pick = _top_by_ev(side[side["selection"].str.startswith("Home")], 1, include_line=True, ah=True)
            away_pick = _top_by_ev(side[side["selection"].str.startswith("Away")], 1, include_line=True, ah=True)
            if home_pick: out.extend(home_pick)
            if away_pick: out.extend(away_pick)
        return out

    per_game = []
    if not cand_df.empty:
        meta_cols = ["match_id","home_team","away_team","kickoff_uk","kickoff_utc"]
        meta = (model[meta_cols].drop_duplicates("match_id")
                if set(meta_cols).issubset(model.columns) else
                cand_df[["match_id","home_team","away_team","kickoff_uk"]].drop_duplicates("match_id"))
        c = cand_df.copy()
        for mid, mrow in meta.set_index("match_id").iterrows():
            sub = c[c["match_id"] == mid].copy()
            if sub.empty: continue
            m1x2    = sub[sub["market"].eq("1X2")]
            mbtts   = sub[sub["market"].eq("BTTS")]
            mtotals = sub[sub["market"].str.startswith("Totals", na=False)]
            mcs     = sub[sub["market"].eq("Correct Score")]
            mah     = sub[sub["market"].eq("Asian Handicap")]

            per_game.append({
                "match_id": int(mid) if pd.notna(mid) else None,
                "home_team": mrow.get("home_team"),
                "away_team": mrow.get("away_team"),
                "kickoff_uk": iso_or_none(mrow.get("kickoff_uk")),
                "kickoff_utc": iso_or_none(mrow.get("kickoff_utc")) if "kickoff_utc" in mrow else None,
                "match_result": _top_by_ev(m1x2, 3),
                "btts": (
                    _top_by_ev(mbtts[mbtts["selection"].str.lower().eq("yes")], 1) +
                    _top_by_ev(mbtts[mbtts["selection"].str.lower().eq("no")],  1)
                ),
                "over_under": _select_totals_lines(mtotals, center=2.5, max_lines=3),
                "match_score": _top_by_ev(mcs, 3),
                "asian_handicap": _select_ah_lines(mah, center=0.0, max_lines=4),
            })

    # -------- singles: top 10 --------
    top10 = cand_df.head(10).copy()

    # -------- accas: best leg per match, then combos (exclude legs without binary p) --------
    best_by_match = (cand_df.sort_values(["match_id","score","kelly"], ascending=[True, False, False])
                            .groupby("match_id", as_index=False).first())
    legs = best_by_match.to_dict(orient="records")
    accas = []
    legs_bin = [l for l in legs if l.get("model_prob") is not None]  # drop AH legs from accas
    M = len(legs_bin)
    if M >= MIN_ACC_LEGS:
        sizes = []
        if M >= PREF_ACC_LEGS: sizes.append(PREF_ACC_LEGS)
        if M >= 4: sizes.append(4)
        if not sizes: sizes = [min(M, MIN_ACC_LEGS)]
        combos = []
        for r in sizes:
            for combo in itertools.combinations(legs_bin, r):
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
                        "odds": l["best_odds"], "p": l.get("model_prob"),
                        "bookmaker": l["best_bookmaker"]
                    } for l in c["legs"]
                ]
            })

    # -------- write JSON --------
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_json = args.outdir / f"picks_next_epl_{target_season}_mw{target_mw:02d}.json"

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "epl_matchweek": {
            "season": int(target_season),
            "mw_mode": int(target_mw),
            "mw_range": [int(target_mw), int(target_mw)]
        },
        "singles_count": int(len(top10)),
        "accas_count": int(len(accas)),
        "singles": top10.to_dict(orient="records"),
        "accas": accas,
        "per_game_markets": per_game,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # -------- preview --------
    print(f"[done] MW {target_mw:02d} (season {target_season}) | singles={len(top10)} accas={len(accas)} → {out_json}")
    if len(top10):
        cols = ["home_team","away_team","market","selection","best_odds","model_prob","score","kelly","kickoff_uk"]
        have = [c for c in cols if c in top10.columns]
        print("\nTop 10 singles:")
        print(top10[have].head(10).to_string(index=False))
    if len(accas):
        print("\nTop 5 accas (best EV):")
        for i, a in enumerate(accas, 1):
            legs_str = " | ".join([f'{l["home_team"]}–{l["away_team"]} {l["market"]} {l["selection"]} @{l["odds"]:.2f}'
                                   for l in a["legs"]])
            print(f"{i}. {a['legs_count']}-fold @ {a['acca_odds']:.2f} | EV={a['score']:+.3f} | {legs_str}")

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
