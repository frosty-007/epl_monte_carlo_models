#!/usr/bin/env python3
"""
best_value_picks_next_mw.py

- Auto-selects the NEXT EPL matchweek (MW) based on files present on disk.
- Odds file:   ../data/raw/odds/epl_odds_<season>_MWNN.parquet
- Prices file: ../data/calibrated/market_prices_epl_<season>_mwNN.parquet  (if exists)
- Fallback λ:  ../data/calibrated/team_match_lambdas.parquet
- Markets scored: 1X2, BTTS, Totals, Correct Score, Asian Handicap
- Output JSON:    ../data/output/picks/picks_next_epl_<season>_mwNN.json

Anti-draw & acca-diversity updates:
- DRAW_SCORE_PENALTY subtracts EV from all 1X2 Draw selections.
- Accas are constructed with NO draw legs (strictly banned).
"""

from __future__ import annotations
import json, re, math, itertools
from pathlib import Path
from datetime import date, timedelta
from typing import Tuple, Optional, Iterable
import numpy as np
import pandas as pd

# ---------- Paths (dirs; files inferred) ----------
ODDS_DIR      = Path("../data/raw/odds")
PRICES_DIR    = Path("../data/calibrated")
LAMBDAS_PARQ  = PRICES_DIR / "team_match_lambdas.parquet"
OUT_DIR       = Path("../data/output/picks")

# ---------- Settings ----------
CS_K_MAX         = 6        # correct-score grid (0..K)
DC_RHO           = 0.10     # POSITIVE to reduce low-score draw mass (was negative)
MIN_ACC_LEGS     = 2
PREF_ACC_LEGS    = 3        # prefer trebles; consider 4-folds if enough matches
MAX_DRAWS_IN_TOP = 3        # guardrail: at most N draws in Top-10 singles

# Draw discourager (applies everywhere before ranking)
DRAW_SCORE_PENALTY = 0.10   # subtract 0.10 EV from all 1X2 Draws (tune 0.05..0.25)

# Acca diversity: define "primary" markets that must appear at least once in each acca
PRIMARY_MARKETS = {"1X2", "Correct Score"}  # plus any market name that startswith("Asian Handicap")

# Canonical EPL keys (lowercase)
EPL_CANON = {
    "arsenal","astonvilla","bournemouth","brentford","brighton","chelsea","crystalpalace",
    "everton","fulham","ipswich","leicester","liverpool","manutd","mancity","newcastle",
    "southampton","tottenham","westham","wolves","forest","luton","sheffieldutd","westbrom"
}

# ---------- Helpers ----------
def canon(s: str) -> str:
    s0 = (s or "").lower()
    s1 = re.sub(r'[^a-z0-9]+', '', s0)
    s1 = re.sub(r'fc$', '', s1)
    syn = {
        "manchesterunited":"manutd","manutd":"manutd","manchesterutd":"manutd","mufc":"manutd",
        "manchestercity":"mancity","mancity":"mancity","mcfc":"mancity",
        "tottenhamhotspur":"tottenham","tottenham":"tottenham","spurs":"tottenham",
        "westhamunited":"westham","westham":"westham","hammers":"westham",
        "newcastleunited":"newcastle","newcastle":"newcastle","magpies":"newcastle",
        "brightonandhovealbion":"brighton","brightonhovealbion":"brighton","brighton":"brighton","seagulls":"brighton",
        "wolverhamptonwanderers":"wolves","wolverhampton":"wolves","wolves":"wolves",
        "nottinghamforest":"forest","nottmforest":"forest","forest":"forest",
        "sheffieldunited":"sheffieldutd","sheffieldutd":"sheffieldutd","sheffutd":"sheffieldutd",
        "crystalpalace":"crystalpalace","palace":"crystalpalace",
        "everton":"everton","fulham":"fulham","brentford":"brentford",
        "bournemouth":"bournemouth","afc":"bournemouth","afcb":"bournemouth",
        "burnley":"burnley","southampton":"southampton",
        "astonvillaa":"astonvilla","astonvilla":"astonvilla","villa":"astonvilla",
        "lutontown":"luton","luton":"luton",
        "ipswichtown":"ipswich","ipswich":"ipswich",
        "leicestercity":"leicester","leicester":"leicester",
        "westbromwichalbion":"westbrom","westbrom":"westbrom",
    }
    return syn.get(s1, s1)

def iso_or_none(ts):
    if ts is None or pd.isna(ts): return None
    try: return pd.Timestamp(ts).isoformat()
    except Exception: return None

def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if not s2.empty else None

# ----- Poisson / DC grid -----
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
    ph = pois_pmf_vec(kmax, lh); pa = pois_pmf_vec(kmax, la)
    cs = np.outer(ph, pa)
    if rho is not None and abs(rho) > 0:
        for (x,y) in [(0,0),(1,0),(0,1),(1,1)]:
            cs[x,y] *= dc_tau(lh, la, rho, x, y)
        s = cs.sum()
        if s > 0: cs = cs / s
    return cs

def prob_btts_yes(lh, la):
    return 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh+la))

def totals_over(mu: float, line: float) -> float:
    k = int(math.floor(line))
    p0 = math.exp(-mu); cdf = p0; pk = p0
    for i in range(1, k+1):
        pk = pk * mu / i
        cdf += pk
    return max(0.0, min(1.0, 1 - cdf))

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
    b = odds - 1.0
    q = 1.0 - p
    f = (b*p - q) / b
    return float(f) if np.isfinite(f) and f > 0 else 0.0

# ----- Asian Handicap (AH) helpers -----
def diff_pmf_from_grid(grid: np.ndarray) -> dict[int, float]:
    k = grid.shape[0]
    pmf = {}
    for h in range(k):
        for a in range(k):
            d = h - a
            pmf[d] = pmf.get(d, 0.0) + float(grid[h,a])
    s = sum(pmf.values())
    if s > 0:
        for d in list(pmf):
            pmf[d] /= s
    return pmf

def ah_split_lines(line: float) -> Iterable[float]:
    # Split quarter lines: e.g., -0.25 -> (-0.5, 0.0), +0.75 -> (+0.5, +1.0)
    frac2 = abs(line*2 - round(line*2))
    if frac2 < 1e-9:   # .0 or .5
        return [line]
    base = math.floor(line*2)/2.0
    return [base, base+0.5]

def ah_win_push_prob_home(pmf_diff: dict[int,float], line: float) -> tuple[float,float]:
    eps = 1e-12
    frac = abs(line - round(line))  # 0.0 int; 0.5 half; else split upstream
    p_win = 0.0; p_push = 0.0
    if abs(frac) < 1e-9:
        for d, p in pmf_diff.items():
            if d > line + eps: p_win += p
            elif abs(d - line) <= eps: p_push += p
    elif abs(frac - 0.5) < 1e-9:
        for d, p in pmf_diff.items():
            if d > line + eps: p_win += p
    return p_win, p_push

def ah_effective_win_prob_home(pmf_diff: dict[int,float], line: float) -> float:
    splits = list(ah_split_lines(line))
    if len(splits) == 1:
        p_win, _ = ah_win_push_prob_home(pmf_diff, splits[0])
        return p_win
    p = 0.0
    for L in splits:
        pw, _ = ah_win_push_prob_home(pmf_diff, L)
        p += pw
    return p / len(splits)

# ---------- Season / MW detection ----------
def season_year_now_uk(today_uk: Optional[pd.Timestamp]=None) -> int:
    now = today_uk or pd.Timestamp.now(tz="Europe/London")
    return now.year if now.month >= 8 else now.year - 1

def season_week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())

def season_start_monday(season_year: int) -> date:
    aug1 = date(season_year, 8, 1)
    return aug1 + timedelta(days=(7 - aug1.weekday()) % 7)

def mw_from_date_uk(d: date, season_year: int) -> int:
    s0 = season_start_monday(season_year)
    return ((season_week_monday(d) - s0).days // 7) + 1

PAT_ODDS = re.compile(r"^epl_odds_(\d{4})_MW(\d{2})\.parquet$", re.I)
def parse_odds_filename(p: Path) -> Optional[Tuple[int,int]]:
    m = PAT_ODDS.match(p.name)
    if not m: return None
    return int(m.group(1)), int(m.group(2))

def choose_next_mw_file(odds_dir: Path, today_uk: Optional[pd.Timestamp]=None) -> Tuple[Path,int,int]:
    """
    1) Use current UK season.
    2) List epl_odds_<season>_MWNN.parquet present.
    3) Estimate today's MW; choose the smallest MW >= estimate, else the max present.
    """
    today = today_uk or pd.Timestamp.now(tz="Europe/London")
    season = season_year_now_uk(today)
    candidates = []
    for p in sorted(odds_dir.glob(f"epl_odds_{season}_MW*.parquet")):
        parsed = parse_odds_filename(p)
        if parsed and parsed[0] == season:
            candidates.append((p, parsed[1]))
    if not candidates:
        anyfiles = sorted(odds_dir.glob("epl_odds_*_MW*.parquet"))
        if not anyfiles:
            raise FileNotFoundError(f"No odds parquet files found in {odds_dir}")
        best = None
        for p in anyfiles:
            parsed = parse_odds_filename(p)
            if parsed:
                if best is None or (parsed[0], parsed[1]) > best[1]:
                    best = (p, (parsed[0], parsed[1]))
        path = best[0]; season = best[1][0]; mw = best[1][1]
        return path, season, mw

    mw_est = mw_from_date_uk(today.date(), season)
    mw_list = sorted([mw for _, mw in candidates])
    for mw in mw_list:
        if mw >= mw_est:
            path = next(p for p, m in candidates if m == mw)
            return path, season, mw
    mw = mw_list[-1]
    path = next(p for p, m in candidates if m == mw)
    return path, season, mw

# ---------- Ratings fallback (λ) ----------
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

# ---------- Main ----------
def main():
    pd.options.mode.copy_on_write = True

    # 1) Choose next-MW odds file from disk
    odds_path, season_year, mw = choose_next_mw_file(ODDS_DIR)
    prices_path = PRICES_DIR / f"market_prices_epl_{season_year}_mw{mw:02d}.parquet"

    print(f"[info] Using odds: {odds_path.name}  (season={season_year}, MW={mw:02d})")
    if prices_path.exists():
        print(f"[info] Prices: {prices_path.name}")
    else:
        print(f"[warn] Prices not found for MW{mw:02d}; will use ratings fallback: {LAMBDAS_PARQ.name}")

    # 2) Load odds
    o_all = pd.read_parquet(odds_path)
    o_all["commence_time"] = pd.to_datetime(o_all["commence_time"], utc=True, errors="coerce")
    o_all["kickoff_uk"] = o_all["commence_time"].dt.tz_convert("Europe/London")

    # Canonical keys + EPL filter
    o_all["home_key"] = o_all["home_team"].apply(canon)
    o_all["away_key"] = o_all["away_team"].apply(canon)
    epl_mask = o_all["home_key"].isin(EPL_CANON) & o_all["away_key"].isin(EPL_CANON)
    o = o_all[epl_mask].copy()
    if o.empty:
        raise RuntimeError("No EPL fixtures found in selected odds file.")

    # One row per match for names/times
    names_by_match = (
        o.groupby("match_id", as_index=False)
         .agg(home_team=("home_team", first_non_null),
              away_team=("away_team", first_non_null),
              kickoff_utc=("commence_time", "min"),
              kickoff_uk=("kickoff_uk", "min"))
    )
    keys_by_match = (
        o.groupby("match_id", as_index=False)
         .agg(home_key=("home_key", first_non_null),
              away_key=("away_key", first_non_null))
    )
    fixtures = names_by_match.merge(keys_by_match, on="match_id", how="left")

    # ---------- 3) Build λ model — prices preferred, else ratings fallback ----------
    def _safe_str_series(s):
        return s.astype(str).str.strip() if s is not None else s

    def _prepare_fixtures_for_join(fixtures_df):
        fx = fixtures_df.copy()
        fx["kick_date_uk"] = pd.to_datetime(fx["kickoff_uk"], errors="coerce").dt.tz_convert("Europe/London").dt.date
        fx["match_id_str"] = _safe_str_series(fx["match_id"])
        if "home_key" not in fx.columns or "away_key" not in fx.columns:
            fx["home_key"] = fx["home_team"].apply(canon)
            fx["away_key"] = fx["away_team"].apply(canon)
        return fx

    def _prepare_prices_for_join(prices_df):
        pr = prices_df.copy()
        lam_h = next((c for c in pr.columns if c.lower() in {"lambda_home","lambda_h","lam_home","lam_h"}), None)
        lam_a = next((c for c in pr.columns if c.lower() in {"lambda_away","lambda_a","lam_away","lam_a"}), None)
        if not lam_h or not lam_a:
            raise RuntimeError("Prices parquet does not contain lambda_home/away columns.")
        if "match_id" in pr.columns:
            pr["match_id_str"] = _safe_str_series(pr["match_id"])
        if "home_key" not in pr.columns and "home_team" in pr.columns:
            pr["home_key"] = pr["home_team"].apply(canon)
        if "away_key" not in pr.columns and "away_team" in pr.columns:
            pr["away_key"] = pr["away_team"].apply(canon)
        dt_col = None
        for c in ["kickoff_uk","kickoff_utc","kickoff_time","datetime","commence_time"]:
            if c in pr.columns:
                dt_col = c; break
        if dt_col:
            pr["kickoff_uk_norm"] = pd.to_datetime(pr[dt_col], utc=True, errors="coerce").dt.tz_convert("Europe/London")
            pr["kick_date_uk"] = pr["kickoff_uk_norm"].dt.date
        keep = list(dict.fromkeys([c for c in [
            "match_id_str","home_key","away_key","kick_date_uk", lam_h, lam_a, "kickoff_uk_norm"
        ] if c in pr.columns]))
        pr = pr[keep].copy().rename(columns={lam_h:"lambda_home", lam_a:"lambda_away"})
        return pr

    fixtures = _prepare_fixtures_for_join(fixtures)
    model = pd.DataFrame()

    if prices_path.exists():
        prices_raw = pd.read_parquet(prices_path)
        prices = _prepare_prices_for_join(prices_raw)
        if {"match_id_str","lambda_home","lambda_away"}.issubset(prices.columns):
            j = fixtures.merge(prices[["match_id_str","lambda_home","lambda_away"]],
                               on="match_id_str", how="left")
            model = j.dropna(subset=["lambda_home","lambda_away"])
        if model.empty and {"home_key","away_key","kick_date_uk","lambda_home","lambda_away"}.issubset(prices.columns):
            j2 = fixtures.merge(
                prices[["home_key","away_key","kick_date_uk","lambda_home","lambda_away"]],
                on=["home_key","away_key","kick_date_uk"], how="left"
            )
            model = j2.dropna(subset=["lambda_home","lambda_away"])
        if model.empty and "kickoff_uk_norm" in prices.columns:
            fx = fixtures.copy()
            fx["kick_ts"] = pd.to_datetime(fx["kickoff_uk"], errors="coerce")
            pr = prices.dropna(subset=["kickoff_uk_norm"]).copy()
            pr["kick_ts"] = pd.to_datetime(pr["kickoff_uk_norm"], errors="coerce")
            rows = []
            for _, r in fx.iterrows():
                hk, ak, ref = r["home_key"], r["away_key"], r["kick_ts"]
                if pd.isna(ref): 
                    continue
                lo = ref - pd.Timedelta(days=2)
                hi = ref + pd.Timedelta(days=2)
                cand = pr[(pr["home_key"]==hk) & (pr["away_key"]==ak) & (pr["kick_ts"].between(lo, hi))]
                if cand.empty:
                    continue
                cand["abs_diff"] = (cand["kick_ts"] - ref).abs()
                best = cand.sort_values("abs_diff").iloc[0]
                rows.append({
                    **r.to_dict(),
                    "lambda_home": best["lambda_home"],
                    "lambda_away": best["lambda_away"],
                })
            if rows:
                model = pd.DataFrame(rows)

    if model.empty:
        if not LAMBDAS_PARQ.exists():
            raise RuntimeError("No matches with λ available (prices didn’t align and ratings parquet missing).")
        lam_hist = pd.read_parquet(LAMBDAS_PARQ)
        team_att, team_def, hfa, c = build_team_ratings(lam_hist)
        model = predict_lambdas(fixtures[["match_id","home_key","away_key"]], team_att, team_def, hfa, c) \
                  .merge(fixtures.drop(columns=["home_key","away_key"]), on="match_id", how="left")

    if model.empty:
        raise RuntimeError("No matches with λ available after merging model with fixtures.")

    # ---------- Extract best prices per market ----------
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
    btts = o[o["market"].astype(str).str.lower().eq("btts") & o["outcome"].str.lower().isin(["yes","no"])].copy()
    btts["outcome"] = btts["outcome"].str.lower()
    if not btts.empty:
        idx = btts.groupby(["match_id","outcome"])["price"].idxmax()
        btts_best = btts.loc[idx, ["match_id","outcome","bookmaker_title","price"]]
        btts_yes = btts_best[btts_best["outcome"]=="yes"].rename(columns={"price":"bk_btts_yes","bookmaker_title":"bk_btts_yes_book"})
        btts_no  = btts_best[btts_best["outcome"]=="no" ].rename(columns={"price":"bk_btts_no","bookmaker_title":"bk_btts_no_book"})
        btts_top = btts_yes.merge(btts_no, on="match_id", how="outer")[["match_id","bk_btts_yes","bk_btts_yes_book","bk_btts_no","bk_btts_no_book"]]
    else:
        btts_top = pd.DataFrame(columns=["match_id","bk_btts_yes","bk_btts_yes_book","bk_btts_no","bk_btts_no_book"])

    # Totals
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

    # Correct Score
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

    # Asian Handicap (spreads)
    ah = o[o["market"].astype(str).str.lower().isin(["spreads","asian_handicap","asian handicap","handicap"]) &
            o["outcome"].str.lower().isin(["home","away"])].copy()
    if not ah.empty:
        ah["point_f"] = pd.to_numeric(ah["point"], errors="coerce")
        ah = ah.dropna(subset=["point_f"])
        ah["outcome"] = ah["outcome"].str.lower()
        idx = ah.groupby(["match_id","outcome","point_f"])["price"].idxmax()
        ah_top = ah.loc[idx, ["match_id","outcome","point_f","bookmaker_title","price"]].rename(
            columns={"bookmaker_title":"bk_ah_book","price":"bk_ah"})
    else:
        ah_top = pd.DataFrame(columns=["match_id","outcome","point_f","bk_ah_book","bk_ah"])

    # ---------- Assemble model + odds ----------
    base = (model
            .merge(h2h_top, on="match_id", how="left")
            .merge(btts_top, on="match_id", how="left"))

    # ---------- Build candidates ----------
    candidates = []

    # de-juice 1X2 implied if available
    if not base.empty and {"bk_home","bk_draw","bk_away"}.issubset(base.columns):
        imp = norm_1x2(base["bk_home"], base["bk_draw"], base["bk_away"])
        base["impH_mkt"], base["impD_mkt"], base["impA_mkt"] = imp[:,0], imp[:,1], imp[:,2]

    for _, r in base.iterrows():
        lh, la = float(r["lambda_home"]), float(r["lambda_away"])
        grid = cs_grid(lh, la, CS_K_MAX, rho=DC_RHO)
        pD = float(np.trace(grid))
        pH = float(np.triu(grid, 1).sum())
        pA = float(np.tril(grid, -1).sum())
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
            o_ = float(y_odds)
            candidates.append({
                "match_id": r["match_id"],
                "home_team": r["home_team"], "away_team": r["away_team"],
                "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                "market": "BTTS", "selection": "Yes",
                "model_prob": float(p_btts),
                "market_imp_prob": float(imp_y) if imp_y is not None else None,
                "edge": float(p_btts - imp_y) if imp_y is not None else None,
                "kelly": kelly(o_, float(p_btts)),
                "best_odds": o_, "best_bookmaker": y_book,
                "score": float(p_btts*o_ - 1.0),
            })
        if pd.notna(n_odds) and float(n_odds) > 1.0:
            o_ = float(n_odds); p_no = 1 - p_btts
            candidates.append({
                "match_id": r["match_id"],
                "home_team": r["home_team"], "away_team": r["away_team"],
                "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                "market": "BTTS", "selection": "No",
                "model_prob": float(p_no),
                "market_imp_prob": float(imp_n) if imp_n is not None else None,
                "edge": float(p_no - imp_n) if imp_n is not None else None,
                "kelly": kelly(o_, float(p_no)),
                "best_odds": o_, "best_bookmaker": n_book,
                "score": float(p_no*o_ - 1.0),
            })

    # Totals
    if 'tot_top' not in locals():
        tot_top = pd.DataFrame(columns=["match_id","point_f","bk_over","bk_over_book","bk_under","bk_under_book"])
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
                o_ = float(o_over)
                candidates.append({
                    "match_id": r["match_id"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                    "market": f"Totals {line}", "selection": f"Over {line}",
                    "model_prob": float(p_over),
                    "market_imp_prob": float(imp_over) if imp_over is not None else None,
                    "edge": float(p_over - imp_over) if imp_over is not None else None,
                    "kelly": kelly(o_, float(p_over)),
                    "best_odds": o_, "best_bookmaker": b_over,
                    "score": float(p_over*o_ - 1.0),
                })
            if pd.notna(o_under) and float(o_under) > 1.0:
                o_ = float(o_under)
                candidates.append({
                    "match_id": r["match_id"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                    "market": f"Totals {line}", "selection": f"Under {line}",
                    "model_prob": float(p_under),
                    "market_imp_prob": float(imp_under) if imp_under is not None else None,
                    "edge": float(p_under - imp_under) if imp_under is not None else None,
                    "kelly": kelly(o_, float(p_under)),
                    "best_odds": o_, "best_bookmaker": b_under,
                    "score": float(p_under*o_ - 1.0),
                })

    # Correct Score
    if 'cs_top' not in locals():
        cs_top = pd.DataFrame(columns=["match_id","sc_h","sc_a","bk_cs_book","bk_cs"])
    if not cs_top.empty:
        cs_join = cs_top.merge(model[["match_id","home_team","away_team","kickoff_utc","kickoff_uk","lambda_home","lambda_away"]],
                               on="match_id", how="inner")
        for _, r in cs_join.iterrows():
            lh, la = float(r["lambda_home"]), float(r["lambda_away"])
            grid = cs_grid(lh, la, CS_K_MAX, rho=DC_RHO)
            h, a = int(r["sc_h"]), int(r["sc_a"])
            if h <= CS_K_MAX and a <= CS_K_MAX:
                p = float(grid[h, a]); o_ = float(r["bk_cs"])
                if o_ > 1.0 and p > 0:
                    candidates.append({
                        "match_id": r["match_id"],
                        "home_team": r["home_team"], "away_team": r["away_team"],
                        "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                        "market": "Correct Score", "selection": f"{h}-{a}",
                        "model_prob": p,
                        "market_imp_prob": None, "edge": None,
                        "kelly": kelly(o_, p),
                        "best_odds": o_, "best_bookmaker": r.get("bk_cs_book"),
                        "score": p*o_ - 1.0,
                    })

    # Asian Handicap
    if 'ah_top' not in locals():
        ah_top = pd.DataFrame(columns=["match_id","outcome","point_f","bk_ah_book","bk_ah"])
    if not ah_top.empty:
        ah_join = ah_top.merge(model[["match_id","home_team","away_team","kickoff_utc","kickoff_uk","lambda_home","lambda_away"]],
                               on="match_id", how="inner")
        for _, r in ah_join.iterrows():
            lh, la = float(r["lambda_home"]), float(r["lambda_away"])
            grid = cs_grid(lh, la, CS_K_MAX, rho=DC_RHO)
            pmf_diff = diff_pmf_from_grid(grid)
            side = r["outcome"]  # 'home' or 'away'
            line = float(r["point_f"])
            home_line = line if side == "home" else (-line)
            p_win_eff = ah_effective_win_prob_home(pmf_diff, home_line)
            o_ = float(r["bk_ah"])
            if o_ > 1.0:
                market_lbl = f"Asian Handicap {home_line:+.2f}" if side=="home" else f"Asian Handicap {(-home_line):+.2f}"
                sel_lbl = "Home" if side=="home" else "Away"
                candidates.append({
                    "match_id": r["match_id"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "kickoff_utc": iso_or_none(r.get("kickoff_utc")), "kickoff_uk": iso_or_none(r.get("kickoff_uk")),
                    "market": market_lbl, "selection": sel_lbl,
                    "model_prob": float(p_win_eff),
                    "market_imp_prob": None, "edge": None,
                    "kelly": kelly(o_, float(p_win_eff)),
                    "best_odds": o_, "best_bookmaker": r.get("bk_ah_book"),
                    "score": float(p_win_eff*o_ - 1.0),
                })

    cand_df = pd.DataFrame(candidates)
    if cand_df.empty:
        raise RuntimeError("No candidate picks were generated for the selected matchweek.")

    # ------------ GLOBAL DRAW PENALTY ------------
    # Push 1X2 Draws down the EV ranking everywhere.
    is_draw = (cand_df["market"].eq("1X2")) & (cand_df["selection"].str.lower().eq("draw"))
    cand_df.loc[is_draw, "score"] = cand_df.loc[is_draw, "score"].astype(float) - DRAW_SCORE_PENALTY

    # Clean/sort master list by EV then Kelly
    cand_df = cand_df.dropna(subset=["home_team","away_team","best_odds","model_prob"])
    cand_df = cand_df.sort_values(["score","kelly"], ascending=[False, False]).reset_index(drop=True)

    # -------- Guardrail to limit draws in Top-10 (OPTION: hard-ban draws by uncommenting the next line) --------
    top10_rows = []
    draw_count = 0
    for _, row in cand_df.iterrows():
        if len(top10_rows) == 10: break
        if row.get("market") == "1X2" and str(row.get("selection")).lower() == "draw":
            if draw_count >= MAX_DRAWS_IN_TOP:
                continue
            draw_count += 1
        top10_rows.append(row)
    top10 = pd.DataFrame(top10_rows)
    # top10 = top10[~(top10["market"].eq("1X2") & top10["selection"].str.lower().eq("draw"))].head(10).copy()

    # -------- Acca logic: NO DRAW LEGS --------
    def market_priority(m):
        ms = str(m)
        if ms == "1X2": return 0
        if ms.startswith("Asian Handicap"): return 1
        if ms == "Correct Score": return 2
        if ms.startswith("Totals"): return 3
        if ms == "BTTS": return 4
        return 5

    def is_primary_market(market: str) -> bool:
        if market in PRIMARY_MARKETS: return True
        return market.startswith("Asian Handicap")

    def is_draw_row(row) -> bool:
        return (str(row.get("market")) == "1X2") and (str(row.get("selection")).lower() == "draw")

    # For each match: pick the best NON-DRAW leg by (priority, score, kelly).
    def pick_leg_no_draw(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        g["_pri"] = g["market"].map(market_priority)
        g = g.sort_values(["_pri","score","kelly"], ascending=[True, False, False])
        non_draw = g[~g.apply(is_draw_row, axis=1)]
        if not non_draw.empty:
            return non_draw.iloc[0]
        return g.iloc[0]  # fallback (only draws available)

    best_by_match = (
        cand_df
        .groupby("match_id", group_keys=False)
        .apply(pick_leg_no_draw)
        .reset_index(drop=True)
    )

    legs = best_by_match.to_dict(orient="records")
    M = len(legs)
    accas = []
    if M >= MIN_ACC_LEGS:
        sizes = []
        if M >= PREF_ACC_LEGS: sizes.append(PREF_ACC_LEGS)
        if M >= 4: sizes.append(4)
        if not sizes: sizes = [min(M, MIN_ACC_LEGS)]
        combos = []
        for rsize in sizes:
            for combo in itertools.combinations(legs, rsize):
                # Ban combos containing any draw leg (should already be filtered, but keep hard check)
                if any((c["market"] == "1X2") and (str(c["selection"]).lower() == "draw") for c in combo):
                    continue
                # also require at least one primary leg
                if not any(is_primary_market(c["market"]) for c in combo):
                    continue
                probs = [c["model_prob"] for c in combo]
                odds  = [c["best_odds"]  for c in combo]
                if any(pd.isna(p) or pd.isna(o) or float(o) <= 1.0 for p, o in zip(probs, odds)):
                    continue
                prod_p = float(np.prod(probs))
                prod_o = float(np.prod(odds))
                ev = prod_p * prod_o - 1.0
                combos.append({"legs": combo, "legs_count": rsize, "acca_prob": prod_p, "acca_odds": prod_o, "score": ev})

        combos = sorted(combos, key=lambda x: (x["score"], x["acca_prob"]), reverse=True)[:5]
        for c in combos:
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

    # -------- Output --------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"picks_next_epl_{season_year}_mw{mw:02d}.json"

    payload = {
        "generated_at_uk": pd.Timestamp.now(tz="Europe/London").isoformat(),
        "season": season_year,
        "matchweek": mw,
        "sources": {
            "odds": str(odds_path),
            "prices": (str(prices_path) if prices_path.exists() else None),
            "ratings_fallback": (str(LAMBDAS_PARQ) if LAMBDAS_PARQ.exists() else None),
        },
        "params": {
            "cs_k_max": CS_K_MAX,
            "dc_rho": DC_RHO,
            "draw_score_penalty": DRAW_SCORE_PENALTY,
            "max_draws_in_top": MAX_DRAWS_IN_TOP,
            "primary_markets": sorted(list(PRIMARY_MARKETS)),
            "acca_requires_primary": True,
            "acca_bans_draws": True
        },
        "singles_count": int(len(top10)),
        "accas_count": int(len(accas)),
        "singles": top10.to_dict(orient="records"),
        "accas": accas,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[done] singles={len(top10)} (draws ≤ {MAX_DRAWS_IN_TOP})  accas={len(accas)} → {out_json}")

if __name__ == "__main__":
    main()
