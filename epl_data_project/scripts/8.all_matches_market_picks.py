#!/usr/bin/env python3
"""
8.epl_market_best_value.py

Per-match BEST VALUE (highest EV) picks for all fixtures in the EPL matchweek encoded
in a saved odds parquet: ../data/raw/odds/epl_odds_<SEASON>_MWnn.parquet

Markets scored:
- Match result (h2h / h2h_3_way): H/D/A
- BTTS: Yes/No
- Totals: searches all available lines; returns best Over/Under at the best line
- Asian Handicap (spreads/asian_handicap/handicap/ah): searches all lines; best Home/Away; handles .0/.25/.5/.75
- Match score (Correct Score): best EV scoreline in a small grid

Output (no singles/accas):
  ../data/output/picks/all_market_picks_next_epl_<SEASON>_mwNN.json
"""

from __future__ import annotations
import argparse, re, json, math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Per-match best value picks (best EV) with best bookmaker prices.")
    ap.add_argument("--odds-file", default="", help="Exact odds parquet (e.g. ../data/raw/odds/epl_odds_2025_MW04.parquet)")
    ap.add_argument("--odds-dir", default="../data/raw/odds", help="Dir to auto-pick latest epl_odds_YYYY_MWnn.parquet")
    ap.add_argument("--outdir", default="../data/output/picks", help="Output directory")
    ap.add_argument("--team-lambdas", default="../data/calibrated/team_match_lambdas.parquet",
                    help="Fallback historical team λ parquet")
    ap.add_argument("--prices-dir", default="../data/calibrated",
                    help="Dir possibly containing market_prices_epl_<SEASON>_mwNN.parquet")
    ap.add_argument("--cs-k-max", type=int, default=6, help="Correct score grid 0..K")
    ap.add_argument("--dc-rho", type=float, default=-0.05, help="Dixon–Coles tweak")
    ap.add_argument("--grid-k", type=int, default=10, help="Grid cap for handicap EV (Skellam from 0..K grid)")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

# --------------- regex / canon ---------------
FILENAME_RX = re.compile(r"epl_odds_(\d{4})_MW(\d{2})(?:-MW(\d{2}))?\.parquet$", re.IGNORECASE)
EPL_CANON = {
    "arsenal","astonvilla","bournemouth","brentford","brighton","chelsea","crystalpalace",
    "everton","fulham","ipswich","leicester","liverpool","manutd","mancity","newcastle",
    "southampton","tottenham","westham","wolves","forest","luton","sheffieldutd","westbrom"
}
def canon(s: str) -> str:
    import re as _re
    s0 = (s or "").lower()
    s1 = _re.sub(r"[^a-z0-9]+","", s0)
    s1 = _re.sub(r"fc$","", s1)
    syn = {
        "manchesterunited":"manutd","manutd":"manutd","manchesterutd":"manutd",
        "manchestercity":"mancity","mancity":"mancity",
        "tottenhamhotspur":"tottenham","spurs":"tottenham",
        "westhamunited":"westham","westham":"westham",
        "newcastleunited":"newcastle",
        "brightonandhovealbion":"brighton","brightonhovealbion":"brighton",
        "wolverhamptonwanderers":"wolves","wolverhampton":"wolves","wolves":"wolves",
        "nottinghamforest":"forest","nottmforest":"forest","forest":"forest",
        "sheffieldunited":"sheffieldutd","sheffieldutd":"sheffieldutd",
        "westbromwichalbion":"westbrom","westbrom":"westbrom",
        "lutontown":"luton","ipswichtown":"ipswich","leicestercity":"leicester",
        "astonvillaa":"astonvilla",
    }
    return syn.get(s1, s1)

def as_utc(ts):
    return pd.to_datetime(ts, utc=True, errors="coerce")

def to_uk(ts):
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if isinstance(t, pd.Series): return t.dt.tz_convert("Europe/London")
    return t.tz_convert("Europe/London")

def iso_or_none(ts):
    if ts is None or pd.isna(ts): return None
    try: return pd.Timestamp(ts).isoformat()
    except Exception: return None

def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if not s2.empty else None

def pick_latest_odds_file(odds_dir: Path) -> Path:
    cand = []
    for p in odds_dir.glob("epl_odds_*_MW*.parquet"):
        m = FILENAME_RX.search(p.name)
        if not m: continue
        season = int(m.group(1)); mw1 = int(m.group(2))
        mw2 = int(m.group(3)) if m.group(3) else mw1
        cand.append((season, mw2, p.stat().st_mtime, p))
    if not cand:
        raise FileNotFoundError(f"No epl_odds_YYYY_MWnn.parquet files in {odds_dir}")
    cand.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return cand[0][3]

def parse_season_mw_from_filename(path: Path) -> tuple[int, List[int]]:
    m = FILENAME_RX.search(path.name)
    if not m: raise ValueError(f"Bad odds filename: {path.name}")
    season = int(m.group(1)); mw1 = int(m.group(2))
    mw2 = int(m.group(3)) if m.group(3) else mw1
    return season, list(range(mw1, mw2+1))

# --------------- math / model ---------------
def pois_pmf_vec(k: int, lam: float) -> np.ndarray:
    out = np.zeros(k+1)
    out[0] = math.exp(-lam)
    for i in range(1, k+1):
        out[i] = out[i-1] * lam / i
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
    if rho and abs(rho) > 0:
        for (x,y) in [(0,0),(1,0),(0,1),(1,1)]:
            cs[x,y] *= dc_tau(lh, la, rho, x, y)
        s = cs.sum()
        if s > 0: cs = cs / s
    return cs

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

def predict_lambdas(fixtures_keys: pd.DataFrame, team_att, team_def, hfa, c):
    default_att = float(team_att.mean()); default_def = float(team_def.mean())
    def pred_pair(hk, ak):
        ah = team_att.get(hk, default_att); da = team_def.get(ak, default_def)
        aa = team_att.get(ak, default_att); dh = team_def.get(hk, default_def)
        lh = c * ah * da * hfa; la = c * aa * dh
        return max(lh, 1e-6), max(la, 1e-6)
    out = fixtures_keys.copy()
    out[["lambda_home","lambda_away"]] = out.apply(
        lambda r: pd.Series(pred_pair(r["home_key"], r["away_key"])), axis=1
    )
    return out

# ----- Handicap EV helpers -----
def skellam_from_grid(cs: np.ndarray) -> Dict[int, float]:
    K = cs.shape[0]-1
    probs = {}
    for h in range(K+1):
        for a in range(K+1):
            d = h - a
            probs[d] = probs.get(d, 0.0) + float(cs[h, a])
    return probs

def cdf_ge_from_skellam(sk: Dict[int,float], thr: float) -> float:
    k = math.ceil(thr - 1e-12)
    return sum(p for d,p in sk.items() if d >= k)

def p_eq_from_skellam(sk: Dict[int,float], val: int) -> float:
    return sk.get(val, 0.0)

def ev_handicap(odds: float, sk: Dict[int,float], hcap: float, side: str) -> float:
    if side == "away":
        sk = { -d: p for d,p in sk.items() }
        hcap = -hcap
    def _ev_simple(h):
        if abs(h - round(h)) < 1e-12:
            p_win  = cdf_ge_from_skellam(sk, 1 - h)
            p_push = p_eq_from_skellam(sk, int(-h))
            p_lose = 1 - p_win - p_push
            return p_win * (odds - 1.0) - p_lose
        else:
            p_win = cdf_ge_from_skellam(sk, -h + 1e-12)
            p_lose = 1 - p_win
            return p_win * (odds - 1.0) - p_lose
    if abs((hcap*2) - round(hcap*2)) < 1e-12:
        return _ev_simple(hcap)
    lower_half = math.floor(hcap*2)/2.0
    upper_half = lower_half + 0.5
    return 0.5*_ev_simple(lower_half) + 0.5*_ev_simple(upper_half)

# --------------- main ---------------
def main():
    args = parse_args()
    odds_path = Path(args.odds_file) if args.odds_file else pick_latest_odds_file(Path(args.odds_dir))
    season, mws = parse_season_mw_from_filename(odds_path)
    if args.debug: print(f"[info] using {odds_path.name} (season={season} MW={mws})")

    # ---- load odds ----
    df = pd.read_parquet(odds_path).copy()
    need = {"match_id","commence_time","home_team","away_team","market","outcome","bookmaker_title","price","point"}
    missing = [c for c in need if c not in df.columns]
    if missing: raise SystemExit(f"Odds parquet missing columns: {missing}")

    df["commence_time"] = as_utc(df["commence_time"])
    df["home_key"] = df["home_team"].apply(canon)
    df["away_key"] = df["away_team"].apply(canon)
    epl_mask = df["home_key"].isin(EPL_CANON) & df["away_key"].isin(EPL_CANON)
    df = df[epl_mask].copy()
    if df.empty: raise SystemExit("No EPL rows in odds parquet.")

    # Fixtures (use ALL fixtures in that saved MW)
    fixtures = (df.groupby("match_id", as_index=False)
                  .agg(home_team=("home_team", first_non_null),
                       away_team=("away_team", first_non_null),
                       kickoff_utc=("commence_time","min")))
    fixtures["kickoff_uk"] = to_uk(fixtures["kickoff_utc"])
    dt_min_uk = fixtures["kickoff_uk"].min(); dt_max_uk = fixtures["kickoff_uk"].max()

    # ---- λ: prices parquet if present, else historical fallback ----
    prices_path = Path(args.prices_dir) / f"market_prices_epl_{season}_mw{mws[0]:02d}.parquet"
    model = pd.DataFrame()
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        have = {"match_id","lambda_home","lambda_away"}
        model = fixtures.merge(prices[list(have)], on="match_id", how="left").dropna(subset=["lambda_home","lambda_away"])
        if args.debug: print(f"[info] λ from {prices_path.name}: {len(model)}/{len(fixtures)}")
    if model.empty:
        lam_parq = Path(args.team_lambdas)
        if not lam_parq.exists():
            raise SystemExit("No λ available and team_match_lambdas.parquet not found.")
        lam_hist = pd.read_parquet(lam_parq).copy()
        team_att, team_def, hfa, c = build_team_ratings(lam_hist)
        keys = fixtures.assign(home_key=fixtures["home_team"].apply(canon),
                               away_key=fixtures["away_team"].apply(canon))[["match_id","home_key","away_key"]]
        model = predict_lambdas(keys, team_att, team_def, hfa, c).merge(
            fixtures, on="match_id", how="left"
        )
        if args.debug: print(f"[info] λ predicted for {len(model)} matches")

    # ---------- helpers ----------
    def best_by_group(frame, keys):
        idx = frame.groupby(keys)["price"].idxmax()
        return frame.loc[idx].copy()

    mkt_lower = df["market"].astype(str).str.lower()

    # H2H
    h2h = df[df["market"].isin(["h2h","h2h_3_way"]) & df["outcome"].isin(["home","draw","away"])].copy()
    h2h_best = best_by_group(h2h, ["match_id","outcome"]) if not h2h.empty else h2h

    # BTTS
    btts = df[mkt_lower.eq("btts") & df["outcome"].str.lower().isin(["yes","no"])].copy()
    if not btts.empty: btts["outcome"] = btts["outcome"].str.lower()
    btts_best = best_by_group(btts, ["match_id","outcome"]) if not btts.empty else btts

    # Totals
    tot = df[mkt_lower.eq("totals") & df["outcome"].isin(["over","under"])].copy()
    if not tot.empty:
        tot["point_f"] = pd.to_numeric(tot["point"], errors="coerce")
        tot = tot.dropna(subset=["point_f"])
        tot_best = best_by_group(tot, ["match_id","point_f","outcome"])
    else:
        tot_best = tot

    # Correct Score
    cs_mask = mkt_lower.str.contains(r"\b(?:correct[\s_]?score|cs)\b", regex=True, na=False)
    cs = df[cs_mask].copy()
    if not cs.empty:
        sc = cs["outcome"].astype(str).str.extract(r"(\d+)\D+(\d+)")
        sc.columns = ["sc_h","sc_a"]
        cs = pd.concat([cs, sc], axis=1).dropna(subset=["sc_h","sc_a"]).copy()
        cs["sc_h"] = cs["sc_h"].astype(int); cs["sc_a"] = cs["sc_a"].astype(int)
        cs_best = best_by_group(cs, ["match_id","sc_h","sc_a"])
    else:
        cs_best = cs

    # Asian Handicap
    ah_mask = mkt_lower.str.contains(r"\b(spreads?|asian[_\s]?handicap|handicap|^ah)$", regex=True, na=False)
    ah = df[ah_mask].copy()
    if not ah.empty:
        ah["point_f"] = pd.to_numeric(ah["point"], errors="coerce")
        ah = ah.dropna(subset=["point_f"])
        def _side(row):
            nm = (str(row.get("outcome")) if pd.notna(row.get("outcome")) else "").lower()
            ht = (row.get("home_team") or "").lower(); at = (row.get("away_team") or "").lower()
            if nm in ("home","away"): return nm
            if ht and (nm==ht or ht in nm): return "home"
            if at and (nm==at or at in nm): return "away"
            return None
        ah["side"] = ah.apply(_side, axis=1)
        ah = ah.dropna(subset=["side"])
        ah_best = best_by_group(ah, ["match_id","point_f","side"])
    else:
        ah_best = ah

    # ---------- build EV & choose best pick per market ----------
    out_rows = []
    lams = model.set_index("match_id")[["lambda_home","lambda_away"]].to_dict(orient="index")

    for _, fx in fixtures.sort_values("kickoff_uk").iterrows():
        mid = fx["match_id"]; meta = {
            "match_id": int(mid) if pd.notna(mid) and str(mid).isdigit() else str(mid),
            "home_team": fx["home_team"], "away_team": fx["away_team"],
            "kickoff_uk": iso_or_none(fx["kickoff_uk"]),
            "kickoff_utc": iso_or_none(fx["kickoff_utc"]),
        }
        lh = lams.get(mid, {}).get("lambda_home")
        la = lams.get(mid, {}).get("lambda_away")
        best = {}

        # --- 1X2 ---
        sub = h2h_best[h2h_best["match_id"].eq(mid)]
        if not sub.empty:
            pick_res = None
            if lh is not None:
                grid = cs_grid(lh, la, kmax=10, rho=args.dc_rho)
                pD = float(np.trace(grid)); pH = float(np.triu(grid, 1).sum()); pA = float(np.tril(grid, -1).sum())
                opts = []
                for lab, p in [("home", pH), ("draw", pD), ("away", pA)]:
                    row = sub[sub["outcome"].eq(lab)]
                    if row.empty: continue
                    o = float(row["price"].iloc[0]); bk = str(row["bookmaker_title"].iloc[0])
                    ev = p*o - 1.0
                    opts.append(("1X2", lab.title(), o, bk, p, ev))
                if opts: pick_res = max(opts, key=lambda x: x[5])
            else:
                row = sub.sort_values("price", ascending=False).head(1).iloc[0]
                pick_res = ("1X2", str(row["outcome"]).title(), float(row["price"]), str(row["bookmaker_title"]), None, None)
            if pick_res:
                best["match_result"] = {
                    "selection": pick_res[1], "price": pick_res[2], "bookmaker": pick_res[3],
                    "model_prob": pick_res[4], "ev": pick_res[5]
                }

        # --- BTTS ---
        sub = btts_best[btts_best["match_id"].eq(mid)]
        if not sub.empty:
            pick_btts = None
            if lh is not None:
                p_yes = 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh+la))
                p_no = 1 - p_yes
                opts = []
                for lab, p in [("yes", p_yes), ("no", p_no)]:
                    row = sub[sub["outcome"].eq(lab)]
                    if row.empty: continue
                    o = float(row["price"].iloc[0]); bk = str(row["bookmaker_title"].iloc[0])
                    ev = p*o - 1.0
                    opts.append(("BTTS", lab.title(), o, bk, p, ev))
                if opts: pick_btts = max(opts, key=lambda x: x[5])
            else:
                row = sub.sort_values("price", ascending=False).head(1).iloc[0]
                pick_btts = ("BTTS", str(row["outcome"]).title(), float(row["price"]), str(row["bookmaker_title"]), None, None)
            if pick_btts:
                best["btts"] = {
                    "selection": pick_btts[1], "price": pick_btts[2], "bookmaker": pick_btts[3],
                    "model_prob": pick_btts[4], "ev": pick_btts[5]
                }

        # --- Totals (all lines) ---
        sub = tot_best[tot_best["match_id"].eq(mid)]
        if not sub.empty:
            pick_totals = None
            if lh is not None:
                mu = lh + la
                opts = []
                for ln, grp in sub.groupby("point_f"):
                    over = grp[grp["outcome"].eq("over")]
                    under= grp[grp["outcome"].eq("under")]
                    if not over.empty:
                        o = float(over["price"].iloc[0]); bk = str(over["bookmaker_title"].iloc[0])
                        k = int(math.floor(ln))
                        pk = math.exp(-mu); cdf = pk
                        for i in range(1, k+1):
                            pk = pk * mu / i; cdf += pk
                        p = max(0.0, min(1.0, 1 - cdf))
                        ev = p*o - 1.0
                        opts.append(("Over", ln, o, bk, p, ev))
                    if not under.empty:
                        o = float(under["price"].iloc[0]); bk = str(under["bookmaker_title"].iloc[0])
                        k = int(math.floor(ln))
                        pk = math.exp(-mu); cdf = pk
                        for i in range(1, k+1):
                            pk = pk * mu / i; cdf += pk
                        p_over = max(0.0, min(1.0, 1 - cdf))
                        p = 1 - p_over
                        ev = p*o - 1.0
                        opts.append(("Under", ln, o, bk, p, ev))
                if opts:
                    pick = max(opts, key=lambda x: x[5])
                    pick_totals = {"selection": f"{pick[0]} {pick[1]:g}", "price": pick[2], "bookmaker": pick[3],
                                   "model_prob": pick[4], "ev": pick[5]}
            else:
                row = sub.sort_values("price", ascending=False).head(1).iloc[0]
                pick_totals = {"selection": f"{row['outcome'].title()} {row['point_f']:g}",
                               "price": float(row["price"]), "bookmaker": str(row["bookmaker_title"]),
                               "model_prob": None, "ev": None}
            if pick_totals:
                best["totals"] = pick_totals

        # --- Asian handicap (all lines) ---
        sub = ah_best[ah_best["match_id"].eq(mid)].copy()
        if not sub.empty:
            pick_ah = None
            if lh is not None:
                grid = cs_grid(lh, la, kmax=args.grid_k, rho=args.dc_rho)
                sk = skellam_from_grid(grid)
                opts = []
                for (ln, side), grp in sub.groupby(["point_f","side"]):
                    row = grp.sort_values("price", ascending=False).head(1).iloc[0]
                    o = float(row["price"]); bk = str(row["bookmaker_title"])
                    ev = ev_handicap(o, sk, float(ln), side.lower())
                    opts.append((f"{'Home' if side=='home' else 'Away'} {ln:+g}", o, bk, None, ev))
                if opts:
                    best_ev = max(opts, key=lambda x: x[4])
                    pick_ah = {"selection": best_ev[0], "price": best_ev[1], "bookmaker": best_ev[2],
                               "model_prob": best_ev[3], "ev": best_ev[4]}
            else:
                row = sub.sort_values("price", ascending=False).head(1).iloc[0]
                pick_ah = {"selection": f"{row['side'].title()} {row['point_f']:+g}",
                           "price": float(row["price"]), "bookmaker": str(row["bookmaker_title"]),
                           "model_prob": None, "ev": None}
            if pick_ah:
                best["asian_handicap"] = pick_ah

        # --- Match score (Correct Score) ---
        sub = cs_best[cs_best["match_id"].eq(mid)].copy()
        if not sub.empty:
            pick_cs = None
            if lh is not None:
                grid = cs_grid(lh, la, kmax=args.cs_k_max, rho=args.dc_rho)
                sub2 = sub.copy()
                sub2["sc_h"] = pd.to_numeric(sub2["sc_h"], errors="coerce")
                sub2["sc_a"] = pd.to_numeric(sub2["sc_a"], errors="coerce")
                sub2 = sub2.dropna(subset=["sc_h","sc_a"])
                sub2 = sub2[(sub2["sc_h"]<=args.cs_k_max) & (sub2["sc_a"]<=args.cs_k_max)]
                if not sub2.empty:
                    sub2["p"] = sub2.apply(lambda r: float(grid[int(r["sc_h"]), int(r["sc_a"])]), axis=1)
                    sub2["ev"] = sub2["p"]*sub2["price"] - 1.0
                    row = sub2.sort_values("ev", ascending=False).head(1).iloc[0]
                    pick_cs = {"selection": f"{int(row['sc_h'])}-{int(row['sc_a'])}",
                               "price": float(row["price"]), "bookmaker": str(row["bookmaker_title"]),
                               "model_prob": float(row["p"]), "ev": float(row["ev"])}
            if pick_cs is None:
                sub["sc_h"] = pd.to_numeric(sub["sc_h"], errors="coerce")
                sub["sc_a"] = pd.to_numeric(sub["sc_a"], errors="coerce")
                sub = sub.dropna(subset=["sc_h","sc_a"])
                row = sub.sort_values("price", ascending=False).head(1).iloc[0]
                pick_cs = {"selection": f"{int(row['sc_h'])}-{int(row['sc_a'])}",
                           "price": float(row["price"]), "bookmaker": str(row["bookmaker_title"]),
                           "model_prob": None, "ev": None}
            best["match_score"] = pick_cs

        out_rows.append({**meta, "best_value_picks": best})

    # ---- write JSON ----
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"all_market_picks_next_epl_{season}_mw{mws[0]:02d}.json"
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "epl_matchweek": {"season": season, "mw": mws},
        "window_uk": {"start": iso_or_none(fixtures['kickoff_uk'].min()), "end": iso_or_none(fixtures['kickoff_uk'].max())},
        "matches_count": len(out_rows),
        "matches": out_rows
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote {out_json} with {len(out_rows)} matches")

if __name__ == "__main__":
    main()
