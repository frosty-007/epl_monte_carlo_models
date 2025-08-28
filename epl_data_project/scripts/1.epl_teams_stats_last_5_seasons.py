#!/usr/bin/env python3
"""
Fetch EPL team match data + shot data from Understat since 2020 up to the current season.

- Seasons determined by UK season start year (Aug roll-over).
- Shots fetched ONLY for completed matches (avoid 404 on future fixtures).
- Casts season to str, and match id to str (fixes type errors in understatapi).
- Robust to 404/InvalidMatch; skips gracefully with backoff.
- Deduplicates matches & shots.
- Adds features: 5-match rolling xG/xGA, rest_days, derby flag.
- Saves CSV + Parquet under ../data/raw/game_stats/
- Also writes convenience copies:
    ../data/raw/all_matches.parquet
    ../data/raw/all_shots.parquet
"""

import argparse
import asyncio
import os
import random
import time
from typing import List

import pandas as pd
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException

try:
    from understatapi import UnderstatClient
    from understatapi.exceptions import InvalidMatch
except Exception:
    UnderstatClient = None
    class InvalidMatch(Exception): ...
    pass

# ======================
# Defaults / Config
# ======================
DEFAULT_LEAGUE = "EPL"
DEFAULT_OUT_DIR = "../data/raw/game_stats"
DERBY_PAIRS = {
    ("Liverpool", "Everton"),
    ("Arsenal", "Tottenham"),
    ("Chelsea", "Tottenham"),
    ("Manchester_United", "Manchester_City"),
    ("Chelsea", "Fulham"),
    ("Aston_Villa", "Birmingham_City"),
    ("Newcastle_United", "Sunderland"),
    ("Liverpool", "Manchester_United"),
}

# ======================
# Helpers
# ======================

def season_years_since(start_year: int = 2020, tz: str = "Europe/London") -> List[int]:
    """List of season start-years from start_year up to current season (Aug roll-over)."""
    now_uk = pd.Timestamp.now(tz=tz)
    current = now_uk.year if now_uk.month >= 8 else now_uk.year - 1
    return list(range(start_year, current + 1))

def slugify(name: str) -> str:
    """Convert Understat team title to a slug the client accepts."""
    return (
        str(name).strip()
        .replace(" ", "_")
        .replace("&", "and")
        .replace("'", "")
        .replace(".", "")
    )

def coerce_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# ======================
# Fetchers
# ======================

async def fetch_team_slugs(season, league: str) -> List[str]:
    """Get team slugs from league match list (collect titles of h/a)."""
    def _sync():
        with UnderstatClient() as client:
            matches = client.league(league).get_match_data(season=str(season)) or []
        slugs = set()
        for m in matches:
            for side in ("h", "a"):
                team_obj = m.get(side) or {}
                title = team_obj.get("title")
                if title:
                    slugs.add(slugify(title))
        return sorted(slugs)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync)

async def fetch_match_data(team_slug: str, season) -> pd.DataFrame:
    """Fetch match-level data for a team & season; flatten goals/xG/xGA; derive is_home/opponent."""
    def _sync():
        with UnderstatClient() as client:
            data = client.team(team=team_slug).get_match_data(season=str(season)) or []
        df = pd.DataFrame(data)
        if df.empty:
            return df

        # datetime → UTC ts; also provide a 'date' column for ordering
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["date"] = df.get("datetime", pd.NaT)

        df["team"] = team_slug

        # is_home from 'side' if present
        if "side" in df.columns:
            df["is_home"] = df["side"].map({"h": True, "a": False})
        else:
            df["is_home"] = pd.NA

        # opponent via h/a titles
        opp = []
        home_names, away_names = [], []
        for _, r in df.iterrows():
            h_obj = r.get("h")
            a_obj = r.get("a")
            h_title = h_obj.get("title") if isinstance(h_obj, dict) else h_obj
            a_title = a_obj.get("title") if isinstance(a_obj, dict) else a_obj
            home_names.append(h_title)
            away_names.append(a_title)
            h_slug = slugify(h_title) if h_title else None
            a_slug = slugify(a_title) if a_title else None
            if pd.notna(r.get("is_home")) and isinstance(r.get("is_home"), (bool, pd.BooleanDtype().type)):
                opp.append(a_slug if r["is_home"] else h_slug)
            else:
                opp.append(a_slug if (h_slug == team_slug) else h_slug if (a_slug == team_slug) else None)
        df["opponent"] = opp
        df["home_title"] = home_names
        df["away_title"] = away_names

        # isResult best effort
        if "isResult" in df.columns:
            df["isResult"] = df["isResult"].astype("boolean")
        else:
            def has_final_goals(g):
                if isinstance(g, dict):
                    return str(g.get("h","")).isdigit() and str(g.get("a","")).isdigit()
                return False
            df["isResult"] = df.get("goals", pd.Series([None]*len(df))).apply(has_final_goals).astype("boolean")

        # Final goals (home/away)
        h_goals, a_goals = [], []
        for _, r in df.iterrows():
            g = r.get("goals")
            if isinstance(g, dict) and {"h","a"} <= set(g.keys()):
                try:
                    h_goals.append(int(g.get("h"))); a_goals.append(int(g.get("a")))
                except Exception:
                    h_goals.append(pd.NA); a_goals.append(pd.NA)
            else:
                h_goals.append(pd.NA); a_goals.append(pd.NA)
        df["h_goals"] = h_goals
        df["a_goals"] = a_goals

        # xG/xGA flatten into team perspective
        xg_vals, xga_vals = [], []
        for _, r in df.iterrows():
            v = r.get("xG")
            if isinstance(v, dict) and {"h","a"} <= set(v.keys()):
                if bool(r.get("is_home")):
                    xg_vals.append(coerce_float(v.get("h"))); xga_vals.append(coerce_float(v.get("a")))
                elif r.get("is_home") is False:
                    xg_vals.append(coerce_float(v.get("a"))); xga_vals.append(coerce_float(v.get("h")))
                else:
                    xg_vals.append(coerce_float(v.get("h"))); xga_vals.append(coerce_float(v.get("a")))
            else:
                xg_vals.append(coerce_float(r.get("xG"))); xga_vals.append(coerce_float(r.get("xGA")))
        df["xG_num"] = pd.to_numeric(xg_vals, errors="coerce")
        df["xGA_num"] = pd.to_numeric(xga_vals, errors="coerce")

        # id as Int64 (nullable)
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

        return df

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync)

async def fetch_shot_data(
    match_ids: List[int],
    max_retries: int = 4,
    base_backoff: float = 1.0,
    timeout_secs: float = 15.0,
) -> pd.DataFrame:
    """Fetch shot-level data for given Understat match IDs with retry/backoff; skip future/invalid."""
    def _sync():
        rows = []
        with UnderstatClient() as client:
            for mid in match_ids:
                for attempt in range(max_retries):
                    try:
                        # IMPORTANT: Understat requires match id as STRING
                        shots = client.match(match=str(mid)).get_shot_data()
                        for side in ("h", "a"):
                            s = pd.DataFrame(shots.get(side, []))
                            if not s.empty:
                                s["match_id"] = int(mid)
                                s["team_side"] = side
                                rows.append(s)
                        break  # success; next match

                    except InvalidMatch:
                        print(f"    ! shots {mid}: invalid/absent Understat match page; skipping")
                        break

                    except HTTPError as e:
                        status = getattr(e.response, "status_code", None)
                        if status == 404:
                            print(f"    ! shots {mid}: 404 (no page yet); skipping")
                            break
                        if status and status not in (429, 500, 502, 503, 504):
                            print(f"    ! shots {mid}: HTTP {status}; skipping")
                            break
                        if attempt == max_retries - 1:
                            print(f"    ! shots {mid}: HTTP error after retries ({e})")
                            break
                        sleep_for = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(sleep_for)

                    except (ConnectionError, Timeout) as e:
                        if attempt == max_retries - 1:
                            print(f"    ! shots {mid}: network failed after retries ({e})")
                            break
                        sleep_for = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(sleep_for)

                    except RequestException as e:
                        print(f"    ! shots {mid}: request error ({e}); skipping")
                        break

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync)

# ======================
# Features
# ======================

def compute_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-game rolling averages for xG and xGA per team slug."""
    if df.empty: return df
    out = df.sort_values(["team", "date"]).copy()
    out["rolling_xg"] = out.groupby("team")["xG_num"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    out["rolling_xga"] = out.groupby("team")["xGA_num"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    return out

def add_contextual(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest_days + derby flags (lightweight)."""
    if df.empty: return df
    out = df.sort_values(["team", "date"]).copy()
    out["rest_days"] = out.groupby("team")["date"].diff().dt.days.fillna(0).astype(int)
    out["is_derby"] = out.apply(
        lambda r: (r.get("home_title"), r.get("away_title")) in DERBY_PAIRS
                  or (r.get("away_title"), r.get("home_title")) in DERBY_PAIRS,
        axis=1
    )
    out["euro_travel"] = False  # placeholder
    return out

# ======================
# CLI / Pipeline
# ======================

def parse_args():
    ap = argparse.ArgumentParser(description="Understat EPL matches+shots since 2020 (robust + dedup + features).")
    ap.add_argument("--league", default=DEFAULT_LEAGUE)
    ap.add_argument("--since", type=int, default=2020, help="Start season (by start year)")
    ap.add_argument("--seasons", nargs="*", type=int, help="Explicit seasons (overrides --since)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between teams")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=15.0)
    return ap.parse_args()

def run_full_pipeline(args):
    async def _main():
        seasons = sorted(set(args.seasons)) if args.seasons else season_years_since(args.since)
        print(f"Seasons: {seasons}")
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)

        all_matches: List[pd.DataFrame] = []
        all_shots: List[pd.DataFrame] = []

        for season in seasons:
            print(f"\nSeason {season}: fetching team slugs…")
            try:
                team_slugs = await fetch_team_slugs(str(season), args.league)  # pass as str
            except Exception as e:
                print(f"  ✘ failed to fetch team slugs for {season}: {e}")
                continue

            if not team_slugs:
                print(f"  ✘ no slugs found for {season}, skipping")
                continue

            for slug in team_slugs:
                print(f"  • {slug}: fetching matches… ", end="", flush=True)
                try:
                    df_m = await fetch_match_data(slug, str(season))  # pass as str
                except Exception as e:
                    print(f"✘ ({e})")
                    await asyncio.sleep(0.8)
                    continue

                if df_m.empty:
                    print("– no matches")
                    await asyncio.sleep(0.5)
                    continue

                # Completed fixtures mask
                if "isResult" in df_m.columns:
                    done_mask = df_m["isResult"].astype("boolean").fillna(False)
                else:
                    def has_final_goals(g):
                        return isinstance(g, dict) and str(g.get("h","")).isdigit() and str(g.get("a","")).isdigit()
                    done_mask = df_m.get("goals", pd.Series([None]*len(df_m))).apply(has_final_goals)

                n_all = len(df_m)
                n_done = int(done_mask.sum()) if isinstance(done_mask, pd.Series) else n_all
                print(f"✓ ({n_all} fixtures, {n_done} completed)")
                all_matches.append(df_m)

                # Shots only for completed matches (avoid 404)
                ids = df_m.loc[done_mask, "id"].dropna().astype(int).tolist()
                if ids:
                    print(f"    shots: fetching for {len(ids)} matches…")
                    df_s = await fetch_shot_data(ids, max_retries=args.max_retries, timeout_secs=args.timeout)
                    if not df_s.empty:
                        print(f"    shots: ✓ {len(df_s)} rows")
                        all_shots.append(df_s)
                    else:
                        print("    shots: – none/failed")
                else:
                    print("    shots: – 0 completed match ids to fetch")

                await asyncio.sleep(args.sleep)  # polite gap between teams

        if not all_matches:
            raise RuntimeError("No match data collected; aborting.")

        df_all_matches = pd.concat(all_matches, ignore_index=True)
        df_all_shots = pd.concat(all_shots, ignore_index=True) if all_shots else pd.DataFrame()

        # --- matches: keep BOTH sides of each match ---
        if "id" in df_all_matches.columns:
            # prefer explicit team column if present
            team_col = "team" if "team" in df_all_matches.columns else None
            if team_col:
                # ✅ one row per (match id, team)
                df_all_matches = (
                    df_all_matches
                    .drop_duplicates(subset=["id", team_col], keep="first")
                    .reset_index(drop=True)
                )
            else:
                # fallback: try to infer team column names from Understat shapes
                possible_team_cols = [c for c in ["h_team", "a_team", "home_team", "away_team"] if c in df_all_matches.columns]
                if possible_team_cols:
                    # still dedup by id + whichever team col(s) exist
                    df_all_matches = (
                        df_all_matches
                        .drop_duplicates(subset=["id"] + possible_team_cols, keep="first")
                        .reset_index(drop=True)
                    )
                else:
                    # last resort: do NOT dedup by only id (would lose one side); skip dedup
                    print("[warn] No team column found; skipping match-level dedup to avoid dropping a side.")
        else:
            # no match id? just a conservative full-row dedup
            df_all_matches = df_all_matches.drop_duplicates().reset_index(drop=True)

        # --- shots: keep unique shot ids; never dedup on match_id alone ---
        if not df_all_shots.empty:
            if "id" in df_all_shots.columns:
                # Understat shot-level id → safest key
                df_all_shots = (
                    df_all_shots
                    .drop_duplicates(subset=["id"], keep="first")
                    .reset_index(drop=True)
                )
            else:
                # build a fingerprint; include match + side + basic shot fields
                fp_cols = [c for c in [
                    "match_id", "h_a", "minute", "X", "Y", "player", "result", "shotType", "situation", "lastAction"
                ] if c in df_all_shots.columns]
                if fp_cols:
                    df_all_shots = df_all_shots.drop_duplicates(subset=fp_cols, keep="first").reset_index(drop=True)
                else:
                    # fallback: trivial dedup
                    df_all_shots = df_all_shots.drop_duplicates().reset_index(drop=True)

        # --- quick sanity diagnostics ---
        vc = df_all_matches.groupby("id").size() if "id" in df_all_matches.columns else pd.Series(dtype=int)
        if not vc.empty:
            print("[diag] per-match row counts value_counts():")
            print(vc.value_counts().head())
            print("[diag] proportion of matches with both sides (2 rows):", float((vc == 2).mean()))


        # Features
        print("Computing rolling features…")
        df_all_matches = compute_rolling(df_all_matches)
        print("Adding contextual features…")
        df_all_matches = add_contextual(df_all_matches)

        # Save CSV
        print("Saving CSV…")
        path_matches_csv = os.path.join(out_dir, "all_matches.csv")
        path_shots_csv   = os.path.join(out_dir, "all_shots.csv")
        df_all_matches.to_csv(path_matches_csv, index=False)
        df_all_shots.to_csv(path_shots_csv, index=False)
        print(f"  - {path_matches_csv}")
        print(f"  - {path_shots_csv}")

        # Save Parquet + convenience copies
        try:
            print("Saving Parquet…")
            path_matches_parq = os.path.join(out_dir, "all_matches.parquet")
            path_shots_parq   = os.path.join(out_dir, "all_shots.parquet")
            df_all_matches.to_parquet(path_matches_parq, index=False, compression="snappy")
            df_all_shots.to_parquet(path_shots_parq, index=False, compression="snappy")
            print(f"  - {path_matches_parq}")
            print(f"  - {path_shots_parq}")

            # convenience copies for the rest of your pipeline
            root_dir = os.path.abspath(os.path.join(out_dir, ".."))
            root_matches = os.path.join(root_dir, "all_matches.parquet")
            root_shots   = os.path.join(root_dir, "all_shots.parquet")
            df_all_matches.to_parquet(root_matches, index=False, compression="snappy")
            df_all_shots.to_parquet(root_shots, index=False, compression="snappy")
            print(f"  - {root_matches}")
            print(f"  - {root_shots}")
        except Exception as e:
            print(f"Note: failed Parquet write (install pyarrow/fastparquet?). Error: {e}")

        print("\nPipeline complete!")
        print(f"Matches: {len(df_all_matches):,} | Shots: {len(df_all_shots):,}")

    asyncio.run(_main())

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    args = parse_args()
    run_full_pipeline(args)
