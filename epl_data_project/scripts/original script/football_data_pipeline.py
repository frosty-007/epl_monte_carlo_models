import pandas as pd
import asyncio
import os
import requests
from understatapi import UnderstatClient

# ----------- MATCH-LEVEL DATA -----------
async def fetch_match_data(team: str, season: str) -> pd.DataFrame:
    def sync_fetch():
        with UnderstatClient() as c:
            return c.team(team=team).get_match_data(season=season) or []

    data = await asyncio.get_event_loop().run_in_executor(None, sync_fetch)
    df   = pd.DataFrame(data)
    df['team'] = team

    # 1. Timestamp: datetime or date?
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise KeyError("Missing 'datetime' or 'date' columns")  # safe guard :contentReference[oaicite:6]{index=6}

    # 2. Dynamic home/away detection
    home_col = next((c for c in ('h_team','home_team','h')   if c in df.columns), None)
    away_col = next((c for c in ('a_team','away_team','a')   if c in df.columns), None)
    if not home_col or not away_col:
        raise KeyError(f"No home/away columns; got {df.columns.tolist()}")  # clear error :contentReference[oaicite:7]{index=7}
    df['is_home'] = df[home_col] == df['team']

    # 3. Preserve and flatten xG
    df['xg_raw']  = df['xG']  # nested dict remains intact
    df['xG_num']  = df.apply(lambda r: float(r['xg_raw']['h']) if r['is_home'] else float(r['xg_raw']['a']), axis=1)
    df['xGA_num'] = df.apply(lambda r: float(r['xg_raw']['a']) if r['is_home'] else float(r['xg_raw']['h']), axis=1)

    return df


# ----------- SHOT-LEVEL DATA -----------
async def fetch_shot_data(ids: list) -> pd.DataFrame:
    def sync_shots():
        with UnderstatClient() as c:
            rows = []
            for mid in ids:
                shot = c.match(match=mid).get_shot_data()
                for side in ('h', 'a'):
                    s = pd.DataFrame(shot.get(side, []))
                    if not s.empty:
                        s['match_id'] = mid
                        s['side']     = side
                        rows.append(s)
            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return await asyncio.get_event_loop().run_in_executor(None, sync_shots)

# ----------- ROLLING FEATURES -----------
def compute_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['team', 'date']).copy()
    df['rolling_xg']  = df.groupby('team')['xG_num'] \
                           .transform(lambda s: s.rolling(5, min_periods=1).mean())
    df['rolling_xga'] = df.groupby('team')['xGA_num'] \
                           .transform(lambda s: s.rolling(5, min_periods=1).mean())
    return df

# ----------- CONTEXTUAL FEATURES -----------
DERBY_PAIRS = [
    ("Liverpool", "Everton"), ("Arsenal", "Tottenham"),
    ("Manchester United", "Manchester City"), ("Chelsea", "Fulham"),
    ("Aston Villa", "Birmingham City"), ("Newcastle", "Sunderland")
]
def add_contextual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['team', 'date'])
    df['rest_days'] = df.groupby('team')['date'].diff().dt.days.fillna(0).astype(int)
    df['is_derby']  = df.apply(lambda r: (r['team'], r.get('opponent')) in DERBY_PAIRS, axis=1)
    df['euro_travel'] = False
    return df

# ----------- MAIN PIPELINE -----------
def run_pipeline(team="Tottenham", season="2024", api_key=None):
    async def async_run():
        os.makedirs("data/processed", exist_ok=True)

        print("Match-level data fetch…")
        df_matches = await fetch_match_data(team, season)

        print("Shot-level data fetch…")
        df_shots = await fetch_shot_data(df_matches['id'].tolist())

        print("Rolling xG/xGA computation…")
        df_matches = compute_rolling(df_matches)

        print("Adding contextual flags…")
        df_matches = add_contextual(df_matches)

        print("Saving…")
        df_matches.to_csv("../data/processed/match_data.csv", index=False)
        df_shots.to_csv("../data/processed/shot_data.csv", index=False)
        df_matches.to_parquet("../data/processed/match_data.parquet",
                              index=False, compression="snappy")
        df_shots.to_parquet("../data/processed/shot_data.parquet",
                            index=False, compression="snappy")

        print("Done!")
    asyncio.run(async_run())

if __name__ == "__main__":
    run_pipeline(api_key=os.getenv("API_SPORTS_KEY"))
