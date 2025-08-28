id: Match ID (integer).

isResult: true if the match has finished (scores are final).

side: The perspective of this row — "h" = home, "a" = away.
(This row is from Arsenal’s point of view: side: "a".)

h / a: Info about the home/away clubs: { id, short_title (abbr), title (full name) }.

Here: home = Fulham, away = Arsenal.

goals: Final score by side as strings: {"h": "0", "a": "3"}.

xG: Expected goals by side as strings: {"h": "0.126327", "a": "2.16287"}.

datetime: Kickoff timestamp in milliseconds since epoch (UTC).
Convert with pd.to_datetime(value, unit="ms", utc=True).

forecast: Pre-match probabilities (should sum to ~1).
Keys: d = draw, l = loss, w = win — from the perspective of team/side.
(Sanity-check this in your data: verify the largest of {w,l,d} usually matches result.)

result: 'w'/'l'/'d' from the row team’s perspective (win/loss/draw).

date: Duplicate of datetime (also ms since epoch).

team / opponent: Names from the row perspective.
(team: "Arsenal", opponent: "Fulham").

is_home: true if this row’s team was home; here false (Arsenal were away).

home_title / away_title: Full club names again (redundant with h.title / a.title).

h_goals / a_goals: Final score as numbers (ints).

xG_num / xGA_num: This row’s team xG for and xG against as floats.
(Here, Arsenal xG ≈ 2.16, conceded xG ≈ 0.13.)

rolling_xg / rolling_xga: Rolling aggregates of xG for/against (sum or avg; check your upstream).
(In your example they equal the single-match xG — likely first game of the roll).

rest_days: Days since this team’s previous match.

is_derby: True if marked a derby (boolean flag from your source).

euro_travel: True if the team had European travel burden (boolean).