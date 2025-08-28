  source .venv_epl_data_project/bin/activate

  python -c "import sys; print('prefix:', sys.prefix); print('base_prefix:', getattr(sys, 'base_prefix', getattr(sys, 'real_prefix', None))); print('In venv?', sys.prefix != getattr(sys, 'base_prefix', getattr(sys, 'real_prefix', None)))"
  
  pip install -r requirements.txt

  python -m pip install -U pip setuptools wheel
  python -m pip uninstall -y statsmodels
  python -m pip install "statsmodels>=0.14.3"

Callibration Tests

python calibrate_lambda.py --sanity
simplify
python calibrate_lambda.py --simple --sanity



prices – compute fair probabilities from λ.
backtest – verify calibration/log-lik on history.
value – only once you’re happy with backtest, compare to bookmaker odds.
simulate – optional; sanity check or scenario runs.

martketspipeline

How to use it

A) One-pass “prices + odds already set”

python markets_pipeline.py prices \
  --lambdas ../data/callibrated/team_match_lambdas.parquet \
  --odds ../data/raw/odds.parquet \
  --out ../data/callibrated/market_prices.parquet


The output now includes fair probs/odds and your bookmaker odds, plus implied probs, edges, and Kelly — all “already set” in one file.

Expected odds columns (decimal):

match_id, odds_bk_home, odds_bk_draw, odds_bk_away
# optional extras:
odds_bk_btts_yes, odds_bk_over25, odds_bk_under25


B) Backtest without a prior prices file

python markets_pipeline.py backtest \
  --lambdas ../data/callibrated/team_match_lambdas.parquet \
  --out ../data/callibrated/backtest_detail.parquet


(If --prices is omitted, it will build fair prices internally.)

C) Simulate (per-match MC)

python markets_pipeline.py simulate \
  --prices ../data/callibrated/market_prices.parquet \
  --out ../data/callibrated/mc_results.parquet \
  -n 200000

Backtest = for evaluation/tuning. It compares past probs to past results (gives you Brier/log-lik, helps pick params). Not for picking today’s bets.
Simulate = synthetic outcomes from your own model (great for bankroll/variance planning), not for value detection.
Prices = your current fair probabilities for upcoming fixtures → exactly what you need to compare vs live odds and surface value.