#!/usr/bin/env bash
set -e

CSV="dataset/criminal_report.csv"


set -e

OUT="./risk_prior_fl_out"

python crimeReport_lag.py \
  --csv_path "$CSV" \
  --time_col "Date" \
  --loc_col "Location Description" \
  --window "1H" \
  --high_quantile 0.70 \
  --min_windows_per_loc 50 \
  --num_clients 5 \
  --rounds 50 \
  --local_epochs 1 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --out_dir "$OUT" \
  --save_every_round

