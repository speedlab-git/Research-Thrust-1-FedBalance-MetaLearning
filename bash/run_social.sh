#https://levelup.gitconnected.com/forecasting-crime-rates-with-time-series-analysis-9c0d7893011b

#!/usr/bin/env bash
set -e

CSV="dataset/merged_crime_tweets_v.csv"

python crimeTweetNet.py \
  --csv_path "$CSV" \
  --text_col tweet \
  --label_col category \
  --model_name roberta-base \
  --max_len 128 \
  --num_clients 5 \
  --rounds 50 \
  --local_epochs 1 \
  --batch_size 16 \
  --lr 2e-5
