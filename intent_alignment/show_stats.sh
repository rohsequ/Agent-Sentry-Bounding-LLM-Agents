#!/usr/bin/env bash
set -e

python3 drift_defense/calculate_drift_agent_sentry_stats.py \
  --input_dir agent_sentry_dataset_drift_results/
