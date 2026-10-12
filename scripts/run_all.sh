#!/usr/bin/env bash
# Reproduces all experiments (3 seeds for the deep models).
set -e
python -m src.build_splits
python -m src.check_leakage
python -m src.baselines
for seed in 42 43 44; do
  for m in mlp cnn resnet; do
    python -m src.train --model $m --seed $seed
    python -m src.train --model $m --augment --seed $seed
  done
done
# Test set is touched only after all training/model selection is finished
for seed in 42 43 44; do
  for m in mlp cnn resnet; do
    python -m src.evaluate --model $m --seed $seed
    python -m src.evaluate --model $m --augment --seed $seed
  done
done
