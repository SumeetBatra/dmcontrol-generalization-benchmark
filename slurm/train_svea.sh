#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=tmp/svea_dmcgb_%j.log

export MUJOCO_GL=egl

set -- 0
DOMAIN="walker"
TASK="walk"
for seed in "$@";
  do echo "Running seed $seed"
  RUN_NAME="svea_${DOMAIN}_${TASK}_seed_${seed}_coloreasy"
  python -m src.train --algorithm=svea \
                           --wandb_group=svea_${DOMAIN}_${TASK}_baseline \
                           --wandb_name=$RUN_NAME \
                           --domain_name=$DOMAIN \
                           --task_name=$TASK \
                           --train_steps=500k \
                           --seed="$seed"
done