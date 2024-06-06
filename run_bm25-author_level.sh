#!/bin/bash
source activate kai
export CUDA_VISIBLE_DEVICES=0;
# allocate at least 160GB CPU memory, must use compute-02/03 for 48GB GPU memory
# sbatch -p rush --nodelist=rush-compute-03 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
# --mem=256G -t 720:00:00 run_bm25-author_level.sh
python /home/ys724/NLP/NLP/run_bm25-author_level.py
