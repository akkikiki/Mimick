#! /bin/bash

#SBATCH --job-name=polyglot_train
#SBATCH --output=polyglot_train.out
#SBATCH --error=polyglot_train.err
#SBATCH --partition=titan
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load gcc_4.9.2

python3 model.py \
    --dataset polyglot_en.out \
    --vocab ../vocabs/en-vocab.txt \
    --model-out model.out \
    --output output.out
#    --dataset GoogleNews.out \
