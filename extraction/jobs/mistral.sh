#!/bin/bash
#SBATCH --cpus-per-task=18
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -t 01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --job-name=mistral

module load 2021
module load 2022
module load Python/3.9.5-GCCcore-10.3.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

pip3 install --user torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user transformers==4.34.1 tokenizers datasets typeguard==2.13.3

rsync -r $HOME/priming/* $TMPDIR/priming
mkdir $TMPDIR/priming/scores

cd $TMPDIR/priming

models="mistralai/Mistral-7B-v0.1"

date

for model in $models; do
 echo $model
 srun python3 main.py --model $model --data data --save scores &
done
wait

date

rsync -r scores/* $HOME/prime_scores
