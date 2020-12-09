#!/bin/bash
LAYERS=$1
ROTATIONS=$2
PERM=$3
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-20:00      # time (DD-HH:MM)
#SBATCH --output=hyper-fixed-mnist-args_dropout-.2%N-%j.out  # %N for node name, %j for jobID
#SBATCH --gres=gpu:v100:1 

module load cuda cudnn
source ../tensorflow/bin/activate

python run_demo_hyper.py --layers $LAYERS --rotations $ROTATIONS --use_dropout True --total_runs 28 --epochs 300 --permutation_arrangement $PERM

