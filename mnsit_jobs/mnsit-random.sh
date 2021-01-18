#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-6:00      # time (DD-HH:MM)
#SBATCH --output=mnist-normal-random-chebyshev-adam-batch-128-layers-1-rotatiosn-2-dropout-.2%N-%j.out  # %N for node name, %j forjobID
#SBATCH --gres=gpu:v100:1 

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r ../requirements.txt --no-deps

pip install /home/logana/scratch/vpnn-tf2/dist/vpnn_tf2-1.0.0-py3-none-any.whl

module load cuda cudnn 
 
cd ..

python run_demo_mnist.py --layers 1 --hidden_activation chebyshev\
        --name mnistSpatialV --tensorboard --cheby_M 2.0\
         --rotations 2 --optimizer adam --epochs 100 --batch_size 128\
         --permutation_arrangement 3  --use_dropout True