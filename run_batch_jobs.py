import os

import json


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.jobs" % os.getcwd()
# scratch = os.environ['SCRATCH']
# data_dir = os.path.join(scratch, '/spatical')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(job_directory + '/img')

# mkdir_p(data_dir)

max_ = 5
min_ = 1
jobs = []
for i in range(1, max_, 2):
    for j in range(1, max_, 2):
        jobs.append({"layers": i, "rotations": j,
                     "title": f"tf-vpnn-spatical-mixedlayers-{i}-rotations-{j}"})


for job in jobs:

    job_file = os.path.join(job_directory, "%s.job" % job['title'])

    layers = job['layers']
    rotations = job['rotations']
    mode = 'a' if os.path.exists(job_file) else 'w'
    with open(job_file, mode) as fh:
        fh.writelines(f"""
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-20:00      # time (DD-HH:MM)
#SBATCH --output=hyper-fixed-mnist-layers-{layers}-rotations-{rotations}_dropout-.2%N-%j.out  # %N for node name, %j for jobID
#SBATCH --gres=gpu:v100:1

module load cuda cudnn
source ../tensorflow/bin/activate

python run_demo_hyper.py --layers {layers} --rotations {rotations} --use_dropout True --total_runs 28 --epochs 300 --permutation_arrangement 4
        """)

    os.system("sbatch %s" % job_file)