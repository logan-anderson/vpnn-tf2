import os
import time
from args import CommandLineArgs

args = CommandLineArgs()


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.jobs" % os.getcwd()
file = os.path.join(os.getcwd(), 'run_demo_hyper.py')
# scratch = os.environ['SCRATCH']
# data_dir = os.path.join(scratch, '/spatical')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(job_directory + '/img')

# mkdir_p(data_dir)

max_ = 6
min_ = 1
jobs = []
for i in range(1, 10):
    for j in range(1, max_):
        jobs.append({"layers": i, "rotations": j,
                     "title": f"tf-vpnn-spatical-mixedlayers-{i}-rotations-{j}"})


for job in jobs:

    job_file = os.path.join(job_directory, "%s.job" % job['title'])

    layers = job['layers']
    rotations = job['rotations']
    if os.path.exists(job_file):
        os.remove(job_file)
    with open(job_file, 'a') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"""#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-20:00      # time (DD-HH:MM)
#SBATCH --output=hyper-batch-fixed-mnist-perm-{args.permutation_arrangement}-layers-{layers}-rotations-{rotations}_dropout-.2%N-%j.out  # %N for node name, %j for jobID
#SBATCH --gres=gpu:v100:1

module load cuda cudnn
source ../tensorflow/bin/activate

python {file} --layers {layers} --rotations {rotations} --use_dropout True --total_runs 28 --epochs 300 --permutation_arrangement {args.permutation_arrangement}
        """)
    time.sleep(.5)
    os.system("sbatch %s" % job_file)
