import os
import numpy as np

seq_length = [25,50,75,100,125,150,175,200]

submit_jobs = 0
jobs_dir = 'jobs'
counter = 0
for sl in seq_length:
    job_name = 'train_lstm_sl%d.pbs'%sl
    with open('%s/%s'%(jobs_dir,job_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#PBS -l nodes=1:gpus=1\n')
        f.write('#PBS -l walltime=48:00:00\n')
        f.write('#PBS -l pmem=8gb\n')
        f.write('#PBS -A cyberlamp -l qos=cl_open\n')
        f.write('#PBS -j oe\n')
        f.write('cd $PBS_O_WORKDIR\n')
        f.write('module load python/3.3.2\n')
        f.write('export PATH="/storage/work/ajs725/conda/install/bin:$PATH"\n\n')
        f.write('CUDA_VISIBLE_DEVICES=0 python train_lstm.py %d > sl%d.txt'%(sl,sl))
    f.close()

    if submit_jobs == 1:
        os.system('mv %s/%s %s'%(jobs_dir,job_name, job_name))
        os.system('qsub %s'%pbs_script_name)
        os.system('mv %s %s/%s'%(job_name,jobs_dir,job_name))
        counter += 1
