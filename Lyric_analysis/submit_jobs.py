# https://wikispaces.psu.edu/display/CyberLAMP/System+Information
# In general 1 GPU per node.

import os
import numpy as np
import itertools

n_layers = [1,2,3]
lstm_size = [128,256,512]
batch_size = [64,128,256,512]

params = list(itertools.product(*[n_layers, lstm_size, batch_size]))

submit_jobs = 1
jobs_dir = 'jobs'
counter = 0
for nl,lstms,bs in params:
    output_name = 'train_sl150_nl%d_size%d_bs%d'%(nl,lstms,bs)
    job_name = '%s.pbs'%output_name
    with open('%s/%s'%(jobs_dir,job_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#PBS -l nodes=1:gpus=1\n')
        f.write('#PBS -l walltime=48:00:00\n')
        f.write('#PBS -l pmem=12gb\n')
        f.write('#PBS -A cyberlamp -l qos=cl_open\n')
        f.write('#PBS -j oe\n')
        f.write('cd $PBS_O_WORKDIR\n')
        f.write('module load python/3.3.2\n')
        f.write('export PATH="/storage/work/ajs725/conda/install/bin:$PATH"\n\n')
        f.write('CUDA_VISIBLE_DEVICES=0 python train_lstm_char.py %d %d %d > output/%s.txt'%(nl,lstms,bs,output_name))
    f.close()

    if submit_jobs == 1:
        os.system('mv %s/%s %s'%(jobs_dir,job_name, job_name))
        os.system('qsub %s'%job_name)
        os.system('mv %s %s/%s'%(job_name,jobs_dir,job_name))
        counter += 1
