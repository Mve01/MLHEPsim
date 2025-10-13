#!/bin/bash

cd /project/atlas/users/mveldijk/MLHEPsim/run

# Remove the existing condor log file
rm /project/atlas/users/mveldijk/MLHEPsim/run/mafmademog* /project/atlas/users/mveldijk/MLHEPsim/run/condor/mveldijk/condorsub/*

condorsub -J mafmademog -q short -n 1 -g 1 -m 16000 \
"source /etc/profile && \
source /project/atlas/users/mveldijk/MLHEPsim/venv311/bin/activate && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH && \
export PYTHONPATH=$PYTHONPATH:/project/atlas/users/mvedlijk/MLHEPsim && \
cd /project/atlas/users/mveldijk/MLHEPsim && \
python -m ml.custom.HIGGS.main_flows hydra.job.chdir=false"
