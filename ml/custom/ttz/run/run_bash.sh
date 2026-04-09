#!/bin/bash

cd /project/atlas/users/mveldijk/LLoCagenerator/MLHEPsim

# Optional cleanup (disabled by default to preserve historical run logs).
# Enable only when explicitly requested:
#   CLEANUP_OLD_RUNS=1 bash ml/custom/ttz/run/run_bash.sh
if [[ "${CLEANUP_OLD_RUNS:-0}" == "1" ]]; then
	rm -f /project/atlas/users/mveldijk/LLoCagenerator/MLHEPsim/run/mafmademog*
	rm -f /project/atlas/users/mveldijk/LLoCagenerator/MLHEPsim/condor/mveldijk/condorsub/mafmademog*
	rm -f /project/atlas/users/mveldijk/LLoCagenerator/MLHEPsim/condor/mveldijk/condorsub/enviromentvariables*
fi

condorsub -J mafmademog -q short -n 8 -g 1 -m 36000 \
"source /etc/profile && \
source /project/atlas/users/mveldijk/LLoCagenerator/MLHEPsim/venv311/bin/activate && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH && \
export PYTHONPATH=$PYTHONPATH:/project/atlas/users/mveldijk/LLoCagenerator/MLHEPsim && \
cd /project/atlas/users/mveldijk/LLoCagenerator/MLHEPsim && \
python -m ml.custom.ttz.main_flows data_config.dataloader_config.num_workers=4 data_config.dataloader_config.pin_memory=True experiment_config.check_metrics_n_epoch=1000000"
