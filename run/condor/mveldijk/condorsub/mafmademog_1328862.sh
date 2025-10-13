#!/bin/bash

export SCRIPT=/project/atlas/users/mveldijk/MLHEPsim/run/condor/mveldijk/condorsub/mafmademog_1328862.sh
export SCRIPTsub=/project/atlas/users/mveldijk/MLHEPsim/run/condor/mveldijk/condorsub/mafmademog_1328862.sub
export enviromentvariablescript=/project/atlas/users/mveldijk/MLHEPsim/run/condor/mveldijk/condorsub/enviromentvariables_1328862.sh
export PATH=/project/atlas/Users/mveldijk/MLHEPsim/venv311/bin:/data/atlas/users/mveldijk/.pyenv/plugins/pyenv-virtualenv/shims:/data/atlas/users/mveldijk/.pyenv/shims:/data/atlas/users/mveldijk/.pyenv/bin:/project/atlas/Users/mveldijk/.vscode-server/cli/servers/Stable-385651c938df8a906869babee516bffd0ddb9829/server/bin/remote-cli:/user/mveldijk/bin:/user/mveldijk/.local/bin:/data/atlas/users/mveldijk/.pyenv/plugins/pyenv-virtualenv/shims:/data/atlas/users/mveldijk/.pyenv/bin:/data/atlas/users/mveldijk/.pyenv/plugins/pyenv-virtualenv/shims:/data/atlas/users/mveldijk/.pyenv/bin:/usr/share/Modules/bin:/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/rocm/bin:/user/mveldijk/bin/:/project/atlas/users/mveldijk/miniforge/bin/:/opt/rocm/bin:/user/mveldijk/bin/:/project/atlas/users/mveldijk/miniforge/bin/:/user/mveldijk/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/opt/rocm/bin:/user/mveldijk/bin/:/project/atlas/users/mveldijk/miniforge/bin/
cd /project/atlas/users/mveldijk/MLHEPsim/run
source /project/atlas/users/mveldijk/MLHEPsim/run/condor/mveldijk/condorsub/enviromentvariables_1328862.sh

source /etc/profile && source /project/atlas/users/mveldijk/MLHEPsim/venv311/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export LD_LIBRARY_PATH=/.singularity.d/libs:/data/atlas/users/mveldijk/local3/lib && export PYTHONPATH=/project/atlas/Users/mveldijk/MLHEPsim:/project/atlas/users/mvedlijk/MLHEPsim && cd /project/atlas/users/mveldijk/MLHEPsim && python -m ml.custom.HIGGS.main_flows hydra.job.chdir=false

