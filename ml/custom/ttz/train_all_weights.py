#!/usr/bin/env python
"""
Submit a separate Condor training job for each SMEFT weight type.

Weight types trained:
  sm            - SM weight only (w_sm from eventWeight)
  linear_pos    - Positive part of the linear SMEFT component (w_lin >= 0)
  linear_neg    - Negative part of the linear SMEFT component (|w_lin| for w_lin < 0)
  quadratic_pos - Positive part of the quadratic SMEFT component (w_quad >= 0)
  quadratic_neg - Negative part of the quadratic SMEFT component (|w_quad| for w_quad < 0)

The pos/neg split allows normalising flows (which require non-negative weights) to model
the signed SMEFT measure exactly.  The combination in smeft_reweighting.py is:
  w(c) ∝ Z_sm + c*(Z_lin_pos*r_lin_pos - Z_lin_neg*r_lin_neg)
               + c²*(Z_quad_pos*r_quad_pos - Z_quad_neg*r_quad_neg)

Each model is registered in MLflow with a name encoding the weight type, e.g.:
  MAFMADEMOG_flow_model_gauss_rank_20260303_cHt5_linear_pos_nall

Jobs run asynchronously on the cluster; monitor with:
  condor_q

Usage:
  python ml/custom/ttz/train_all_weights.py
  python ml/custom/ttz/train_all_weights.py --weight-types sm linear   # subset
  python ml/custom/ttz/train_all_weights.py --queue short              # override queue
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # MLHEPsim root
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / f"train_all_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ---------------------------------------------------------------------------
# Condor settings (mirrors run_bash.sh)
# ---------------------------------------------------------------------------
CONDOR_JOB_PREFIX = "mafmademog"
CONDOR_QUEUE      = "medium" # Default queue; can be overridden with --queue
CONDOR_N_NODES    = 8
CONDOR_N_GPUS     = 1
CONDOR_MEMORY_MB  = 64000

VENV_ACTIVATE = str(PROJECT_ROOT / "venv311" / "bin" / "activate")

# Paths whose old run logs are cleaned before new submissions.
# Do NOT delete condor wrapper scripts in condor/mveldijk/condorsub, because
# queued jobs may still reference those absolute paths at execute time.
CLEANUP_PATTERNS = [
    str(PROJECT_ROOT / "run" / f"{CONDOR_JOB_PREFIX}*.condorlog"),
    str(PROJECT_ROOT / "run" / f"{CONDOR_JOB_PREFIX}*.log"),
    str(PROJECT_ROOT / "run" / f"{CONDOR_JOB_PREFIX}*.stderr"),
]

# ---------------------------------------------------------------------------
# Logging – console + timestamped file
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    fmt     = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE),
        ],
    )

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight types
# ---------------------------------------------------------------------------
ALL_WEIGHT_TYPES = ["sm", "linear_pos", "linear_neg", "quadratic_pos", "quadratic_neg"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cleanup_old_condor_files() -> None:
    """Remove stale condor log/submission files from previous runs."""
    removed = 0
    for pattern in CLEANUP_PATTERNS:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                removed += 1
            except OSError as e:
                log.warning(f"Could not remove {f}: {e}")
    log.info(f"Cleaned up {removed} old condor file(s).")


def build_job_command(weight_type: str) -> str:
    """Build the shell command string that Condor will execute for one weight type."""
    return (
        f"source /etc/profile && "
        f"source {VENV_ACTIVATE} && "
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH && "
        f"export PYTHONPATH=$PYTHONPATH:{PROJECT_ROOT} && "
        f"cd {PROJECT_ROOT} && "
        f"python -m ml.custom.ttz.main_flows "
        f"data_config.weight_type={weight_type} "
        f"data_config.load_weights=True "
        f"data_config.use_weights=True "
        f"training_config.use_sm_weights=True "
        # Request enough CPU threads to keep the GPU fed and avoid idle watchdog holds.
        f"data_config.dataloader_config.num_workers=4 "
        f"data_config.dataloader_config.pin_memory=True "
        # Avoid long CPU-only tracker phases on batch nodes with GPU-idle watchdogs.
        f"experiment_config.check_metrics_n_epoch=1000000"
    )


def submit_job(weight_type: str, queue: str) -> bool:
    """
    Submit one condor job for the given weight_type.
    Returns True if submission succeeded (exit code 0).
    """
    job_name = f"{CONDOR_JOB_PREFIX}_{weight_type}"
    job_cmd  = build_job_command(weight_type)

    condorsub_cmd = [
        "condorsub",
        "-J", job_name,
        "-q", queue,
        "-n", str(CONDOR_N_NODES),
        "-g", str(CONDOR_N_GPUS),
        "-m", str(CONDOR_MEMORY_MB),
        job_cmd,
    ]

    log.info(f"Submitting job: {job_name}")
    log.info(f"  condorsub command: {' '.join(condorsub_cmd[:8])} ...")

    result = subprocess.run(condorsub_cmd, cwd=str(PROJECT_ROOT))

    if result.returncode == 0:
        log.info(f"  -> Submitted successfully: {job_name}")
        return True
    else:
        log.error(f"  -> Submission FAILED for {job_name} (return code {result.returncode})")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit one Condor training job per SMEFT weight type."
    )
    parser.add_argument(
        "--weight-types",
        nargs="+",
        choices=ALL_WEIGHT_TYPES,
        default=ALL_WEIGHT_TYPES,
        metavar="TYPE",
        help=f"Weight types to submit. Choices: {ALL_WEIGHT_TYPES}. Default: all.",
    )
    parser.add_argument(
        "--queue",
        default=CONDOR_QUEUE,
        help=f"Condor queue to use (default: {CONDOR_QUEUE}).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove old mafmademog run logs before submitting (disabled by default).",
    )
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 60)
    log.info("train_all_weights.py – SMEFT weight type sweep (Condor)")
    log.info(f"Project root  : {PROJECT_ROOT}")
    log.info(f"Log file      : {LOG_FILE}")
    log.info(f"Weight types  : {args.weight_types}")
    log.info(f"Condor queue  : {args.queue}")
    log.info("=" * 60)

    if args.cleanup:
        cleanup_old_condor_files()
    else:
        log.info("Cleanup disabled by default; preserving historical files in run/.")

    results: dict[str, bool] = {}
    for weight_type in args.weight_types:
        results[weight_type] = submit_job(weight_type, args.queue)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("Submission summary")
    log.info("=" * 60)
    any_failed = False
    for wt, ok in results.items():
        status = "SUBMITTED" if ok else "FAILED"
        log.info(f"  {wt:12s}  {status}")
        if not ok:
            any_failed = True
    log.info("=" * 60)
    log.info("Jobs are running asynchronously. Monitor with: condor_q")
    log.info(f"Submission log written to: {LOG_FILE}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
