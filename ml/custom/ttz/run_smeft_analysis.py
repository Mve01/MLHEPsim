#!/usr/bin/env python
"""
Run the ttZ sample analyzer for each trained SMEFT weight-type model.

For each weight type (sm, linear_pos, linear_neg, quadratic_pos, quadratic_neg) this script:
  1. Finds the latest MLflow model whose name contains 'cHt5_<weight_type>'
  2. Creates a dated output directory: sample_analyzer/figures/smeft_<weight_type>_<date>/
  3. Runs run_analysis.py locally (sequentially)

Usage:
  python ml/custom/ttz/run_smeft_analysis.py
  python ml/custom/ttz/run_smeft_analysis.py --weight-types sm linear_pos linear_neg
"""

import argparse
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
ANALYZER_DIR = PROJECT_ROOT / "ml" / "custom" / "ttz" / "sample_analyzer"
FIGURES_DIR  = ANALYZER_DIR / "figures"
LOGS_DIR     = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / f"smeft_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE),
        ],
    )

log = logging.getLogger(__name__)

ALL_WEIGHT_TYPES = ["sm", "linear_pos", "linear_neg", "quadratic_pos", "quadratic_neg"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_latest_model(weight_type: str) -> str | None:
    """
    Search the MLflow model registry for the most recently registered model
    whose name contains 'cHt5_<weight_type>'.
    Returns the model name, or None if nothing is found.
    """
    import mlflow
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    client = mlflow.tracking.MlflowClient()

    matching = []
    try:
        for rm in client.search_registered_models():
            if f"cHt5_{weight_type}" in rm.name:
                versions = client.get_latest_versions(rm.name)
                if versions:
                    matching.append((rm.name, versions[0].creation_timestamp))
    except Exception as e:
        log.warning(f"Error querying MLflow registry: {e}")

    if not matching:
        log.warning(f"No MLflow model found containing 'cHt5_{weight_type}'")
        return None

    matching.sort(key=lambda x: x[1], reverse=True)
    best = matching[0][0]
    log.info(f"  Found model for '{weight_type}': {best}")
    return best


def make_figures_dir(weight_type: str) -> Path:
    """Create and return a dated output directory for this weight type's figures."""
    timestamp = datetime.now().strftime("%Y%m%d")
    out_dir = FIGURES_DIR / f"smeft_{weight_type}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Figures directory: {out_dir}")
    return out_dir


def run_analysis(weight_type: str, model_name: str, figures_dir: Path) -> bool:
    """Run run_analysis.py locally for the given model and figures directory."""
    cmd = [
        sys.executable,
        str(ANALYZER_DIR / "run_analysis.py"),
        "--model-name", model_name,
        "--figures-dir", str(figures_dir),
    ]

    log.info(f"Running analysis for weight_type='{weight_type}'")
    log.info(f"  Model      : {model_name}")
    log.info(f"  Figures dir: {figures_dir}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode == 0:
        log.info(f"  -> Analysis for '{weight_type}' completed successfully.")
        return True
    else:
        log.error(f"  -> Analysis for '{weight_type}' FAILED (return code {result.returncode})")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ttZ analyzer for each trained SMEFT weight-type model."
    )
    parser.add_argument(
        "--weight-types",
        nargs="+",
        choices=ALL_WEIGHT_TYPES,
        default=ALL_WEIGHT_TYPES,
        metavar="TYPE",
        help=f"Weight types to analyse. Choices: {ALL_WEIGHT_TYPES}. Default: all.",
    )
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 60)
    log.info("run_smeft_analysis.py – SMEFT analysis sweep")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Log file     : {LOG_FILE}")
    log.info(f"Weight types : {args.weight_types}")
    log.info("=" * 60)

    results: dict[str, bool] = {}
    for weight_type in args.weight_types:
        log.info(f"\nPreparing analysis for weight_type='{weight_type}'")
        model_name = find_latest_model(weight_type)
        if model_name is None:
            log.error(f"  Skipping '{weight_type}' — no model found in MLflow registry.")
            results[weight_type] = False
            continue

        figures_dir = make_figures_dir(weight_type)
        results[weight_type] = run_analysis(weight_type, model_name, figures_dir)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("Analysis summary")
    log.info("=" * 60)
    any_failed = False
    for wt, ok in results.items():
        status = "OK" if ok else "FAILED/SKIPPED"
        log.info(f"  {wt:12s}  {status}")
        if not ok:
            any_failed = True
    log.info("=" * 60)
    log.info(f"Figures saved in: {FIGURES_DIR}/smeft_<weight_type>_<date>/")
    log.info(f"Log: {LOG_FILE}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
