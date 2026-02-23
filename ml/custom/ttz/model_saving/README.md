# Model Saving and Management

This directory contains utilities for managing trained MLflow models with user-friendly names.

## Overview

When you train a model, MLflow automatically registers it with a name based on the config (e.g., `MAFMADEMOG_flow_model_gauss_rank_20260218_nall`). These names are descriptive but cumbersome to use. This utility allows you to register models, for later use and easy access, with shorter, memorable names while preserving all metadata (scalers, selection, model architecture).

## Directory Structure

```
model_saving/
├── save_model_with_name.py    # Script to register models with custom names
├── saved_models/               # Directory for any exported model files (optional)
└── README.md                   # This file
```

## Usage

### Basic Usage (Default Settings)

Register the latest trained model with a default name:

```bash
cd /path/to/MLHEPsim
python ml/custom/ttz/model_saving/save_model_with_name.py
```

This will register the model `MAFMADEMOG_flow_model_gauss_rank_20260218_nall` as `ttz_cHt5_phi_scaled`.

### Custom Model Registration

Register a specific model with a custom name:

```bash
python ml/custom/ttz/model_saving/save_model_with_name.py \
  --source MAFMADEMOG_flow_model_gauss_rank_20260218_nall \
  --name my_best_model
```

### Arguments

- `--source`: Source model name from MLflow registry (default: `MAFMADEMOG_flow_model_gauss_rank_20260218_nall`)
- `--name`: New user-friendly name for the model (default: `ttz_cHt5_phi_scaled`)
- `--version`: Specific version of the source model to copy (default: latest version)
- `--description`: Custom description for the registered model (optional)

### Examples

**Example 1: Register your best production model**
```bash
python ml/custom/ttz/model_saving/save_model_with_name.py \
  --source MAFMADEMOG_flow_model_gauss_rank_20260218_nall \
  --name ttz_production \
  --description "Production model with Phi scaling, val_loss=15.5"
```

**Example 2: Register a specific version**
```bash
python ml/custom/ttz/model_saving/save_model_with_name.py \
  --source MAFMADEMOG_flow_model_gauss_rank_20260122_nall \
  --name ttz_sm_baseline \
  --version 2 \
  --description "SM baseline model without BSM weights"
```

**Example 3: Quick registration with default settings**
```bash
python ml/custom/ttz/model_saving/save_model_with_name.py --name ttz_v1
```

## Using Registered Models

Once registered, use your model with the analyzer:

```python
from ml.custom.ttz.sample_analyzer.analyzer import ttzSampleAnalyzer

analyzer = ttzSampleAnalyzer(
    data_dir="ml/data/ttz/ttz.npy",
    variables_json_path="ml/data/ttz/variables.json",
    model_name="ttz_production"  # Use your custom name here!
)

# Generate samples
analyzer.generate_samples(n_samples=100000)
analyzer.plot_all()
```

## What Gets Preserved

When you register a model with a custom name, **everything** is preserved:

- ✅ **Model architecture** (flow layers, hidden dimensions, etc.)
- ✅ **Trained weights** (all parameters)
- ✅ **Scalers** (Gaussian rank transform interpolation functions)
- ✅ **Feature selection** (which features were used, their types, order)
- ✅ **Metadata** (original model name, version, registration date)

This ensures the analyzer uses the **exact same preprocessing pipeline** that was used during training, preventing CDF mismatch issues.

## Common Workflows

### Workflow 1: Keep Best Model After Training

1. Train multiple models with different hyperparameters
2. Evaluate all models using the analyzer
3. Register the best one for production use:
   ```bash
   python ml/custom/ttz/model_saving/save_model_with_name.py \
     --source MAFMADEMOG_flow_model_gauss_rank_20260218_nall \
     --name ttz_best_v1
   ```

### Workflow 2: Version Control for Production

1. Register your current production model:
   ```bash
   python ml/custom/ttz/model_saving/save_model_with_name.py \
     --source <current_model> \
     --name ttz_production_v1
   ```
2. Train a new improved model
3. Register it as v2:
   ```bash
   python ml/custom/ttz/model_saving/save_model_with_name.py \
     --source <new_model> \
     --name ttz_production_v2
   ```
4. Compare v1 vs v2 using the analyzer
5. Switch production to v2 if better

### Workflow 3: Preserve Important Experiments

After completing an experiment, save it with a descriptive name:

```bash
python ml/custom/ttz/model_saving/save_model_with_name.py \
  --source MAFMADEMOG_flow_model_gauss_rank_20260218_nall \
  --name ttz_phi_scaling_experiment \
  --description "First model with Phi angles Gaussian-rank scaled. Val loss: 15.5"
```

## Listing Registered Models

To see all your registered models:

```python
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.MlflowClient()

for model in client.search_registered_models():
    print(f"{model.name}")
    for version in client.search_model_versions(f"name='{model.name}'"):
        print(f"  - Version {version.version}: {version.description}")
```

Or from command line:
```bash
cd mlruns/models && ls -d */
```

## Troubleshooting

**Error: "No versions found for model"**
- The source model name doesn't exist in MLflow registry
- Check available models: `ls mlruns/models/`

**Error: "Failed to register model"**
- Make sure you're in the project root directory
- Verify MLflow tracking URI is set to `file:./mlruns`

**Warning: "Other features detected"**
- This is normal if you have weight columns or non-processed features
- Only affects features explicitly marked in `no_process` config

## Notes

- Registering a model with an existing name will create a new version (not overwrite)
- The source model is never modified - this creates a new registry entry
- All preprocessing state (scalers, selection) is preserved in the new registration
- Models are stored in `mlruns/` directory - back this up if needed

## Related Files

- **Analyzer**: `ml/custom/ttz/sample_analyzer/analyzer.py`
- **Training**: `ml/custom/ttz/main_flows.py`
- **Data Config**: `ml/custom/ttz/config/flows/data_config.yaml`
- **Preprocessing**: `ml/common/data_utils/processors.py`
