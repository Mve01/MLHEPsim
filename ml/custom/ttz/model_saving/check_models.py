#!/usr/bin/env python
"""Quick script to check model architectures."""
import torch
import mlflow

# Check the 3 versions of Feb 18 model
run_ids = {
    'Version 1 (12:36)': 'b860ee92e94046108a2a28b2ee8abbb0',
    'Version 2 (14:27)': '699090eb33834030b4a9e35eb547be3e',
    'Version 3 (16:31)': 'd6bd6a31ed6b4d62835c91fd3d6b85ca',
}

print('Feb 18 Model Architectures:')
print('='*60)
for label, run_id in run_ids.items():
    try:
        path = f'mlruns/0/{run_id}/artifacts/model/data/model.pth'
        ckpt = torch.load(path, map_location='cpu')
        n_flows = ckpt['hyper_parameters']['num_flows']
        hidden_dims = ckpt['hyper_parameters']['hidden_layer_dim']
        print(f'{label}: {n_flows} flows × {hidden_dims} dims')
    except Exception as e:
        print(f'{label}: ERROR - {e}')

# Check what we actually saved
print('\n' + '='*60)
print('Currently saved models:')
print('='*60)

client = mlflow.MlflowClient()
for model_name in ['ttz_small_8x512_phi_scaled', 'ttz_medium_10x768_phi_scaled']:
    try:
        version = client.get_model_version(model_name, 1)
        run_id = version.run_id
        path = f'mlruns/0/{run_id}/artifacts/model/data/model.pth'
        ckpt = torch.load(path, map_location='cpu')
        n_flows = ckpt['hyper_parameters']['num_flows']
        hidden_dims = ckpt['hyper_parameters']['hidden_layer_dim']
        print(f'{model_name}:')
        print(f'  Actually: {n_flows} flows × {hidden_dims} dims')
    except Exception as e:
        print(f'{model_name}: ERROR - {e}')
