import logging
import time

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.utils.loggers import log_num_trainable_params, setup_logger, timeit
from ml.common.utils.register_model import register_from_checkpoint
from ml.custom.ttz.ttz_dataset import ttzDataModule
from ml.custom.ttz.process_ttz_dataset import ttzNpyProcessor, ttzFeatureSelector
from ml.flows.models import (
    MADEMOG,
    MAF,
    MAFMADEMOG,
    FlowModel,
    MOGFlowModel,
)
from ml.flows.trackers import FlowTracker as Tracker


# ---------------------------------------------------------------------------
# Data-adaptive model scaling
# ---------------------------------------------------------------------------

def _data_size_tier(n_train: int) -> tuple[str, float]:
    """Return (tier_name, scale_factor) based on training set size.

    Tiers are defined by the number of training events so that the model
    capacity roughly scales with the available data, reducing overfitting
    and wasted compute for the smaller _neg subsets.

      large  (≥ 1 000 000)  →  scale 1.00  (full default config)
      medium (≥   200 000)  →  scale 0.67
      small  (≥    50 000)  →  scale 0.50
      tiny   (         < 50 000)  →  scale 0.20  (more aggressive for BSM weight subsets)
    """
    if n_train >= 1_000_000:
        return "large", 1.0
    elif n_train >= 200_000:
        return "medium", 0.67
    elif n_train >= 50_000:
        return "small", 0.50
    else:
        return "tiny", 0.20


def _scale_config_to_data(n_train: int, model_conf: dict, data_module) -> dict:
    """Scale model architecture and batch size to match training set size.

    Modifies ``data_module.dataloader_kwargs`` in-place to update batch_size.
    Returns a *modified copy* of ``model_conf`` (plain dict, safe to mutate).

    Scaled parameters
    -----------------
    - hidden_layer_dim / hidden_layer_mog_dim  (rounded to nearest 32)
    - num_flows
    - num_hidden_layers / num_hidden_layers_mog_net
    - batch_size  (rounded to nearest 128)

    Parameters left unchanged
    -------------------------
    - n_mixtures  (kept at default; changing base distribution complexity
      has a larger quality impact than capacity)
    - res_layers_in_block  (kept at 2; residual blocks are cheap)
    - batchnorm_flow, conv1x1, activation, etc.
    """
    tier, scale = _data_size_tier(n_train)
    model_conf = dict(model_conf)  # mutable copy so caller's dict is unchanged

    if tier == "large":
        logging.info(
            f"Data scaling: {n_train} train events → tier '{tier}'. "
            "Using default model config and batch size."
        )
        return model_conf

    def _round32(x):
        return max(64, round(x / 32) * 32)

    def _round128(x):
        return max(128, round(x / 128) * 128)

    changes = {}

    if "hidden_layer_dim" in model_conf:
        original = model_conf["hidden_layer_dim"]
        model_conf["hidden_layer_dim"] = _round32(original * scale)
        changes["hidden_layer_dim"] = f"{original} → {model_conf['hidden_layer_dim']}"

    if "hidden_layer_mog_dim" in model_conf:
        original = model_conf["hidden_layer_mog_dim"]
        model_conf["hidden_layer_mog_dim"] = _round32(original * scale)
        changes["hidden_layer_mog_dim"] = f"{original} → {model_conf['hidden_layer_mog_dim']}"

    if "num_flows" in model_conf:
        original = model_conf["num_flows"]
        model_conf["num_flows"] = max(2, round(original * scale))
        changes["num_flows"] = f"{original} → {model_conf['num_flows']}"

    if "num_hidden_layers" in model_conf:
        original = model_conf["num_hidden_layers"]
        model_conf["num_hidden_layers"] = max(2, round(original * scale))
        changes["num_hidden_layers"] = f"{original} → {model_conf['num_hidden_layers']}"

    if "num_hidden_layers_mog_net" in model_conf:
        original = model_conf["num_hidden_layers_mog_net"]
        model_conf["num_hidden_layers_mog_net"] = max(2, round(original * scale))
        changes["num_hidden_layers_mog_net"] = f"{original} → {model_conf['num_hidden_layers_mog_net']}"

    # Batch size: update the dataloader directly (model_conf doesn't own batch_size)
    current_batch = data_module.dataloader_kwargs.get("batch_size", 1024)
    scaled_batch = _round128(current_batch * scale)
    data_module.dataloader_kwargs["batch_size"] = scaled_batch
    changes["batch_size"] = f"{current_batch} → {scaled_batch}"

    logging.info(
        f"Data scaling: {n_train} train events → tier '{tier}' (scale={scale:.2f}). "
        "Config changes: " + ", ".join(f"{k}: {v}" for k, v in changes.items())
    )
    return model_conf


class PeriodicEpochLogger(Callback):
    """Logs training metrics every N epochs."""
    
    def __init__(self, log_every_n_epochs=5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        # Log every N epochs or on first epoch
        if current_epoch % self.log_every_n_epochs == 0 or current_epoch == 0:
            # Get losses from logged metrics
            train_loss = trainer.callback_metrics.get('train_loss', float('nan'))
            val_loss = trainer.callback_metrics.get('val_loss', float('nan'))
            
            # Get early stopping callback to access patience counter
            early_stop_callback = None
            for callback in trainer.callbacks:
                if isinstance(callback, EarlyStopping):
                    early_stop_callback = callback
                    break
            
            if early_stop_callback is not None:
                wait_count = early_stop_callback.wait_count
                patience = early_stop_callback.patience
                logging.info(
                    f"Epoch {current_epoch:4d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Patience: {wait_count}/{patience}"
                )
            else:
                logging.info(
                    f"Epoch {current_epoch:4d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )


class FinalEpochLogger(Callback):
    """Logs only the final epoch information when training ends."""
    
    def on_train_end(self, trainer, pl_module):
        logging.info(f"Training completed at epoch {trainer.current_epoch}")
        logging.info(f"Total steps: {trainer.global_step}")
        logging.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
        logging.info(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")


@timeit(unit="min")
@hydra.main(config_path="config/", config_name="main_config", version_base=None)
def main(config):
    setup_logger()
    
    # Set MLflow tracking URI at the very start to ensure correct working directory
    import mlflow
    import os
    mlflow.set_tracking_uri(f"file://{os.path.abspath('./mlruns')}")
    logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # get configuration
    experiment_conf = config.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    experiment_name = "flows"

    data_conf = config.data_config
    model_conf = config.model_config
    training_conf = config.training_config

    # match model postfix to rescale type
    experiment_conf["model_postfix"] = data_conf["preprocessing"]["cont_rescale_type"]

    if data_conf["preprocessing"]["disc_rescale_type"] is not None:
        experiment_conf["model_postfix"] += f"_{data_conf['preprocessing']['disc_rescale_type']}"

    # matmul precision and seed
    torch.set_float32_matmul_precision("high")
    L.seed_everything(experiment_conf["seed"], workers=True)

    # data processing 
    npy_proc = ttzNpyProcessor(
        data_dir="ml/data/ttz/", 
        base_file_name="ttz", 
        list_data_features=data_conf["feature_selection"]["keep_names"],
        load_weights=data_conf.get("load_weights", False),  # Enable weight loading if specified
        **data_conf["input_processing"]
    ) 

    f_sel = ttzFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    pre = Preprocessor(**data_conf["preprocessing"])

    chainer = ProcessorChainer(npy_proc, f_sel, pre)

    # create a data module
    data_module = ttzDataModule(
        chainer,
        train_split=data_conf["train_split"],
        val_split=data_conf["val_split"],
        use_weights=data_conf.get("use_weights", False),  # Enable weight usage in training
        weight_type=data_conf.get("weight_type", "unknown"),
        **data_conf["dataloader_config"],
    )

    # Run setup early so we know n_train before building the model.
    # ttzDataModule.setup() is idempotent — Lightning's later call is a no-op.
    data_module.setup("fit")
    n_train = len(data_module.train)

    # Convert Hydra DictConfig → plain dict so we can mutate it for scaling.
    from omegaconf import OmegaConf
    model_conf = OmegaConf.to_container(model_conf, resolve=True)

    # Scale model architecture and batch_size to match this run's dataset size.
    model_conf = _scale_config_to_data(n_train, model_conf, data_module)

    # model configuration
    logging.info(f"Setting up {model_conf['model_name']} model.")

    # Log key configuration parameters (may differ from yaml if scaling was applied)
    logging.info(f"batch_size: {data_module.dataloader_kwargs.get('batch_size')}")
    if 'num_flows' in model_conf:
        logging.info(f"num_flows: {model_conf['num_flows']}")
    if 'num_hidden_layers' in model_conf:
        logging.info(f"num_hidden_layers: {model_conf['num_hidden_layers']}")
    if 'hidden_layer_dim' in model_conf:
        logging.info(f"hidden_layer_dim: {model_conf['hidden_layer_dim']}")
    if 'res_layers_in_block' in model_conf:
        logging.info(f"res_layers_in_block: {model_conf['res_layers_in_block']}")
    if 'n_mixtures' in model_conf:
        logging.info(f"n_mixtures: {model_conf['n_mixtures']}")

    # https://arxiv.org/abs/1705.07057
    if model_conf["model_name"].lower() == "maf":
        model = MAF(model_conf, data_conf, experiment_conf)

    elif model_conf["model_name"].lower() == "mafmademog":
        model = MAFMADEMOG(model_conf, data_conf, experiment_conf)

    # https://arxiv.org/abs/1306.0186
    elif model_conf["model_name"].lower() == "mademog":
        model = MADEMOG(model_conf, data_conf, experiment_conf)

    else:
        raise NameError(f"Unknown model: {model_conf['model_name']}")

    tracker = Tracker(experiment_conf, tracker_path="ml/custom/ttz/metrics")

    logging.info("Done model setup.")

    log_num_trainable_params(model, unit="k")

    if model_conf["model_name"].lower() not in ["mademog", "mafmademog"]:
        flow = FlowModel(model_conf, training_conf, data_conf, model, tracker=tracker)
    else:
        flow = MOGFlowModel(model_conf, training_conf, data_conf, model, tracker=tracker)

    # Adjust early stopping patience for small datasets: reduce gradient noise impact
    # by giving more tolerance when training data is scarce (high variance in validation loss)
    base_patience = (
        experiment_conf["epochs"]
        if training_conf["early_stop_patience"] is None
        else training_conf["early_stop_patience"]
    )
    if n_train < 5000:
        actual_patience = min(base_patience + 5, 20)  # Allow 5 more epochs, cap at 20
        logging.info(
            f"Small dataset ({n_train} events): increasing early stop patience from "
            f"{base_patience} to {actual_patience} to reduce gradient noise impact."
        )
    else:
        actual_patience = base_patience

    # define callbacks
    callbacks = [
        #TQDMProgressBar(),    # WARNING: Turn off when running on batch system to avoid huge log files
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=actual_patience,
        ),
        ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss"),
        PeriodicEpochLogger(log_every_n_epochs=5),
        FinalEpochLogger(),
    ]
 
    # initialize mlflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f'{model_conf["model_name"]}_{experiment_conf["run_name"]}',
        save_dir=experiment_conf["save_dir"],
        log_model=True,
    )

    # define trainer
    trainer = L.Trainer(
        max_epochs=training_conf["epochs"],
        accelerator=experiment_conf["accelerator"],
        devices=experiment_conf["devices"],
        check_val_every_n_epoch=experiment_conf["check_eval_n_epoch"],
        log_every_n_steps=experiment_conf["log_every_n_steps"],
        num_sanity_val_steps= experiment_conf["num_sanity_val_steps"],
        precision=experiment_conf["precision"],
        logger=mlf_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        enable_progress_bar=False,  # WARNING: Turn off (False) when running on batch system to avoid huge log files
    )

    # Set model name
    if experiment_conf["model_postfix"] is not None:
        model_name = f"{model_conf['model_name']}_flow_model_{experiment_conf['model_postfix']}"
    else:
        model_name = f"{model_conf['model_name']}_flow_model"

    # run training
    ckpt_path = experiment_conf.get("resume_checkpoint", None)
    if ckpt_path:
        logging.info(f"Resuming training from checkpoint: {ckpt_path}")
    trainer.fit(flow, data_module, ckpt_path=ckpt_path)

    # Create detailed model name with date, weight_type and n_data
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    n_data_str = str(data_conf['feature_selection']['n_data']) if data_conf['feature_selection']['n_data'] is not None else "all"
    weight_type_str = data_conf.get("weight_type", "full")
    detailed_model_name = f"{model_name}_{date_str}_cHt5_{weight_type_str}_n{n_data_str}"
    
    # save model
    register_from_checkpoint(trainer, flow, model_name=detailed_model_name)


if __name__ == "__main__":
    main()

