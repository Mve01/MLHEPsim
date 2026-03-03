import copy
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from ml.common.data_utils.feature_scaling import (
    RescalingHandler,
    rescale_continuous_data,
    rescale_discrete_data,
)
from ml.common.data_utils.utils import load_dataset_variables


class NpyProcessor(ABC):
    def __init__(self, data_dir, base_file_name):
        """Base class for npy processors.

        Parameters
        ----------
        data_dir : str
            Downloaded in this directory (needs to exist).
        base_file_name : str, optional
            Base file name.

        Other parameters
        ----------------
        npy_file : str
            File name of .npy file.
        features : dict (loaded from variables.json in data_dir)
            Dictionary of column names and types.

        """
        self.data_dir = data_dir
        self.base_file_name = base_file_name
        self.npy_file = os.path.join(self.data_dir, self.base_file_name + ".npy")
        self.features = load_dataset_variables(self.data_dir)

    def __call__(self, *args, **kwargs) -> tuple[str, dict[str, str | list[str]]]:
        dataset = self.get_dataset()

        if dataset is None:
            logging.info(f"{self.npy_file} already exists!")
        else:
            dataset = self.process_dataset(dataset)
            self.make_npy_file(dataset)

        return self.npy_file, self.features

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def process_dataset(self):
        pass

    @abstractmethod
    def make_npy_file(self):
        pass


class FeatureSelector(ABC):
    def __init__(
        self,
        file_path=None,
        features=None,
        drop_types=None,
        drop_names=None,
        drop_labels=None,
        keep_names=None,
    ):
        """Base class for feature selection.

        Parameters
        ----------
        file_path : str or None
            Path to .npy file.
        drop_types : list of str, optional
            Drop these column types (`label`, `disc`, `cont`, `uni`), by default None.
        drop_names : list of str, optional
            Drop these column names, by default None.
        drop_labels : list of bool, optional
            Drop these labels, by default None.
        keep_names : list of str, optional
            Keep these column names, by default None.

        Other parameters
        ----------------
        features : dict
            Dictionary of column names and types.
        types : list of str
            Types of columns (`label`, `disc`, `cont`, `uni`).
        selection : pd.DataFrame
            Dataframe of column names and types with selection (True or False).

        Note
        ----
        type_lst = [df[df["type"] == t] for t in self.types]

        """
        self.file_path = file_path
        self.features = features
        self.drop_types, self.drop_names, self.drop_labels = drop_types, drop_names, drop_labels
        self.keep_names = keep_names
        self.types = ["label", "disc", "cont", "uni"]
        self.selection = None

    def __call__(self, file_path=None, features=None, **kwargs):
        # this gets loaded from converter if not given in init
        assert not (file_path is None and self.file_path is None), "file_path is None!"
        assert not (features is None and self.features is None), "features are None!"

        if file_path is not None:
            self.file_path = file_path

        if features is not None:
            self.features = features

        self.selection = self._select_colnames()
        data = self.load_data()
        sel_data = self.select_features(data)
        return sel_data, self.selection

    def _select_colnames(self) -> pd.DataFrame:
        """Selects column names to keep.

        Returns
        -------
        pd.DataFrame
            Dataframe of column names and types with selection.

        """
        if self.drop_types is None:
            self.drop_types = []
        if self.drop_names is None:
            self.drop_names = []
        if self.keep_names is None:
            self.keep_names = []

        df = pd.DataFrame(
            self.features["colnames"].items(),
            index=range(len(self.features["colnames"])),
            columns=["feature", "type"],
        )
        df["select"] = [True for _ in range(len(df))]

        df.loc[df["type"].isin(self.drop_types), "select"] = False
        df.loc[df["feature"].isin(self.drop_names), "select"] = False
        df.loc[df["feature"].isin(self.keep_names), "select"] = True

        return df

    @abstractmethod
    def load_data(self) -> np.ndarray:
        logging.info(f"Loading data from {self.file_path}!")
        return np.load(self.file_path)

    @abstractmethod
    def select_features(self, data: np.ndarray) -> np.ndarray:
        select_idx = self.selection[self.selection["select"] == True].index

        data = data[:, select_idx]

        logging.info(f"Selected features! Data shape: {data.shape}.")

        if self.drop_labels is not None:
            labels_idx = self.selection[self.selection["type"] == "label"].index

            for label_idx in labels_idx:
                for drop_label in self.drop_labels:
                    mask_label = data[:, label_idx] == drop_label
                    data = data[~mask_label]
                    logging.info(f"Dropped label {drop_label}! New data shape: {data.shape}.")

        return data


class Preprocessor:
    def __init__(self, cont_rescale_type, disc_rescale_type=None, no_process=None):
        """General preprocessor for continuous and discrete data in numpy arrays.
        
        Features are processed in their original order from the input data,
        preserving physics-motivated feature groupings (e.g., Jets → Leptons → MET).
        
        Both "cont" and "uni" (uniform/periodic) type features are scaled using cont_rescale_type.
        For normalizing flows with Gaussian base distributions, scaling uniform features
        (e.g., Phi angles) to Gaussian makes learning easier (Gaussian → Gaussian vs Gaussian → Uniform).

        Parameters
        ----------
        cont_rescale_type : str
            Rescale type for continuous features (including "uni" type), see `rescale_data` in `ml.common.data_utils.feature_scaling`.
        disc_rescale_type : str, optional
            Rescale type for discrete features, by default None.
        no_process : list of str, optional
            List of column types to not process (e.g., ['weight']), by default None.
        """
        super().__init__()
        self.cont_rescale_type, self.disc_rescale_type = cont_rescale_type, disc_rescale_type
        self.no_process = no_process
        self.selection = None

    def __call__(self, data: np.ndarray, selection: pd.DataFrame, *args, **kwargs):
        logging.info(f"Preprocessing data using cont. {self.cont_rescale_type} and disc. {self.disc_rescale_type} (preserving feature order).")
        return self.preprocess(data, selection)

    def preprocess(self, data: np.ndarray, selection: pd.DataFrame, *args, no_rescale=False, **kwargs):
        """Preprocess data in original order (preserves physics-motivated feature ordering).

        Parameters
        ----------
        data : np.ndarray
            Data to preprocess.
        selection : pd.DataFrame
            Dataframe of column names and types with selection.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame, dict]
            (data, selection, scalers)

        """
        if self.no_process is None:
            self.no_process = []

        # make selections where select is True (drop columns where select is False)
        selection = selection[selection["select"] == True].reset_index(drop=True)

        # Initialize output array (will be filled with scaled features in original positions)
        processed_data = np.zeros_like(data)
        
        # Separate features by type for fitting scalers
        type_mask = selection["type"].isin(self.no_process)
        
        disc_sel = selection[~type_mask & (selection["type"] == "disc")]
        # Include both "cont" and "uni" (uniform/periodic) features for continuous scaling
        # Uniform features (Phi angles) benefit from Gaussian rank scaling: Uniform → Gaussian
        cont_sel = selection[~type_mask & selection["type"].isin(["cont", "uni"])]
        other_sel = selection[type_mask]
        
        # Fit and transform discrete features as a group
        if len(disc_sel) > 0:
            disc_indices = disc_sel.index.tolist()
            disc_data = data[:, disc_indices]
            if no_rescale:
                disc_scaled = disc_data
                disc_scaler = [("none", None)]
            else:
                disc_scaled, disc_scaler = rescale_discrete_data(disc_data, self.disc_rescale_type)
            # Place scaled discrete features back in their original positions
            for i, idx in enumerate(disc_indices):
                processed_data[:, idx] = disc_scaled[:, i]
            logging.debug(f"{self.disc_rescale_type} scaled {len(disc_indices)} discrete features")
        else:
            disc_scaler = None
        
        # Fit and transform continuous features as a group (includes both "cont" and "uni")
        if len(cont_sel) > 0:
            cont_indices = cont_sel.index.tolist()
            cont_data = data[:, cont_indices]
            if no_rescale:
                cont_scaled = cont_data
                cont_scaler = [("none", None)]
            else:
                cont_scaled, cont_scaler = rescale_continuous_data(cont_data, self.cont_rescale_type)
            # Place scaled continuous features back in their original positions
            for i, idx in enumerate(cont_indices):
                processed_data[:, idx] = cont_scaled[:, i]
            logging.debug(f"{self.cont_rescale_type} scaled {len(cont_indices)} continuous/uniform features")
        else:
            cont_scaler = None
        
        # Copy other features (weights, etc.) unchanged
        if len(other_sel) > 0:
            other_indices = other_sel.index.tolist()
            for idx in other_indices:
                processed_data[:, idx] = data[:, idx]
            logging.debug(f"Kept {len(other_indices)} other features unchanged")
            
            # Mark skipped features as "other" type so RescalingHandler knows not to scale them
            for idx in other_indices:
                original_type = selection.loc[idx, "type"]
                selection.loc[idx, "type"] = "other"
            logging.debug(f"Updated selection: marked {len(other_indices)} no_process features as 'other' type")
        
        # make scalers dictionary
        scalers = {"disc": disc_scaler, "cont": cont_scaler}
        self.selection = selection

        return processed_data, selection, scalers

class SeparateLabelPreprocessor(Preprocessor):
    def __init__(self, cont_rescale_type, disc_rescale_type=None, no_process=None):
        """Rescales for each label separately and returns a dict with labels as keys and scalers as values."""
        super().__init__(cont_rescale_type, disc_rescale_type, no_process)

    def __call__(self, data: np.ndarray, selection: pd.DataFrame, *args, **kwargs):
        logging.info(f"Preprocessing data using cont. {self.cont_rescale_type} and disc. {self.disc_rescale_type} (preserving feature order, separate by label).")
        return self.label_preprocess(data, selection)

    def label_preprocess(self, data: np.ndarray, selection: pd.DataFrame):
        logging.warning("Separate rescale labels is True! Will rescale labels separately.")
        label_scalers = {"disc": [], "cont": []}

        label_idx = selection[selection["type"] == "label"].index[0]

        data_fit = []
        label_row = data[:, label_idx]
        unique_labels = np.unique(label_row)

        disc_scalers, cont_scalers = dict(), dict()

        for label in unique_labels:
            label_mask = label_row == label
            data_masked = data[label_mask]

            data_fit_masked, new_selection, scalers = self.preprocess(data_masked, copy.deepcopy(selection))

            disc_scalers[label] = scalers["disc"]
            cont_scalers[label] = scalers["cont"]

            data_fit.append(data_fit_masked)

        label_scalers["disc"].append(disc_scalers)
        label_scalers["cont"].append(cont_scalers)

        data_fit = np.concatenate(data_fit, axis=0)

        self._log_separate_labels(label_scalers)

        return data_fit, new_selection, label_scalers

    @staticmethod
    def _log_separate_labels(label_scalers: dict[str, list[dict[str, Any]]]):
        try:
            cont_info_str, disc_info_str = "", ""
            for s in label_scalers["cont"]:
                for k, v in s.items():
                    cont_info_str += f"{k} - {v.__class__.__name__}, "

            for s in label_scalers["disc"]:
                for k, v in s.items():
                    disc_info_str += f"{k} - {v.__class__.__name__}, "

            logging.info(f"Cont label scalers: {cont_info_str[:-2]}.")
            logging.info(f"Disc label scalers: {disc_info_str[:-2]}.")

        except Exception as e:
            logging.warning(f"Could not log separate label scalers: {e}")


class SingleLabelPreprocessor(Preprocessor):
    def __init__(self, cont_rescale_type, scaler_label, disc_rescale_type=None, no_process=None):
        """Will rescale all data to data column that has scaler_label in label column.

        Example
        -------
        Have signal (1) and background (0) labels. Set scaler_label to 1. This then calculates scalers for signal data
        and applies it to background data as well.

        Parameters
        ----------
        scaler_label : int
            Label to use for rescaling. Only one label can be used for rescaling.

        """
        super().__init__(cont_rescale_type, disc_rescale_type, no_process)
        self.scaler_label = scaler_label

    def __call__(self, data: np.ndarray, selection: pd.DataFrame, *args, **kwargs):
        return self.single_label_preprocess(data, selection)

    def single_label_preprocess(self, data: np.ndarray, selection: pd.DataFrame):
        label_idx = selection[selection["type"] == "label"].index[0]

        label_row = data[:, label_idx]

        label_mask = label_row == self.scaler_label
        data_masked, data_other = data[label_mask], data[~label_mask]

        logging.info(f"Rescaling using label {self.scaler_label}. Label split: {data_masked.shape}, {data_other.shape}")

        data_masked, new_selection, scalers = self.preprocess(data_masked, copy.deepcopy(selection))
        data_other, _, _ = self.preprocess(data_other, copy.deepcopy(selection), no_rescale=True)

        rescale_handler = RescalingHandler(new_selection, copy.deepcopy(scalers))

        data_other = rescale_handler.transform(data_other)

        data = np.concatenate((data_masked, data_other), axis=0)

        data = shuffle(data)

        return data, new_selection, scalers


class AddFeatureProcessor:
    def __init__(self, feature_pack: dict[str, tuple[Callable, list[str]]]) -> None:
        """Initilaize AddFeatureProcessor. This class is used to add transformed features to the dataset.

        Parameters
        ----------
        feature_pack : dict[str, tuple[Callable, list[str]]]
            The structure of the dictionary is as follows:
            - key: name of the new feature,
            - value: tuple of two elements:
                - function that transforms the features,
                - list of features that are used as arguments.

        Example:
        ---
        ```
        feature_pack = {
            "Jet1Px_square": (lambda x: x**2, ["Jet1Px"]),
            "Eta_diff": (lambda x, y: np.abs(x - y), ["Jet1Eta", "Jet2Eta"]),
        }
        ```
        """
        self.feature_pack = feature_pack
        for i in feature_pack.values():
            assert isinstance(i, tuple), "The value of the dictionary must be a tuple."
            assert isinstance(i[0], Callable), "The first element of the tuple must be a function."
            assert (
                len(i[1]) == i[0].__code__.co_argcount
            ), "The number of arguments must be equal to the number of features."
            assert all(isinstance(j, str) for j in i[1]), "The second element of the tuple must be a list of strings."
            assert isinstance(i[1], list), "The second element of the tuple must be a list."

    def __call__(self, data: np.ndarray, selection: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """The function that adds new features to the dataset.

        Parameters
        ----------
        data : np.ndarray
            The dataset.
        selection : pd.DataFrame
            The dataframe with the information about the dataset.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame]
            The dataset with new features and the updated dataframe.
        """
        selection = selection[selection["select"] == True].reset_index(drop=True)
        for key, value in self.feature_pack.items():
            if not (set(value[1]) <= set(selection["feature"])):
                print("Feature(s) {} is(are) not (all) in the dataset.".format(value[1]))
                continue
            new_column = value[0](*[data[:, selection[selection["feature"] == f].index] for f in value[1]])
            data = np.append(data, new_column, axis=1)
            selection = pd.concat(
                [
                    selection,
                    selection[selection["feature"] == value[1][0]].replace(value[1][0], key),
                ],
                ignore_index=True,
            )
        return data, selection


class ProcessorChainer:
    def __init__(self, *args):
        """Chain processors together."""
        self.processors = args

    @staticmethod
    def _x_str(x):
        x_str, get_str = "", lambda x_obj: f"{x_obj.__class__.__name__}, "

        for i in x if type(x) in [tuple, list] else [x]:
            x_str += get_str(i)

        return str({x_str[:-2]})

    def __call__(self, x=None):
        for processor in self.processors:
            logging.info(f"[b][green]RUNNING:[/green][/b] {processor.__class__.__name__} on inputs {self._x_str(x)}")
            logging.debug(f"Processor attrs: {[(k, v.__class__.__name__) for k, v in processor.__dict__.items()]}")

            if isinstance(x, tuple):
                x = processor(*x)
            else:
                x = processor(x)

        return x
