from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.toymodels import structural_causal_processes as toys



class AdjacencyMatrix:
    """
    Adjacency matrix representation of causal graph, based on time series data
    Weight qualifies the strength of the causal link
    Initialized with unnormalized adjacency matrices (link_matrix, val_matrix, adj_matrix)
    """ 

## Conver to parent neighbor coefficient style
    # def preprocess_json_to_pnc(self, json_file_path: str, local_lag: int = 8, oci_lag: int = 31) -> dict[str, list[tuple[tuple[int, int], float]]]:
    #     """
    #     Preprocess JSON file to extract coefficients and time lags
    #     Args:
    #         json_file_path: path to the JSON file containing the data
    #         local_lag: time lag for local variables (default is 8 days)
    #         oci_lag: time lag for OCI variables (default is 31 days)
    #         Convert JSON data to a dictionary with the format:
    #         { 
    #             variable1: [((sample_index, -time_lag_days), coefficient), ...],
    #             variable2: [((sample_index, -time_lag_days), coefficient), ...], 
    #             ...
    #             target: [((sample_index, 0), value), ...],
    #         }
    #     Returns:
    #         result: dictionary with variable names as keys and lists of tuples as values
    #             Each tuple contains a sample index, -time lag in days, and coefficient
    #     """

    #     result = {}
        
    #     # Check if file exists
    #     if not os.path.exists(json_file_path):
    #         raise FileNotFoundError(f"File not found: {json_file_path}")
        
    #     # Read the JSON file
    #     with open(json_file_path, 'r') as file:
    #         json_data = file.read()
        
    #     # Process one sample at a time to extract coefficients
    #     samples = json_data.strip().split('\n')
        
    #     for sample_idx, sample in enumerate(samples):
    #         data = json.loads(sample)
            
    #         # Process local variables (with 39 time lags * 8 days each)
    #         for var_name, values in data['local_variables'].items():
    #             if var_name not in result:
    #                 result[var_name] = []
                
    #             # For each value in the time series, calculate the time lag in days
    #             for time_lag_idx, coefficient in enumerate(values):
    #                 # Time lag in days: each unit is 8 days
    #                 time_lag_days = -(time_lag_idx + 1) * local_lag
    #                 result[var_name].append(((sample_idx, time_lag_days), coefficient))
            
    #         # Process OCI variables (with 10 time lags * 1 month each)
    #         for var_name, values in data['ocis'].items():
    #             if var_name not in result:
    #                 result[var_name] = []
                
    #             # For each value in the time series, calculate the time lag in days
    #             for time_lag_idx, coefficient in enumerate(values):
    #                 # Time lag in days: each unit is 30 days (1 month)
    #                 time_lag_days = -(time_lag_idx + 1) * oci_lag
    #                 result[var_name].append(((sample_idx, time_lag_days), coefficient))
            
    #         # Process target variable (with time lag of 0 days)
    #         if 'target' in data:
    #             if 'target' not in result:
    #                 result['target'] = []
                
    #             # Add target with time lag of 0 (current value)
    #             result['target'].append(((sample_idx, 0), data['target']))
        
    #     # Print the dictionary in a readable format
    #     for var_name, coefficients in result.items():
    #         print(f"{var_name}:")
    #         for i, ((sample_idx, time_lag), coef) in enumerate(coefficients[:5]):  # Show first 5 entries
    #             print(f"  {i+1}. ((sample_idx={sample_idx}, time_lag={time_lag}), coefficient={coef})")
    #         print(f"  ... and {len(coefficients)-5} more entries\n")

    #     return result
    

## 40 timestep sample version
    # def preprocess_json_to_df(
    #     self,
    #     json_file_path: str,
    #     target_timestep: int = -1,
    #     total_timesteps: int = 40,
    #     analysis_mode: str = "multiple"
    # ) -> Tuple[pp.DataFrame, list[str]]:
    #     """
    #     Reads an NDJSON of records with
    #     - rec['local_variables'][var] -> list of length T-1
    #     - rec['ocis'][var]            -> list of length T-1
    #     - rec['target']               -> scalar

    #     Builds a data array of shape (M, T, V+1) where the last channel is 'target'
    #     placed at `target_timestep`, and masks all other missing values.
    #     """
    #     panels = []
    #     var_names = None

    #     for line in open(json_file_path, 'r'):
    #         rec = json.loads(line)

    #         # 1) merge the time-series channels
    #         ts = {**rec['local_variables'], **rec['ocis']}

    #         # 2) determine var_names once (assume same keys for every record)
    #         if var_names is None:
    #             var_names = sorted(ts.keys()) + ['target']
    #         V = len(var_names)
    #         T = total_timesteps

    #         # 3) make an array full of NaNs
    #         arr = np.full((T, V), np.nan, dtype=float)
    #         j_target = V - 1
    #         arr[:, j_target] = 0.0

    #         # 4) fill all measured variables (columns 0..N-2)
    #         expected_length = T - 1
    #         for j, var in enumerate(var_names[:-1]):
    #             series = np.asarray(ts[var], dtype=float)
    #             if series.shape[0] != expected_length:
    #                 raise ValueError(
    #                     f"Variable `{var}` has {series.shape[0]} steps; expected {expected_length}"
    #                 )
    #             arr[:expected_length, j] = series

    #         # 5) fill the target in the last channel at the requested timestep
    #         j_target = V - 1
    #         idx = target_timestep % T
    #         arr[idx, j_target] = float(rec['target'])

    #         panels.append(arr)

    #     # 6) stack into (M, T, N)
    #     data_array = np.stack(panels, axis=0)

    #     # 7) build mask & fill NaNs
    #     mask = np.isnan(data_array)
    #     data_filled = np.nan_to_num(data_array, nan=0.0)

    #     # 7a) convert to dict
    #     data_dict = {i: data_filled[i] for i in range(data_filled.shape[0])}
    #     mask_dict = {i: mask[i]        for i in range(mask.shape[0])}

    #     # 8) create Tigramite DataFrame
    #     self.dataframe = pp.DataFrame(
    #         data         = data_dict,
    #         mask         = mask_dict,
    #         var_names    = var_names,
    #         analysis_mode='multiple'
    #     )

    #     # 9) debug print: show last 5 timesteps (including your filled target)
    #     for i in range(min(3, data_array.shape[0])):
    #         print(f"\n### Observation {i} (last 5 timesteps) =")
    #         df_panel = pd.DataFrame(data_array[i], columns=var_names)
    #         print(df_panel.tail())

    #     return self.dataframe, var_names
    
    
# Full timeseries sample version
    def preprocess_json_to_df(
        self,
        json_file_path: str,
        analysis_mode: str = "multiple"
    ) -> Tuple[pp.DataFrame, List[str]]:
        """
        Read an NDJSON where each record has
          - rec['local_variables'][var] → list of length T
          - rec['ocis'][var]            → list of length T
          - rec['target']               → list of length T (0/1)

        Returns:
          - a Tigramite DataFrame with M panels of shape (T × V)
          - var_names: the list of V variable names, including 'target'
        """
        panels = []
        var_names = None
        T = None

        with open(json_file_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                # merge all series into one dict
                ts = {**rec['local_variables'], **rec['ocis'], 'target': rec['target']}

                # on first line infer T and var order
                if var_names is None:
                    var_names = sorted(ts.keys())
                    T = len(ts['target'])

                # sanity check length
                if len(ts['target']) != T:
                    raise ValueError(f"Inconsistent length {len(ts['target'])} vs {T}")

                # build array (T × V)
                V = len(var_names)
                arr = np.zeros((T, V), dtype=float)
                for j, var in enumerate(var_names):
                    series = np.asarray(ts[var], dtype=float)
                    if series.shape[0] != T:
                        raise ValueError(f"Var `{var}` has len {series.shape[0]}; expected {T}")
                    arr[:, j] = series

                panels.append(arr)

        # stack into (M, T, V)
        data_array = np.stack(panels, axis=0)
        # no missing values in full‐series → all‐False mask
        mask = np.zeros_like(data_array, dtype=bool)

        # convert to dict-of-panels
        data_dict = {m: data_array[m] for m in range(data_array.shape[0])}
        mask_dict = {m: mask[m]        for m in range(mask.shape[0])}

        # create Tigramite DataFrame
        pcmci_df = pp.DataFrame(
            data          = data_dict,
            mask          = mask_dict,
            var_names     = var_names,
            analysis_mode = analysis_mode
        )

        # debug print
        for m in range(min(3, data_array.shape[0])):
            print(f"\n### Panel {m} last 5 rows ###")
            print(pd.DataFrame(data_array[m], columns=var_names).tail())

        return pcmci_df, var_names



    def compute_pcmci_links(
        self,
        dataframe: pp.DataFrame,
        tau_max: int,
        pc_alpha: float,
        independence_test: str = "ParCorr",
        use_gpu: bool = True,          # whether to offload GPDCtorch to GPU
        gpu_device: str = "cuda:0"     # which CUDA device
    ):
        # 1) Choose cond ind test
        if independence_test == "ParCorr":
            # ParCorr can spawn multiple worker processes
            ind_test = ParCorr()
        elif independence_test == "RobustParCorr":
            ind_test = RobustParCorr()
        elif independence_test == "GPDCtorch":
            ind_test = GPDCtorch(verbosity=1)
        else:
            print(f"Unknown test {independence_test}; defaulting to ParCorr")
            ind_test = ParCorr()

        # 2) build and run PCMCI
        print("preping pcmci...")
        pcmci = PCMCI(
            dataframe   = dataframe,
            cond_ind_test = ind_test,
            verbosity=2,
        )

        ### Time consuming
        print("running pcmci...")
        results = pcmci.run_pcmci(
            tau_max = tau_max,
            pc_alpha= pc_alpha,
        )

        # 3) extract what you need
        print("extracting links...")
        link_matrix = (results["p_matrix"] <= pc_alpha).astype(bool)

        return link_matrix, results["p_matrix"], results["val_matrix"]
    

    def __init__(self, json_file_path: str, independence_test: str = "ParCorr", tau_max: int = 23, pc_alpha: float = 0.05, analysis_mode: str = "multiple"):
        """
        Initialize the AdjacencyMatrix class
        Args:
            json_file_path: path to the JSON file containing the data
        """
        self.json_file_path = json_file_path
        self.ind_test = independence_test
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha

        # Preprocess the JSON file and convert to DataFrame
        self.dataframe, self.varlist = self.preprocess_json_to_df(
            json_file_path,
            analysis_mode=analysis_mode
        )

        """# self.dataframe is your tigramite.data_processing.DataFrame
        print("Variables:", self.dataframe.var_names)
        print("Datasets keys:", self.dataframe.datasets)   # usually [0] for a single panel

        for m in self.dataframe.datasets[:5]:
            data = self.dataframe.values[m]
            mask = self.dataframe.mask[m]

            print(f"\n--- Dataset {m} (first 5 timesteps) ---")
            df_data = pd.DataFrame(data, columns=self.dataframe.var_names)
            print(df_data.head())

            print(f"\n--- Mask for dataset {m} (True = originally NaN) ---")
            df_mask = pd.DataFrame(mask, columns=self.dataframe.var_names)
            print(df_mask.head())"""
        self.V = len(self.varlist)
        
        # Calculate PCMCI matrices
        self.link_matrix, self.p_matrix, self.val_matrix = self.compute_pcmci_links(
            dataframe=self.dataframe,
            independence_test=self.ind_test,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
        )


    def reduce_lag_matrix(
        self,
        matrix: np.ndarray,
        method: str = "mean"
    ) -> np.ndarray:
        """
        Collapse a 3-D (V*V*τ) matrix into 2-D (V*V) by mean/max/min.
        If already 2-D, returns a copy unchanged.
        """
        if matrix.ndim == 2:
            return matrix.copy()
        if matrix.ndim != 3:
            raise ValueError(f"Expected 2-D or 3-D array, got {matrix.ndim}-D")
        m = method.lower()
        if m in ("mean", "average"):
            return matrix.mean(axis=2)
        elif m == "max":
            return matrix.max(axis=2)
        elif m == "min":
            return matrix.min(axis=2)
        else:
            raise ValueError("method must be 'mean', 'max' or 'min'")


    def normalize_adj_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Row-normalize so each row sums to 1, then zero the diagonal.
        """
        mat = matrix.astype(float)
        row_sums = np.abs(mat).sum(axis=1, keepdims=True)
        # avoid divide by zero
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums
        np.fill_diagonal(mat, 0.0)
        return mat


    def mask_variable(
        self,
        matrix: np.ndarray,
        target_var: str = "target"
    ) -> tuple[np.ndarray, list[str]]:
        """
        Remove the given variable’s row and column from a square adjacency matrix,
        and drop it from self.varlist.

        Parameters
        ----------
        matrix : np.ndarray
            A (V×V) adjacency matrix.
        target_var : str
            Name of the variable to remove.

        Returns
        -------
        new_matrix : np.ndarray
            A (V-1)×(V-1) adjacency matrix with that variable removed.
        new_varlist : list[str]
            The updated varlist (length V-1).
        """
        if target_var not in self.varlist:
            raise ValueError(f"Variable '{target_var}' not in varlist")
        idx = self.varlist.index(target_var)

        # 1) delete the row
        mat = np.delete(matrix, idx, axis=0)
        # 2) delete the column
        mat = np.delete(mat, idx, axis=1)

        # 3) drop target from varlist
        new_vars = [v for v in self.varlist if v != target_var]

        return mat, new_vars


    def gen_adj_matrix(
        self,
        source: str = "val",
        collapse_method: str = "mean",
        normalize: bool = True,
        mask_target: str | None = "target",
        only_significant: bool = True
    ):
        # 1) pick the 3D tensor
        if source == "val":
            mat3 = self.val_matrix.copy()
            if only_significant:
                mat3 *= self.link_matrix   # zero out non‐significant lags
        elif source == "link":
            mat3 = self.link_matrix.astype(float)
        elif source == "p":
            mat3 = self.p_matrix
        else:
            raise ValueError("source must be 'val','link' or 'p'")

        # 2) collapse to 2D
        adj2d = self.reduce_lag_matrix(mat3, collapse_method)
        output = adj2d

        # 3) normalize if wanted
        if normalize:
            adj2d = self.normalize_adj_matrix(adj2d)
            output = adj2d

        # 4) mask out the target variable
        if mask_target is not None:
            adj2d, new_vars = self.mask_variable(adj2d, mask_target)
            output = (adj2d, new_vars)

        return output
        
    @staticmethod
    def save_matrix(
        matrix: np.ndarray,
        filepath: str
    ) -> None:
        """
        Save a 2-D or 3-D adjacency matrix to disk in a “readable” format:
        - .csv (only for 2D)
        - .npy (only for 2D)
        - .npz (for 2D or 3D)

        Parameters
        ----------
        adj : np.ndarray
            The matrix to save.
        filepath : str
            Path including extension. Supported:
            - .csv  → saved with comma delimiter (only for 2D)
            - .npy  → NumPy .npy binary (only for 2D)
            - .npz  → NumPy .npz compressed archive (matrix stored under key "matrix")
            If no supported extension is given, ".npz" will be appended.
        """
        base, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext == ".csv":
            if matrix.ndim != 2:
                raise ValueError("CSV only supports 2-D matrices")
            np.savetxt(filepath, matrix, delimiter=",")
        elif ext == ".npy":
            if matrix.ndim != 2:
                raise ValueError("`.npy` only supports 2-D; use `.npz` for 3-D")
            np.save(filepath, matrix)
        elif ext == ".npz":
            np.savez_compressed(filepath, matrix=matrix)
        else:
            # default to .npz
            np.savez_compressed(base + ".npz", matrix=matrix)


    @staticmethod
    def load_matrix(
        filepath: str
    ) -> np.ndarray:
        """
        Load a matrix previously saved with `save_matrix`.

        Parameters
        ----------
        filepath : str
            Path to .csv, .npy, or .npz file.

        Returns
        -------
        np.ndarray
            The loaded adjacency matrix (2-D or 3-D).
        """
        base, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext == ".csv":
            mat = np.loadtxt(filepath, delimiter=",")
        elif ext == ".npy":
            mat = np.load(filepath)
        elif ext == ".npz":
            data = np.load(filepath)
            mat = data["matrix"]
        else:
            # try .npz fallback
            data = np.load(base + ".npz")
            mat = data["matrix"]

        return mat


    @staticmethod
    def sample_json_file(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        subset_frac: float = 1.0,
        rng_seed: int = 0
    ) -> None:
        """
        Read an NDJSON file, randomly keep only `subset_frac` of the lines,
        and write them back out to `output_path` in the same NDJSON format.

        Parameters
        ----------
        input_path
            Path to your original NDJSON (one JSON object per line).
        output_path
            Where to write the sampled NDJSON. Overwrites if it exists.
        subset_frac
            Fraction in (0, 1] of lines to keep.
        rng_seed
            Seed for the random number generator to make sampling reproducible.
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        lines = input_path.read_text().splitlines()
        M = len(lines)
        if not (0 < subset_frac <= 1):
            raise ValueError("subset_frac must be in (0, 1].")

        n_keep = int(np.floor(M * subset_frac))
        rng = np.random.default_rng(rng_seed)
        # choose *without* replacement and sort so relative order is preserved
        keep_idx = np.sort(rng.choice(M, size=n_keep, replace=False))

        # write subset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            for i in keep_idx:
                fout.write(lines[i] + "\n")
    
