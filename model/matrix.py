from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import os
import json

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

    def preprocess_json_to_pnc(self, json_file_path: str, local_lag: int = 8, oci_lag: int = 31) -> dict[str, list[tuple[tuple[int, int], float]]]:
        """
        Preprocess JSON file to extract coefficients and time lags
        Args:
            json_file_path: path to the JSON file containing the data
            local_lag: time lag for local variables (default is 8 days)
            oci_lag: time lag for OCI variables (default is 31 days)
            Convert JSON data to a dictionary with the format:
            { 
                variable1: [((sample_index, -time_lag_days), coefficient), ...],
                variable2: [((sample_index, -time_lag_days), coefficient), ...], 
                ...
                target: [((sample_index, 0), value), ...],
            }
        Returns:
            result: dictionary with variable names as keys and lists of tuples as values
                Each tuple contains a sample index, -time lag in days, and coefficient
        """

        result = {}
        
        # Check if file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File not found: {json_file_path}")
        
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            json_data = file.read()
        
        # Process one sample at a time to extract coefficients
        samples = json_data.strip().split('\n')
        
        for sample_idx, sample in enumerate(samples):
            data = json.loads(sample)
            
            # Process local variables (with 39 time lags * 8 days each)
            for var_name, values in data['local_variables'].items():
                if var_name not in result:
                    result[var_name] = []
                
                # For each value in the time series, calculate the time lag in days
                for time_lag_idx, coefficient in enumerate(values):
                    # Time lag in days: each unit is 8 days
                    time_lag_days = -(time_lag_idx + 1) * local_lag
                    result[var_name].append(((sample_idx, time_lag_days), coefficient))
            
            # Process OCI variables (with 10 time lags * 1 month each)
            for var_name, values in data['ocis'].items():
                if var_name not in result:
                    result[var_name] = []
                
                # For each value in the time series, calculate the time lag in days
                for time_lag_idx, coefficient in enumerate(values):
                    # Time lag in days: each unit is 30 days (1 month)
                    time_lag_days = -(time_lag_idx + 1) * oci_lag
                    result[var_name].append(((sample_idx, time_lag_days), coefficient))
            
            # Process target variable (with time lag of 0 days)
            if 'target' in data:
                if 'target' not in result:
                    result['target'] = []
                
                # Add target with time lag of 0 (current value)
                result['target'].append(((sample_idx, 0), data['target']))
        
        # Print the dictionary in a readable format
        for var_name, coefficients in result.items():
            print(f"{var_name}:")
            for i, ((sample_idx, time_lag), coef) in enumerate(coefficients[:5]):  # Show first 5 entries
                print(f"  {i+1}. ((sample_idx={sample_idx}, time_lag={time_lag}), coefficient={coef})")
            print(f"  ... and {len(coefficients)-5} more entries\n")

        return result
    

    def preprocess_json_to_df(
        self,
        json_file_path: str,
        target_timestep: int = -1,
        total_timesteps: int = 40,
        analysis_mode: str = "multiple"
    ) -> Tuple[pp.DataFrame, list[str]]:
        """
        Reads an NDJSON of records with
        - rec['local_variables'][var] -> list of length T-1
        - rec['ocis'][var]            -> list of length T-1
        - rec['target']               -> scalar

        Builds a data array of shape (M, T, V+1) where the last channel is 'target'
        placed at `target_timestep`, and masks all other missing values.
        """
        panels = []
        var_names = None

        for line in open(json_file_path, 'r'):
            rec = json.loads(line)

            # 1) merge the time-series channels
            ts = {**rec['local_variables'], **rec['ocis']}

            # 2) determine var_names once (assume same keys for every record)
            if var_names is None:
                var_names = sorted(ts.keys()) + ['target']
            V = len(var_names)
            T = total_timesteps

            # 3) make an array full of NaNs
            arr = np.full((T, V), np.nan, dtype=float)
            j_target = V - 1
            arr[:, j_target] = 0.0

            # 4) fill all measured variables (columns 0..N-2)
            expected_length = T - 1
            for j, var in enumerate(var_names[:-1]):
                series = np.asarray(ts[var], dtype=float)
                if series.shape[0] != expected_length:
                    raise ValueError(
                        f"Variable `{var}` has {series.shape[0]} steps; expected {expected_length}"
                    )
                arr[:expected_length, j] = series

            # 5) fill the target in the last channel at the requested timestep
            j_target = V - 1
            idx = target_timestep % T
            arr[idx, j_target] = float(rec['target'])

            panels.append(arr)

        # 6) stack into (M, T, N)
        data_array = np.stack(panels, axis=0)

        # 7) build mask & fill NaNs
        mask = np.isnan(data_array)
        data_filled = np.nan_to_num(data_array, nan=0.0)

        # 7a) convert to dict
        data_dict = {i: data_filled[i] for i in range(data_filled.shape[0])}
        mask_dict = {i: mask[i]        for i in range(mask.shape[0])}

        # 8) create Tigramite DataFrame
        self.dataframe = pp.DataFrame(
            data         = data_dict,
            mask         = mask_dict,
            var_names    = var_names,
            analysis_mode='multiple'
        )

        # 9) debug print: show last 5 timesteps (including your filled target)
        for i in range(min(3, data_array.shape[0])):
            print(f"\n### Observation {i} (last 5 timesteps) =")
            df_panel = pd.DataFrame(data_array[i], columns=var_names)
            print(df_panel.tail())

        return self.dataframe, var_names
    
    

    def compute_pcmci_links(
        self,
        dataframe: pp.DataFrame,
        tau_max: int,
        pc_alpha: float,
        independence_test: str = "ParCorr",
        n_jobs: int = 4,               # number of CPU cores to use
        use_gpu: bool = True,          # whether to offload GPDCtorch to GPU
        gpu_device: str = "cuda:0"     # which CUDA device
    ):
        # 1) pick your cond. independence test
        if independence_test == "ParCorr":
            # ParCorr can spawn multiple worker processes
            ind_test = ParCorr(n_jobs=n_jobs)
        elif independence_test == "RobustParCorr":
            ind_test = RobustParCorr(n_jobs=n_jobs)
        elif independence_test == "GPDCtorch":
            # GPDCtorch will run on GPU if you tell it to
            ind_test = GPDCtorch(
                cuda=use_gpu,
                gpu_device=gpu_device,
                # GPDCtorch internally will use the GPU’s parallelism,
                # so we don’t need n_jobs here
            )
        else:
            print(f"Unknown test {independence_test}; defaulting to ParCorr")
            ind_test = ParCorr(n_jobs=n_jobs)

        # 2) build and run PCMCI
        pcmci = PCMCI(
            dataframe   = dataframe,
            cond_ind_test = ind_test,
            verbosity   = 0
        )

        # This is the expensive step; now parallelized
        results = pcmci.run_pcmci(
            tau_max = tau_max,
            pc_alpha= pc_alpha
        )

        # 3) extract what you need
        link_matrix = (results["p_matrix"] <= pc_alpha).astype(int)
        val_matrix  = results["val_matrix"]
        V = link_matrix.shape[0]

        adj_matrix = np.zeros((V, V), dtype=float)
        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                sig = np.where(link_matrix[i, j, :] == 1)[0]
                if sig.size:
                    coeffs   = val_matrix[i, j, sig]
                    best_lag = sig[np.argmax(np.abs(coeffs))]
                    adj_matrix[i, j] = val_matrix[i, j, best_lag]

        return link_matrix, results["p_matrix"], val_matrix, adj_matrix
    

    def __init__(self, json_file_path: str, target_timestep: int = -1, total_timesteps: int = 40, independence_test: str = "ParCorr", tau_max: int = 19, pc_alpha: float = 0.05):
        """
        Initialize the AdjacencyMatrix class
        Args:
            json_file_path: path to the JSON file containing the data
        """
        self.json_file_path = json_file_path
        self.total_timesteps = total_timesteps
        self.target_timestep = target_timestep
        self.ind_test = independence_test
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha

        # Preprocess the JSON file and convert to DataFrame
        self.dataframe, self.varlist = self.preprocess_json_to_df(
            json_file_path, 
            target_timestep=target_timestep, 
            total_timesteps=total_timesteps
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
        self.p_matrix, self.val_matrix, self.adj_matrix = self.compute_pcmci_links(
            dataframe=self.dataframe,
            independence_test=self.ind_test,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha
        )
    


    def normalize_adj_matrix(self, matrix):
        """
        Normalize the adjacency matrix to sum to 1
        """
        # Normalize the weight matrix
        norm_adj_matrix = matrix / np.sum(np.abs(matrix), axis=1, keepdims=True)
        
        # Set diagonal to 0
        np.fill_diagonal(norm_adj_matrix, 0)
        
        return norm_adj_matrix
    

    def mask_target (self, matrix, target_var: str = "target"):
        """
        Mask the target variable in the adjacency matrix
        Args:
            target: name of the target variable
        """
        # Get index of target variable
        target_index = self.varlist.index(target_var)
        
        # Set all weights to 0 for the target variable
        masked_adj_matrix = matrix

        masked_adj_matrix[target_index, :] = 0.0
        masked_adj_matrix[:, target_index] = 0.0
        
        return masked_adj_matrix