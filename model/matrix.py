from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import os
import json

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
import tigramite.independence_tests as it
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
        json_file_path: str,
        target_timestep: int = -1,
        total_timesteps: int = 40,
        analysis_mode: str = "multiple"
    ) -> Tuple[np.ndarray, List[str]]:
            """
            Reads JSON file where each line is represented as:
            {'local_variables'  : {var_name: [T-1 value], ...}
             'ocis'             : {var_name: [T-1 value], ...}
             'target'           : binary
            }

            Convert to data with T = total_timesteps rows [obs, T, V]
            
            Returns:
                df: DataFrame
                    Tigramite DataFrame
                var_names: list[str]
                    Name of the V columns, as in the same order as in the array
            """

            panels = []
            for line in open(json_file_path, 'r'):
                rec = json.loads(line)
                # Merge all measured series into one dict
                ts = {**rec['local_variables'], **rec['ocis']}
                # Fix a stable ordering, then append the 'target' column last
                var_names = sorted(ts.keys()) + ['target']
                N = len(var_names)

                # Prepare an array of shape (T, N), filled with NaN
                arr = np.full((total_timesteps, N), np.nan, dtype=float)

                # Fill in each column
                for j, var in enumerate(var_names):
                    if var == 'target':
                        # Allow negative index so -1 => last row, -2 => second‐to‐last, etc.
                        idx = target_timestep % total_timesteps
                        arr[idx, j] = rec['target']
                    else:
                        series = np.asarray(ts[var], dtype=float)
                        expected = total_timesteps - 1
                        if series.shape[0] != expected:
                            raise ValueError(
                                f"Variable `{var}` has {series.shape[0]} steps; "
                                f"expected {expected}"
                            )
                        # Fill rows 0..T-2 with the ascending‐time series
                        arr[:expected, j] = series

                panels.append(arr)

            # Stack into shape (M, T, N)
            data_array = np.stack(panels, axis=0)

            # Masking NaN
            mask = np.isnan(data_array)
            data_filled = np.nan_to_num(data_array, nan=0.0)

            # Create Tigramite DataFrame
            pcmci_df = pp.DataFrame(
                data=data_filled,
                mask=mask,
                var_names=var_names,
                analysis_mode=analysis_mode
            )

            # Print example for verification
            num_examples = min(3, data_array.shape[0])
            for i in range(num_examples):
                print(f"\n### Masked observation {i} (last 5 timesteps) =")
                df_panel = pd.DataFrame(data_array[i], columns=var_names)
                print(df_panel.tail())

            return pcmci_df, var_names
    
    

    def compute_pcmci_links(self, dataframe: pp.DataFrame, independence_test: str = "ParCorr", tau_max: int = 23, pc_alpha: float = 0.05):
        """
        Run PCMCI to infer causal adjacency among time series variables.
        Args:
            independence_test   : type of independence test to use (e.g., "ParCorr", used in baseline)
            tau_max             : max lag in timesteps for causal inference (e.g., 23 is chosen for 6 months of 8-day temporal resampling)
            pc_alpha            : significance threshold for link selection
        Returns:
            link_matrix         : binary matrix of shape (V, V, tau_max) indicating causal links (above pc_alpha)
            val_matrix          : matrix of coefficients of shape (V, V, tau_max)
            adj_matrix          : weighted adjacency matrix of shape (V, V)            
        """
        
        # Determine independence test
        if independence_test == "ParCorr":
            ind_test = it.ParCorr()
        elif independence_test == "GausCI":
            ind_test = it.GausCI()
        else:
            print(f"Unknown independence test: {independence_test}", "defaulted to ParCorr")
            ind_test = it.ParCorr()
    
        # Run PCMCI with data and independence test to derive links
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ind_test, verbosity=0)
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

        link_matrix = results['link_matrix']   # binary
        val_matrix = results['val_matrix']     # coefficients
        
        # Build weighted matrices (binary and weighted)
        adj_matrix = np.zeros((self.V, self.V), dtype=float)
        for i in range(self.V):
            for j in range(self.V):
                if i == j:
                    continue

                # Find lags with significant links
                sig_lags = np.where(link_matrix[i, j, :] != 0)[0]
                if sig_lags.size > 0:
                    # Select lag with maximum magnitutde coefficient (positive or negative)
                    coeffs = val_matrix[i, j, sig_lags]
                    max_index = sig_lags[np.argmax(np.abs(coeffs))]
                    adj_matrix[i, j] = val_matrix[i, j, max_index]
        
        return link_matrix, val_matrix, adj_matrix
    

    def __init__(self, json_file_path: str, target_timestep: int = -1, total_timesteps: int = 40, independence_test: str = "ParCorr", tau_max: int = 23, pc_alpha: float = 0.05):
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
        self.V = len(self.varlist)
        
        # Calculate PCMCI matrices
        self.link_matrix, self.val_matrix, self.adj_matrix = self.compute_pcmci_links(
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