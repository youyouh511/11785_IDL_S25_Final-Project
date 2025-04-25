from typing import Optional
import numpy as np
import pandas as pd
import torch

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
import tigramite.independence_tests as it
from tigramite.toymodels import structural_causal_model as toys



class AdjacencyMatrix:
    """
    Adjacency matrix representation of causal graph, based on time series data
    Weight qualifies the strength of the causal link
    """
    def __init__(self, input_data: pd.DataFrame, varlist: Optional[list[str]] = None):
        """
        Args:
            input_data: numpy array of shape (T, V), where T = timesteps, V = variables
            matrix_dim: number of variables, i.e., columns in input_data, equals to V
            var_list: list of variable names
            adj_matrix: adjacency matrix of shape (V, V)
        """
        self.data = input_data.to_numpy()
        self.varlist = list(input_data.columns)
        self.V = len(self.varlist)


    def compute_pcmci_links(self, independence_test: str = "ParCorr", tau_max: int = 23, pc_alpha: float = 0.05):
        """
        Run PCMCI to infer causal adjacency among time series variables.
        Args:
            independence_test   : type of independence test to use (e.g., "ParCorr", used in baseline)
            tau_max             : max lag in timesteps for causal inference (e.g., 23 is chosen for 6 months of 8-day temporal resampling)
            pc_alpha            : significance threshold for link selection
        Returns:
            link_matrix         : binary matrix of shape (V, V, tau_max) indicating causal links (above pc_alpha)
            val_matrix          : matrix of coefficients of shape (V, V, tau_max)
            weight_matrix       : weighted adjacency matrix of shape (V, V)            
        """
        
        dataframe = pp.DataFrame(self.input_data, self.var_list)

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
        weight_matrix = np.zeros((self.V, self.V), dtype=float)
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
                    weight_matrix[i, j] = val_matrix[i, j, max_index]
        
        self.link_matrix, self.val_matrix, self.weight_matrix = link_matrix, val_matrix, weight_matrix

        return link_matrix, val_matrix, weight_matrix



        
        
