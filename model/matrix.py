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
            input_data: pandas array of shape (T, V), where T = timesteps, V = variables
            var_list: list of variable names
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
            adj_matrix          : weighted adjacency matrix of shape (V, V)            
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
        
        self.link_matrix, self.val_matrix, self.adj_matrix = link_matrix, val_matrix, adj_matrix

        return link_matrix, val_matrix, adj_matrix
    

    def normalize_adj_matrix(self):
        """
        Normalize the adjacency matrix to sum to 1
        """
        # Normalize the weight matrix
        norm_adj_matrix = self.adj_matrix / np.sum(np.abs(self.adj_matrix), axis=1, keepdims=True)
        
        # Set diagonal to 0
        np.fill_diagonal(norm_adj_matrix, 0)
        
        return norm_adj_matrix
    

    def mask_target (self, target_var: str):
        """
        Mask the target variable in the adjacency matrix
        Args:
            target: name of the target variable
        """
        # Get index of target variable
        target_index = self.varlist.index(target_var)
        
        # Set all weights to 0 for the target variable
        self.adj_matrix[target_index, :] = 0
        self.adj_matrix[:, target_index] = 0
        
        return self.adj_matrix



        
        
