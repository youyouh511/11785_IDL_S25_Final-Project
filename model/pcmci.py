###################################################################
# This script sets up the PCMCI algorithm for causal inference
# PCMCI (Parent-Child Mixed Conditional Independence) is a method
# using the Tigramite library (https://github.com/jakobrunge/tigramite)
###################################################################

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
        self.links = torch.Tensor(np.zeros((self.V, self.V), dtype=float))


    def compute_pcmci_links(self, independence_test: str = "ParCorr", tau_max: int = 8, pc_alpha: float = 0.05):
        """
        Run PCMCI to infer causal adjacency among time series variables.
        Args:
            independence_test   : type of independence test to use (e.g., "ParCorr", used in baseline)
            tau_max             : max lag in timesteps for causal inference
            pc_alpha            : significance threshold for link selection
        Returns:
            adj_matrix: binary numpy array (V, V) where adj[i, j] = 1 if X_i -> X_j
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
        self.links = results['link_matrix']  # shape (V, V, 2*tau_max+1)
        
