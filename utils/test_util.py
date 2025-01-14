import numpy as np
import torch
torch.set_printoptions(precision=10)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def compare_dict_values(results1, results2, keys_to_compare, vertices=None, viewmat=None, tile_size=None):
    """
    Compare values from two dictionaries, checking for error magnitude and convergence.
    
    Args:
        results1 (dict): First results dictionary (with more samples)
        results2 (dict): Second results dictionary (with fewer samples)
        keys_to_compare (list): List of tuples containing:
            - key1: Key for first value to compare
            - key2: Key for second value to compare
            - description: String description of what's being compared
        vertices (torch.Tensor, optional): Vertex values for error reporting
        viewmat (torch.Tensor, optional): View matrix for error reporting
    """
    for key1, key2, description, abs_thresh, rel_thresh in keys_to_compare:
        error1 = np.abs(results1[key1] - results1[key2]).mean()
        error2 = np.abs(results2[key1] - results2[key2]).mean()
        
        # Check if error is both large and non-decreasing
        if error1 > abs_thresh and error2 >= error1:
            error_message = (
                f"\n{bcolors.FAIL}{description}{bcolors.ENDC} error is large and non-decreasing:"
                f"\nError with n_samples=10000: {error1:.6f}"
                f"\nError with n_samples=5000: {error2:.6f}"
                f"\nRelative error increase: {(error2/error1 - 1)*100:.2f}%"
            )
            if vertices is not None:
                error_message += f"\nvertices = torch.{vertices}"
            if viewmat is not None:
                error_message += f"\nviewmat = torch.{viewmat}"
            error_message += f"\n{bcolors.FAIL}END MESSAGE{bcolors.ENDC}"
            raise AssertionError(error_message)
            
        try:
            np.testing.assert_allclose(results1[key1], results1[key2], atol=abs_thresh, rtol=rel_thresh)
        except AssertionError as e:
            error_message = f"\n{bcolors.FAIL}{description}{bcolors.ENDC} error: {error1:.6f}"
            error_message += f"\n{e}"
            if vertices is not None:
                error_message += f"\nvertices = torch.{vertices}"
            if viewmat is not None:
                error_message += f"\nviewmat = torch.{viewmat}"
            if tile_size is not None:
                error_message += f"\ntile_size = {tile_size}"
            error_message += f"\n{bcolors.FAIL}END MESSAGE{bcolors.ENDC}"
            raise AssertionError(error_message) from e