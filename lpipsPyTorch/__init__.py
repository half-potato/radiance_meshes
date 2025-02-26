import torch

from .modules.lpips import LPIPS

class LPIPSEval:
    r"""Class for measuring lpips between two images

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    criterion = None
    
    def __init__(self, device, net_type: str = 'alex', version: str = '0.1'):
        self.criterion = LPIPS(net_type, version).to(device)
        
    r"""Compute lpips for two images

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
    """
    def eval(self, x: torch.Tensor, y: torch.Tensor):
        return self.criterion(x, y)