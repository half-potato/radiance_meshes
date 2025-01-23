import torch
from torch import nn
from torch.optim import Adam

class CustomAdam:
    def __init__(self, params, lr=1e-3, ignore_param_list=None, **kwargs):
        self.optimizer = Adam(params, lr=lr, **kwargs)
        self.ignore_param_list = ignore_param_list if ignore_param_list is not None else []

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        Replace a tensor in the optimizer with a new tensor, maintaining its state if possible.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in self.ignore_param_list:
                continue
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state and "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_optimizer(self, mask):
        """
        Prune tensors in the optimizer based on a mask, adjusting optimizer state accordingly.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in self.ignore_param_list:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Concatenate new tensors to the existing tensors in the optimizer, updating optimizer state.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in self.ignore_param_list:
                continue
            assert len(group['params']) == 1, "Each parameter group must contain exactly one parameter."

            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors

    def step(self):
        """
        Perform a single optimization step.
        """
        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out the gradients for all optimized parameters.
        """
        self.optimizer.zero_grad()
