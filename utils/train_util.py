import torch

class ScaleGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lr_matrix):
        ctx.save_for_backward(lr_matrix)
        return input  # Identity operation

    @staticmethod
    def backward(ctx, grad_output):
        lr_matrix, = ctx.saved_tensors
        return grad_output * lr_matrix, None