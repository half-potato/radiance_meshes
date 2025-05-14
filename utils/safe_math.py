import torch
import math
import numpy as np

# -------------------------------------------------------------------
# Constants (float32 equivalents to jax's np.finfo(np.float32))
# -------------------------------------------------------------------
tiny_val = torch.finfo(torch.float32).tiny     # ~1.1755e-38
min_val = torch.finfo(torch.float32).min       # -3.4028e+38
max_val = torch.finfo(torch.float32).max       #  3.4028e+38

# -------------------------------------------------------------------
# Helper: remove_zero
# Shifts `x` away from 0 by tiny_val if abs(x) < tiny_val
# -------------------------------------------------------------------
def remove_zero(x, tiny_val:float = torch.finfo(torch.float32).tiny):
    return torch.where(torch.abs(x) < tiny_val, 
                       torch.tensor(tiny_val, device=x.device, dtype=x.dtype), 
                       x)

# -------------------------------------------------------------------
# plus_eps: Add machine epsilon in a custom way, skipping derivative
# -------------------------------------------------------------------
class _PlusEps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # We'll ignore the gradient w.r.t. x, but we still return x with a tiny "step"
        inf = torch.tensor(float('inf'), dtype=x.dtype, device=x.device)
        # if |x| < tiny_val => tiny_val else nextafter(x, +inf)
        y = torch.where(torch.abs(x) < tiny_val,
                        torch.tensor(tiny_val, dtype=x.dtype, device=x.device),
                        torch.nextafter(x, inf))
        # We do not need to save anything for backward since we skip gradient
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # "Make plus_eps()'s gradient a no-op" => derivative is 1 => pass grad_output
        return grad_output

def plus_eps(x):
    return _PlusEps.apply(x)

# -------------------------------------------------------------------
# minus_eps: Subtract machine epsilon in a custom way, skipping derivative
# -------------------------------------------------------------------
class _MinusEps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ninf = torch.tensor(float('-inf'), dtype=x.dtype, device=x.device)
        # if |x| < tiny_val => -tiny_val else nextafter(x, -inf)
        y = torch.where(torch.abs(x) < tiny_val,
                        torch.tensor(-tiny_val, dtype=x.dtype, device=x.device),
                        torch.nextafter(x, ninf))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # "Make minus_eps()'s gradient a no-op"
        return grad_output

def minus_eps(x):
    return _MinusEps.apply(x)

# -------------------------------------------------------------------
# clip_nograd: clamp to [a_min, a_max] but pass gradient straight through
# -------------------------------------------------------------------
class _ClipNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_min, a_max):
        # We do not save anything for backward
        return torch.clamp(x, a_min, a_max)

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient unmodified (derivative = 1 everywhere)
        return grad_output, None, None

def generate_clip_nograd_fn(a_min, a_max):
    def clip_nograd(x):
        return _ClipNoGrad.apply(x, a_min, a_max)
    return clip_nograd

clip_finite_nograd = generate_clip_nograd_fn(min_val, max_val)
clip_pos_finite_nograd = generate_clip_nograd_fn(tiny_val, max_val)

# -------------------------------------------------------------------
# clip_pos: clamp x to be at least tiny_val
# -------------------------------------------------------------------
def clip_pos(x):
    return torch.clamp(x, min=tiny_val)

# -------------------------------------------------------------------
# safe_sign: sign(x), but assume 0 -> +1
# -------------------------------------------------------------------
def safe_sign(x):
    return torch.where(x < 0, torch.tensor(-1.0, device=x.device, dtype=x.dtype),
                           torch.tensor(1.0,  device=x.device, dtype=x.dtype))

# -------------------------------------------------------------------
# safe_trig_helper: reduce large inputs by mod t before sin/cos
# -------------------------------------------------------------------
def safe_trig_helper(x, fn, t=100 * math.pi):
    x_mod = torch.where(torch.abs(x) < t, x, x % t)
    return fn(x_mod)

def safe_sin(x, t:float =100 * math.pi):
    x_mod = torch.where(torch.abs(x) < t, x, x % t)
    return x_mod.sin()

def safe_cos(x, t:float =100 * math.pi):
    x_mod = torch.where(torch.abs(x) < t, x, x % t)
    return x_mod.cos()

# -------------------------------------------------------------------
# safe_arctan2 with a custom backward
#    y = arctan2(x1, x2)
#    dy/dx1 = x2 / (x1^2 + x2^2)
#    dy/dx2 = -x1 / (x1^2 + x2^2)
# with zero-protection in denominator
# -------------------------------------------------------------------
class _SafeArctan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        y = torch.atan2(x1, x2)
        ctx.save_for_backward(x1, x2)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        denom = remove_zero(x1**2 + x2**2)
        # partial derivatives
        dx1 = grad_output * (x2 / denom)
        dx2 = grad_output * (-x1 / denom)
        return dx1, dx2

def safe_arctan2(x1, x2):
    return _SafeArctan2.apply(x1, x2)

# -------------------------------------------------------------------
# safe_div with custom backward:
# forward: r = clamp( n / remove_zero(d), min_val, max_val )
# backward w.r.t n: clamp( grad / remove_zero(d), min_val, max_val )
# backward w.r.t d: clamp( -grad * r / remove_zero(d), min_val, max_val )
# and zero out forward if |d| < tiny_val
# -------------------------------------------------------------------
class _SafeDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n, d) -> torch.Tensor:
        d_nozero = remove_zero(d)
        r = torch.clamp(n / d_nozero, min_val, max_val)
        # if |d| < tiny_val => 0
        out = torch.where(torch.abs(d) < tiny_val, 
                          torch.tensor(0.0, dtype=d.dtype, device=d.device),
                          r)
        ctx.save_for_backward(n, d, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        n, d, out = ctx.saved_tensors
        d_nozero = remove_zero(d)

        # dn = clamp(grad / remove_zero(d), min_val, max_val)
        dn = torch.clamp(grad_output / d_nozero, min_val, max_val)

        # dd = clamp(-grad * out / remove_zero(d), min_val, max_val)
        dd = torch.clamp(-grad_output * out / d_nozero, min_val, max_val)

        return dn, dd

def safe_div(n, d) -> torch.Tensor:
    return _SafeDiv.apply(n, d)

# -------------------------------------------------------------------
# Generate "safe" versions of basic functions: log, exp, sqrt, etc.
# in JAX we used a custom_jvp that clips input in both fwd and bwd.
# In PyTorch, define a custom autograd Function that ensures the
# forward pass is clipped and the backward pass uses clipped inputs.
# -------------------------------------------------------------------
def generate_safe_fn(forward_fn, backward_fn, x_min, x_max):
    """
    forward_fn(x): the normal forward operation (e.g. torch.log)
    backward_fn(x, y, x_dot): how to compute derivative wrt x
      - x is the clipped input
      - y is forward_fn(x)
      - x_dot is the incoming gradient
    x_min, x_max: clamp range for x
    """
    class _SafeFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_clip = torch.clamp(x, x_min, x_max)
            y = forward_fn(x_clip)
            # Save x_clip and y for backward
            ctx.save_for_backward(x_clip, y)
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x_clip, y = ctx.saved_tensors
            # derivative wrt x = backward_fn(x_clip, y, grad_output)
            return backward_fn(x_clip, y, grad_output)

    def safe_fn(x):
        return _SafeFn.apply(x)
    return safe_fn

# -------------------------------------------------------------------
# safe_log(x) = log(x), with input clipped to [tiny_val, max_val]
# grad:  d/dx (log(x)) = 1/x
# -------------------------------------------------------------------
def _log_backward_fn(x_clip, y, grad_output):
    # x_clip is guaranteed >= tiny_val
    return grad_output / x_clip

safe_log = generate_safe_fn(
    forward_fn=torch.log,
    backward_fn=_log_backward_fn,
    x_min=tiny_val, 
    x_max=max_val
)

# -------------------------------------------------------------------
# safe_exp(x), with input clipped so we don't blow up
# grad: d/dx (exp(x)) = exp(x)
# -------------------------------------------------------------------
# We clamp x to [min_val, log(max_val)] 
log_max_val_approx = torch.nextafter(torch.log(torch.tensor(max_val, dtype=torch.float32)), torch.tensor(0.0))

def _exp_backward_fn(x_clip, y, grad_output):
    # derivative is exp(x_clip) * grad (which is y)
    return y * grad_output

safe_exp = generate_safe_fn(
    forward_fn=torch.exp,
    backward_fn=_exp_backward_fn,
    x_min=min_val,
    x_max=log_max_val_approx
)

# -------------------------------------------------------------------
# safe_sqrt(x), with input clipped to [0, max_val]
# grad: d/dx sqrt(x) = 0.5 / sqrt(x)
# but we ensure x >= tiny_val inside to avoid divide by zero
# -------------------------------------------------------------------
def _sqrt_backward_fn(x_clip, y, grad_output):
    # x_clip >= 0
    # derivative = 0.5 * grad / sqrt(x_clip)
    # if x_clip is extremely small, we ensure it's at least tiny_val
    x_nozero = torch.where(x_clip < tiny_val,
                           torch.tensor(tiny_val, device=x_clip.device, dtype=x_clip.dtype),
                           x_clip)
    return 0.5 * grad_output / torch.sqrt(x_nozero)

safe_sqrt = generate_safe_fn(
    forward_fn=torch.sqrt,
    backward_fn=_sqrt_backward_fn,
    x_min=tiny_val,
    x_max=max_val
)

# -------------------------------------------------------------------
# inverse_sigmoid(x) = log(x / (1 - x))
# No custom gradient override: PyTorch handles it normally
# -------------------------------------------------------------------
def inverse_sigmoid(x):
    return torch.log(x / (1.0 - x))

def generate_safe_fn(forward_fn, backward_fn, x_min, x_max):
    """
    A helper generator that returns a custom autograd.Function
    with clamped input in forward pass, and uses a custom grad function in backward.
    """
    class _SafeFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_clamped = torch.clamp(x, x_min, x_max)
            y = forward_fn(x_clamped)
            ctx.save_for_backward(x_clamped, y)
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x_clamped, y = ctx.saved_tensors
            # Use user-supplied backward function: backward_fn(x_clamped, y, grad_output)
            return backward_fn(x_clamped, y, grad_output)

    def safe_fn(x):
        return _SafeFn.apply(x)

    return safe_fn

# Define the forward function: y = x^exponent
def forward_fn(x_clamped, exponent):
    return x_clamped.pow(exponent)

# Define the backward function: dy/dx = exponent * x^(exponent - 1)
def backward_fn(x_clamped, y, grad_output, exponent):
    return grad_output * exponent * x_clamped.pow(exponent - 1)

class SafePowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_min, x_max, exponent):
        x_clamped = torch.clamp(x, x_min, x_max)
        y = forward_fn(x_clamped, exponent)
        ctx.save_for_backward(x_clamped, y)
        ctx.exponent = exponent
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_clamped, y = ctx.saved_tensors
        exponent = ctx.exponent
        grad_input = backward_fn(x_clamped, y, grad_output, exponent)
        return grad_input, None, None, None

def safe_pow(x, exponent:float=0.5, x_min:float=1e-6, x_max:float=1e6):
    """
    Safely compute x^exponent with:
      - Forward pass clamping x in [x_min, x_max]
      - Backward pass using derivative of x^exponent on the clamped input.

    Useful for exponents < 1 to avoid NaNs or infinities near x=0.
    """
    return SafePowFunction.apply(x, x_min, x_max, exponent)
