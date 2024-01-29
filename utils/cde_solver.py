from torch import nn
from torchdiffeq import odeint


class VectorFieldConv(nn.Module):
  def __init__(self, spline, func):
    super().__init__()
    self.spline = spline
    self.func = func

  def forward(self, t, x):
    # needs 'self.func' and 'self.spline' to return tensors of same shape (batch_size, channels, height, width)
    # note the '*' instead of '@': Interpret this as 'self.func' returning a diagonal matrix
    return self.func(x) * self.spline.deriv(t)

def cde_solve(spline, func, z0, times):
  """
    spline: Interpolating spline representing the control path, as in CubicSplineInt
    func: nn.Module function representing the ODE term
    z0: tensor representing the initial state for CDE
    times: 1D tensor of time points at which the solution is computed
  """
  vector_field = VectorFieldConv(spline, func)
  return odeint(vector_field, z0, times)          # uses torchdiffeq's 'odeint' method   (https://github.com/rtqichen/torchdiffeq)