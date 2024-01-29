import torch
from scipy.interpolate import CubicSpline

class CubicSplineInt():
    def __init__(self, times, data):
        """
        data: tensor representing data to be interpolated,
              tensor.shape[0] is the batch_size,
              tensor.shape[1] is the number of frames,
              tensor.shape[1:] is the dimension of the data points
        times: tensor of shape (batch_size, frames) specifying the time points of the data points
        """
        times_np = times.detach().numpy()
        data_np = data.detach().numpy()
        self.spline = CubicSpline(times_np, data_np)
        # self.spline = self.__init_spline__(times, data)

    def __init_spline__(self, times, data):
        splines = []
        for i in range(data.shape[0]):
            x = data[i]
            t = times[i]
            splines.append(CubicSpline(t, x))
        return splines

    def eval(self, t):
      t_np = t.detach().numpy()
      return torch.tensor(self.spline(t_np), requires_grad=True).view(-1)

    def deriv(self, t):
      t_np = t.detach().numpy()
      return torch.tensor(self.spline(t_np, 1), requires_grad=True, dtype=torch.float32).view(-1)
