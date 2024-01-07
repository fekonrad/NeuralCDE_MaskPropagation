import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.nn import MSELoss
from Models.neuralCDE import NeuralCDE
from scipy.interpolate import CubicSpline
from Splines.splines import CubicSplineInt
from torchdiffeq import odeint


def test_ncde(epochs=1):
    class TestCDEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.id = nn.Sequential(nn.Linear(1, 1, bias=True))

        def forward(self, x):
            # return torch.ones([1])
            return self.id(x)

    times = torch.tensor(np.linspace(0, 1, 10), requires_grad=True)
    data = torch.sin(2 * np.pi * times).type(torch.float32)

    lr = 0.1
    cdefunc = TestCDEFunc()
    model = NeuralCDE(cdefunc, times, data)
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    loss_fn = MSELoss()

    # get initial prediction for comparison
    with torch.no_grad():
        predictions_t0 = odeint(model, torch.tensor([0.0]), torch.tensor(np.linspace(0, 1, 100)))

    for _ in range(epochs):
        optimizer.zero_grad()
        pred_y = odeint(model, torch.tensor([0.0], requires_grad=True), times)
        loss = loss_fn(pred_y[:, 0], data)
        loss.backward(retain_graph=True)
        optimizer.step()

    # plot data and predictions
    final_params = [param.data for param in model.parameters()]
    with torch.no_grad():
        predictions = odeint(model, torch.tensor([0.0]), torch.tensor(np.linspace(0, 1, 100)))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.linspace(0, 1, 10), np.sin(2 * np.pi * np.linspace(0, 1, 10)), 'o', label="Data")
    ax.plot(np.linspace(0, 1, 100),
            CubicSpline(np.linspace(0, 1, 10), np.sin(2 * np.pi * np.linspace(0, 1, 10)))(np.linspace(0, 1, 100)),
            label="Spline")
    ax.plot(np.linspace(0, 1, 100), predictions.detach().numpy(), label="Trained NeuralCDE")
    ax.plot(np.linspace(0, 1, 100), predictions_t0.detach().numpy(), label="Initial NeuralCDE")

    ax.legend(loc='lower left', ncol=2)
    ax.set_title(
        f"Final parameters: Weight {final_params[0].detach().numpy()}, Bias {final_params[1].detach().numpy()} \n Should approximately be 0 and 1 ")
    plt.show()


def test_cde_mult(epochs=1):
    class TestCDEFuncMult(nn.Module):
        def __init__(self, dim=(2, 2)):
            """
            dim: tuple (dim_out, dim_in), where:
                - 'dim_in' is the dimension of the input data (spline),
                - 'dim_out' is the dimension of the output data
            """
            super().__init__()
            self.dim_in = dim[1]
            self.dim_out = dim[0]
            self.net = nn.Sequential(
                nn.Linear(self.dim_out, self.dim_in * self.dim_out, bias=True))

        def forward(self, x):
            out = self.net(x)
            return out.view(-1, self.dim_in)

    times = torch.tensor(np.linspace(0, 1, 10), requires_grad=True)
    data = torch.rand((10, 2))

    dim_in = 2
    dim_out = 2

    lr = 0.1
    cdefunc = TestCDEFuncMult(dim=(dim_out, dim_in))
    model = NeuralCDE(cdefunc, times, data)
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    loss_fn = MSELoss()

    # get initial prediction for comparison
    with torch.no_grad():
        predictions_t0 = odeint(model, data[0], torch.tensor(np.linspace(0, 1, 100)))

    for _ in range(epochs):
        optimizer.zero_grad()
        pred_y = odeint(model, data[0], times)
        loss = loss_fn(pred_y, data)
        loss.backward(retain_graph=True)
        optimizer.step()

    # plot data and predictions
    final_params = [param.data for param in model.parameters()]
    with torch.no_grad():
        predictions = odeint(model, data[0], torch.tensor(np.linspace(0, 1, 100)))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.linspace(0, 1, 10), data[:, 0], 'o', label="Data")
    ax.plot(np.linspace(0, 1, 100), CubicSpline(np.linspace(0, 1, 10), data)(np.linspace(0, 1, 100))[:, 0],
            label="Spline")
    ax.plot(np.linspace(0, 1, 100), predictions.detach().numpy()[:, 0], label="Trained NeuralCDE")
    ax.plot(np.linspace(0, 1, 100), predictions_t0.detach().numpy()[:, 0], label="Initial NeuralCDE")

    ax.legend(loc='lower left', ncol=2)
    ax.set_title(f"First Component")
    plt.show()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.linspace(0, 1, 10), data[:, 1], 'o', label="Data")
    ax.plot(np.linspace(0, 1, 100), CubicSpline(np.linspace(0, 1, 10), data)(np.linspace(0, 1, 100))[:, 1],
            label="Spline")
    ax.plot(np.linspace(0, 1, 100), predictions.detach().numpy()[:, 1], label="Trained NeuralCDE")
    ax.plot(np.linspace(0, 1, 100), predictions_t0.detach().numpy()[:, 1], label="Initial NeuralCDE")

    ax.legend(loc='lower left', ncol=2)
    ax.set_title(f"Second Component")
    plt.show()
