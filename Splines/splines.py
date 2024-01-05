from scipy.interpolate import CubicSpline


class CubicSplineInt:
    def __init__(self, times, data):
        """
        data: array representing data to be interpolated
        times: numpy array of strictly increasing numbers specifying the time points of the data points
        """
        self.spline = CubicSpline(times, data)

    def eval(self, t):
        return self.spline(t)

    def deriv(self, t):
        return self.spline(t, 1)
