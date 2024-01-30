# Neural CDEs for Mask Propagation - A "continuous-time U-Net" 

The goal of this repository is to implement the model presented in ["Exploiting Inductive Biases in Video Modeling through Neural CDEs"](https://arxiv.org/pdf/2311.04986.pdf) by J. Chiu et.al. 
Note that there already is a repository for this paper [here](https://github.com/normal-computing/ct-video-modeling/tree/main). 

![](/Figures/MaskPropCT.png)

# How to use 
The class `CT_UNet` implements the "continuous-time U-Net" as described in Section 3 of the paper. It creates an `nn.Module` whose initialization takes the following (optional) arguments: 
- `hidden_channels` : Specifies the amount of hidden channels in the first convolutional block, doubling in each convolutional block
- `atol` : Absolute tolerance for numerical integrator `odeint` 
- `rtol` : Relative tolerance for numerical integrator `odeint`
- `method` : String specifying the method of numerical integration used in `odeint`. See the [implementation](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py) for available methods

It's `forward()`-method takes the following arguments: 
- `x` : Tensor of shape (batch_size, frames, channels, height, width)
- `times` : 1-D Tensor with increasing entries representing the time points for which the mask is computed
- `y0` : Tensor of shape (height, width) representing the initial mask at time `t0`

**Returns:** 
- Tensor of shape (batch_size, frames, height, width) containing the masks for each frame. 

# Demo 
*...to be added...*

