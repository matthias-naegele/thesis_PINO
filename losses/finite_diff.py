import torch


def fd_derivatives_periodic(x: torch.Tensor, ell: list):
    """
    Finite-difference first-order spatial derivatives on a periodic domain.
    Returns wx of shape (batch, 2*C, nx, ny).
    First C channels = d/dx, next C channels = d/dy.
    Assumes periodic domain with lengths ell = [Lx, Ly].
    Input x: (batch, C, nx, ny)
    """
    Lx, Ly = ell
    nx, ny = x.shape[2], x.shape[3]
    dx = Lx / nx
    dy = Ly / ny

    xp1_x = torch.roll(x, -1, dims=2)   # x[i+1, j]
    xm1_x = torch.roll(x,  1, dims=2)   # x[i-1, j]
    xp1_y = torch.roll(x, -1, dims=3)   # x[i, j+1]
    xm1_y = torch.roll(x,  1, dims=3)   # x[i, j-1]

    # first derivatives
    wx_x  = (xp1_x - xm1_x) / (2.0 * dx)
    wx_y  = (xp1_y - xm1_y) / (2.0 * dy)
    wx    = torch.cat([wx_x, wx_y], dim=1)

    return wx
