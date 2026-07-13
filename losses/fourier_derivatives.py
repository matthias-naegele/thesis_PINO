import numpy as np
import torch
from torch import Tensor
from typing import List


def fourier_derivatives(x: Tensor, ell: List[float]) -> Tensor:
    """
    Fourier first-order derivative only (no second-order, saves GPU memory).
    Based on physicsnemo.models.layers.spectral_layers.fourier_derivatives.
    """

    # check that input shape maches domain length
    if len(x.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = x.shape[2:]
    dim = len(ell)

    # get device
    device = x.device

    # compute fourier transform
    x_h = torch.fft.fftn(x, dim=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )

    # compute first-order derivatives in fourier space
    j = torch.complex(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )
    wx_h = [j * k_x_i * x_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]

    # inverse fourier transform out
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in wx_h],
        dim=1,
    )
    return wx
