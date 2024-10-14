"""
Created on Tue Jul 10 2022
Last Modified on Thu Apr 6 2023

@author: Agamdeep Chopra, achopra4@uw.edu
@affiliation: University of Washington, Seattle WA
@reference: Thevenot, A. (2022, February 17). Implement canny edge detection
            from scratch with PyTorch. Medium. Retrieved July 10, 2022, from
            https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
"""
from numpy import asarray, float32
from torch import from_numpy, sum as tsum, stack, cat, float32 as tfloat32
from torch.nn import Module, Conv3d, L1Loss
from torch.nn.functional import pad as tpad


def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    '''
    Returns 3D Sobel kernels Sx, Sy, Sz, & diagonal kernels for edge detection.

    Parameters
    ----------
    n1 : int, optional
        Kernel value 1 (default 1).
    n2 : int, optional
        Kernel value 2 (default 2).
    n3 : int, optional
        Kernel value 3 (default 2).

    Returns
    -------
    list
        List of all the 3D Sobel kernels (Sx, Sy, Sz, diagonal kernels).
    '''
    Sx = asarray(
        [[[-n1, 0, n1],
          [-n2, 0, n2],
          [-n1, 0, n1]],
         [[-n2, 0, n2],
          [-n3*n2, 0, n3*n2],
          [-n2, 0, n2]],
         [[-n1, 0, n1],
          [-n2, 0, n2],
          [-n1, 0, n1]]])

    Sy = asarray(
        [[[-n1, -n2, -n1],
          [0, 0, 0],
          [n1, n2, n1]],
         [[-n2, -n3*n2, -n2],
          [0, 0, 0],
          [n2, n3*n2, n2]],
         [[-n1, -n2, -n1],
          [0, 0, 0],
          [n1, n2, n1]]])

    Sz = asarray(
        [[[-n1, -n2, -n1],
          [-n2, -n3*n2, -n2],
          [-n1, -n2, -n1]],
         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
         [[n1, n2, n1],
          [n2, n3*n2, n2],
          [n1, n2, n1]]])

    Sd11 = asarray(
        [[[0, n1, n2],
          [-n1, 0, n1],
          [-n2, -n1, 0]],
         [[0, n2, n2*n3],
          [-n2, 0, n2],
          [-n2*n3, -n2, 0]],
         [[0, n1, n2],
          [-n1, 0, n1],
          [-n2, -n1, 0]]])

    Sd12 = asarray(
        [[[-n2, -n1, 0],
          [-n1, 0, n1],
          [0, n1, n2]],
         [[-n2*n3, -n2, 0],
          [-n2, 0, n2],
          [0, n2, n2*n3]],
         [[-n2, -n1, 0],
          [-n1, 0, n1],
          [0, n1, n2]]])

    Sd21 = Sd11.T
    Sd22 = Sd12.T
    Sd31 = asarray([-S.T for S in Sd11.T])
    Sd32 = asarray([S.T for S in Sd12.T])

    return [Sx, Sy, Sz, Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]


class GradEdge3D():
    '''
    Implements Sobel edge detection for 3D images using PyTorch.

    Parameters
    ----------
    n1 : int, optional
        Filter size for the first dimension (default is 1).
    n2 : int, optional
        Filter size for the second dimension (default is 2).
    n3 : int, optional
        Filter size for the third dimension (default is 2).
    '''

    def __init__(self, n1=1, n2=2, n3=2):
        super(GradEdge3D, self).__init__()
        k_sobel = 3
        S = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filters = []

        # Initialize Sobel filters for edge detection
        for s in S:
            sobel_filter = Conv3d(
                in_channels=1, out_channels=1, stride=1,
                kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
            sobel_filter.weight.data = from_numpy(
                s.astype(float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
            sobel_filter = sobel_filter.to(dtype=tfloat32)
            self.sobel_filters.append(sobel_filter)

    def __call__(self, img, a=1):
        '''
        Perform edge detection on the given 3D image.

        Parameters
        ----------
        img : torch.Tensor
            3D input tensor of shape (B, C, x, y, z).
        a : int, optional
            Padding size (default is 1).

        Returns
        -------
        torch.Tensor
            Tensor containing the gradient magnitudes of the edges.
        '''
        pad = (a, a, a, a, a, a)
        B, C, H, W, D = img.shape

        img = tpad(img, pad, mode='reflect')

        # Calculate gradient magnitude of edges
        grad_mag = (1 / C) * tsum(stack([tsum(cat(
            [s.to(img.device)(img[:, c:c+1]) for c in range(C)],
            dim=1) + 1e-6, dim=1) ** 2 for s in self.sobel_filters],
            dim=1) + 1e-6, dim=1) ** 0.5
        grad_mag = grad_mag[:, a:-a, a:-a, a:-a]

        num = grad_mag - grad_mag.min()
        dnm = grad_mag.max() - grad_mag.min() + 1e-6
        norm_grad_mag = num / dnm

        return norm_grad_mag.view(B, 1, H, W, D)


class GMELoss3D(Module):
    '''
    Implements Gradient Magnitude Edge Loss for 3D image data.

    Parameters
    ----------
    n1 : int
        Filter size for the first dimension.
    n2 : int
        Filter size for the second dimension.
    n3 : int
        Filter size for the third dimension.
    lam_errors : list
        List of tuples (weight, loss function) for computing error.
    reduction : str
        Reduction method for loss ('sum' or 'mean').
    '''

    def __init__(self, n1=1, n2=2, n3=2,
                 lam_errors=[(1.0, L1Loss())], reduction='sum'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(n1, n2, n3)
        self.lam_errors = lam_errors
        self.reduction = reduction

    def forward(self, x, y):
        '''
        Compute the loss based on the edges detected in the input tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, x, y, z).
        y : torch.Tensor
            Target tensor of shape (B, C, x, y, z).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        '''
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'

        edge_x = self.edge_filter(x)
        edge_y = self.edge_filter(y)

        if self.reduction == 'sum':
            error = 1e-6 + sum([lam * err_func(edge_x, edge_y)
                                for lam, err_func in self.lam_errors])
        else:
            error = 1e-6 + (
                sum(
                    [lam * err_func(
                        edge_x, edge_y) for lam, err_func in self.lam_errors]
                ) / len(self.lam_errors))

        return error
