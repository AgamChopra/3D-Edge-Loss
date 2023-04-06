# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 2022

@author: Agam Chopra, achopra4@uw.edu
@affiliation: Department of Mechanical Engineering, University of Washington, Seattle WA
@reference: Thevenot, A. (2022, February 17). Implement canny edge detection from scratch with PyTorch. Medium. Retrieved July 10, 2022, from https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed 
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

EPSILON = 1E-12


def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    Sx = np.asarray([[[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]], [[-n2, 0, n2],
                    [-n3*n2, 0, n3*n2], [-n2, 0, n2]], [[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]]])
    Sy = np.asarray([[[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]], [[-n2, -n3*n2, -n2],
                    [0, 0, 0], [n2, n3*n2, n2]], [[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]]])
    Sz = np.asarray([[[-n1, -n2, -n1], [-n2, -n3*n2, -n2], [-n1, -n2, -n1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[n1, n2, n1], [n2, n3*n2, n2], [n1, n2, n1]]])

    Sd11 = np.asarray([[[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]], [[0, n2, n2*n3],
                      [-n2, 0, n2], [-n2*n3, -n2, 0]], [[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]]])
    Sd12 = np.asarray([[[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]], [[-n2*n3, -n2, 0],
                      [-n2, 0, n2], [0, n2, n2*n3]], [[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]]])

    Sd21 = Sd11.T
    Sd22 = Sd12.T

    Sd31 = np.asarray([-S.T for S in Sd11.T])
    Sd32 = np.asarray([S.T for S in Sd12.T])

    return [Sx, Sy, Sz, Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]


class GradEdge3D():
    def __init__(self, n1=1, n2=2, n3=2, device='cpu'):
        super(GradEdge3D, self).__init__()
        self.device = device
        k_sobel = 3
        S = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filters = []

        for s in S:
            sobel_filter = nn.Conv3d(in_channels=1, out_channels=1, stride=1,
                                     kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
            sobel_filter.weight.data = torch.from_numpy(
                s.astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
            sobel_filter = sobel_filter.to(device, dtype=torch.float32)
            self.sobel_filters.append(sobel_filter)

    def detect(self, img, a=1):
        pad = (a, a, a, a, a, a)
        B, C, H, W, D = img.shape

        img = nn.functional.pad(img, pad, mode='reflect')

        grad_mag = (1 / C) * torch.sum(torch.stack([torch.sum(torch.cat([s(img[:, c:c+1])for c in range(
            C)], dim=1), dim=1) ** 2 for s in self.sobel_filters], dim=1), dim=1) ** 0.5
        grad_mag = grad_mag[:, a:-a, a:-a, a:-a]

        return grad_mag.view(B, 1, H, W, D)


class GMELoss3D(nn.Module):
    def __init__(self, criterion=nn.MSELoss(), n1=1, n2=2, n3=2, device='cpu'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(n1, n2, n3, device)
        self.criterion = criterion

    def forward(self, y, yp):
        y_edge = self.edge_filter.detect(y)
        yp_edge = self.edge_filter.detect(yp)
        error = self.criterion(y_edge, yp_edge)

        return error


if __name__ == "__main__":
    loss = GMELoss3D()
    filter_ = GradEdge3D(n1=1, n2=2, n3=2)
    for k in range(1, 5):
        # path = 'R:/img (%d).pkl' % (k)  # '/brats/data/img (%d).pkl'%(k
        #data = np.load(path, allow_pickle=True)
        # x = torch.from_numpy(data[0]).view(
        #    1, 1, data[0].shape[0], data[0].shape[1], data[0].shape[2]).to(dtype=torch.float)
        x = torch.rand((1, 1, 150, 150, 20))
        Y = filter_.detect(x)
        print(Y.shape)
        titles = ['grad_magnitude']
        for j in range(0, 20, 1):
            for i, y in enumerate(Y):
                if i == 0:
                    cmap = 'gray'
                elif i == 4 or i == 5:
                    cmap = 'bone'
                elif i == 6:
                    cmap = 'binary'
                else:
                    cmap = 'PiYG'
                plt.imshow(y[0, :, :, j].squeeze(
                ).cpu().detach().numpy().T, cmap=cmap)
                plt.title(titles[i] + ' slice%d' % (j))
                plt.show()
