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

    return Sx, Sy, Sz, [Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]


class GradEdge3D():
    def __init__(self, n1=1, n2=2, n3=2, device='cpu'):
        super(GradEdge3D, self).__init__()
        self.device = device
        k_sobel = 3
        Sx, Sy, Sz, Sd = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filter_x = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_x.weight.data = torch.from_numpy(
            Sx.astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_y = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_y.weight.data = torch.from_numpy(
            Sy.astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_z = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_z.weight.data = torch.from_numpy(
            Sz.astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_d1 = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_d1.weight.data = torch.from_numpy(
            Sd[0].astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_d2 = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_d2.weight.data = torch.from_numpy(
            Sd[1].astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_d3 = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_d3.weight.data = torch.from_numpy(
            Sd[2].astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_d4 = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_d4.weight.data = torch.from_numpy(
            Sd[3].astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_d5 = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_d5.weight.data = torch.from_numpy(
            Sd[4].astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
        self.sobel_filter_d6 = nn.Conv3d(
            in_channels=1, out_channels=1, stride=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
        self.sobel_filter_d6.weight.data = torch.from_numpy(
            Sd[5].astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)

        self.sobel_filter_x = self.sobel_filter_x.to(
            device, dtype=torch.float32)
        self.sobel_filter_y = self.sobel_filter_y.to(
            device, dtype=torch.float32)
        self.sobel_filter_z = self.sobel_filter_z.to(
            device, dtype=torch.float32)
        self.sobel_filter_d1 = self.sobel_filter_d1.to(
            device, dtype=torch.float32)
        self.sobel_filter_d2 = self.sobel_filter_d2.to(
            device, dtype=torch.float32)
        self.sobel_filter_d3 = self.sobel_filter_d3.to(
            device, dtype=torch.float32)
        self.sobel_filter_d4 = self.sobel_filter_d4.to(
            device, dtype=torch.float32)
        self.sobel_filter_d5 = self.sobel_filter_d5.to(
            device, dtype=torch.float32)
        self.sobel_filter_d6 = self.sobel_filter_d6.to(
            device, dtype=torch.float32)

    def detect(self, img):
        B, C, H, W, D = img.shape
        grad_x, grad_y, grad_z, grad_d1, grad_d2, grad_d3, grad_d4, grad_d5, grad_d6 = [
        ], [], [], [], [], [], [], [], []

        for c in range(C):
            grad_x.append(self.sobel_filter_x(img[:, c:c+1]))
            grad_y.append(self.sobel_filter_y(img[:, c:c+1]))
            grad_z.append(self.sobel_filter_z(img[:, c:c+1]))
            grad_d1.append(self.sobel_filter_d1(img[:, c:c+1]))
            grad_d2.append(self.sobel_filter_d2(img[:, c:c+1]))
            grad_d3.append(self.sobel_filter_d3(img[:, c:c+1]))
            grad_d4.append(self.sobel_filter_d4(img[:, c:c+1]))
            grad_d5.append(self.sobel_filter_d5(img[:, c:c+1]))
            grad_d6.append(self.sobel_filter_d6(img[:, c:c+1]))

        grad_magnitude = (1 / C) * (((torch.sum(grad_x, dim=1)) ** 2 + (torch.sum(grad_y, dim=1)) ** 2 + (torch.sum(grad_z, dim=1)) ** 2 + (torch.sum(grad_d1, dim=1)) ** 2 + (torch.sum(
            grad_d2, dim=1)) ** 2 + (torch.sum(grad_d3, dim=1)) ** 2 + (torch.sum(grad_d4, dim=1)) ** 2 + (torch.sum(grad_d5, dim=1)) ** 2 + (torch.sum(grad_d6, dim=1)) ** 2 + EPSILON) ** 0.5)

        return grad_magnitude


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
    import matplotlib.pyplot as plt
    loss = GMELoss3D()
    filter_ = GradEdge3D(k_gaussian=3, mu=0, sigma=1, n1=1, n2=2, n3=2)
    for k in range(1, 5):
        path = 'R:/img (%d).pkl' % (k)  # '/brats/data/img (%d).pkl'%(k
        data = np.load(path, allow_pickle=True)
        x = torch.from_numpy(data[0]).view(
            1, 1, data[0].shape[0], data[0].shape[1], data[0].shape[2]).to(dtype=torch.float)
        Y = filter_.detect(x, low_threshold=0.05, high_threshold=0.15)
        titles = ['grad_magnitude']
        for j in range(0, 155, 20):
            for i, y in enumerate(Y):
                if i == 0:
                    cmap = 'bone'
                elif i == 4 or i == 5:
                    cmap = 'hot'
                elif i == 6:
                    cmap = 'binary'
                else:
                    cmap = 'PiYG'
                plt.imshow(y[:, :, :, :, j].squeeze(
                ).cpu().detach().numpy().T, cmap=cmap)
                plt.title(titles[i] + ' slice%d' % (j))
                plt.show()
