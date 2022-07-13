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


EPSILON = 1E-9


def get_gaussian_kernel3D(k=3, mu=0, sigma=1, normalize=True):
    gaussian_1D = np.linspace(-1, 1, k)
    x, y, z = np.meshgrid(gaussian_1D, gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2 + z** 2) ** 0.5

    gaussian_3D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2 + EPSILON))
    gaussian_3D = gaussian_3D / (((2 * np.pi + EPSILON) **(3/2)) * (sigma **3) + EPSILON)
    if normalize:
        gaussian_3D = gaussian_3D / (np.sum(gaussian_3D) + EPSILON)
        
    return gaussian_3D


def get_sobel_kernel3D():
    Sx = np.asarray([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
    Sy = np.asarray([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],[[-2, -4, -2], [0, 0, 0], [2, 4, 2]],[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
    Sz = np.asarray([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],[[0, 0, 0], [0, 0, 0], [0, 0, 0]],[[1, 2, 1], [2, 4, 2], [1, 2, 1]]])
    return Sx, Sy, Sz


class GradEdge3D():
    def __init__(self,k_gaussian=3,mu=0,sigma=1,device='cpu'):
        super(GradEdge3D, self).__init__()
        self.device = device

        gaussian_3D = get_gaussian_kernel3D(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=k_gaussian,padding=k_gaussian // 2,bias=False)
        self.gaussian_filter.weight.data=torch.from_numpy(gaussian_3D.astype(np.float32)).reshape(1,1,k_gaussian,k_gaussian,k_gaussian)
        
        self.gaussian_filter = self.gaussian_filter.to(device)
        
        k_sobel = 3
        Sx, Sy, Sz = get_sobel_kernel3D()
        self.sobel_filter_x = nn.Conv3d(in_channels=1,out_channels=1,stride = 1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        self.sobel_filter_x.weight.data=torch.from_numpy(Sx.astype(np.float32)).reshape(1,1,k_sobel,k_sobel,k_sobel)

        self.sobel_filter_y = nn.Conv3d(in_channels=1,out_channels=1,stride = 1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        self.sobel_filter_y.weight.data=torch.from_numpy(Sy.astype(np.float32)).reshape(1,1,k_sobel,k_sobel,k_sobel)
        
        self.sobel_filter_z = nn.Conv3d(in_channels=1,out_channels=1,stride = 1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        self.sobel_filter_z.weight.data=torch.from_numpy(Sz.astype(np.float32)).reshape(1,1,k_sobel,k_sobel,k_sobel)
        
        self.sobel_filter_x = self.sobel_filter_x.to(device)
        self.sobel_filter_y = self.sobel_filter_y.to(device)
        self.sobel_filter_z = self.sobel_filter_z.to(device)

    def detect(self, img, low_threshold=0.05, high_threshold=1, hysteresis=False):
        B, C, H, W, D = img.shape
        blurred = torch.zeros((B, C, H, W, D)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W, D)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W, D)).to(self.device)
        grad_z = torch.zeros((B, 1, H, W, D)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W, D)).to(self.device)

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])
            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])
            grad_z = grad_z + self.sobel_filter_z(blurred[:, c:c+1])

        grad_x, grad_y, grad_z = grad_x / C, grad_y / C, grad_z / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + EPSILON) ** 0.5
        norm_grad_mag = (grad_magnitude - torch.min(grad_magnitude))/(torch.max(grad_magnitude) - torch.min(grad_magnitude) + EPSILON)
        mask = torch.where(norm_grad_mag > low_threshold, 1, 0) * torch.where(norm_grad_mag < high_threshold, 1, 0)
        cleaned_grad_mag = grad_magnitude * mask

        return blurred, grad_x, grad_y, grad_z, grad_magnitude, cleaned_grad_mag, mask
    

class GMELoss3D():
    def __init__(self, criterion=nn.MSELoss(), k_gaussian=3, mu=0, sigma=1, k_sobel=3, device='cpu'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(k_gaussian, mu, sigma, device)
        self.criterion = criterion        

    def compute(self, y, yp):
        _,_,_,_,y_edge,_,_ = self.edge_filter.detect(y)
        _,_,_,_,yp_edge,_,_ = self.edge_filter.detect(yp)
        error = self.criterion(y_edge, yp_edge)
        return error
    
    
if __name__ == "__main__":  
    import matplotlib.pyplot as plt 
    loss = GMELoss3D()
    filter_ = GradEdge3D()
    for k in range(1,5):
        data = np.load('/brats/data/img (%d).pkl'%(k),allow_pickle=True)
        x = torch.from_numpy(data[0]).view(1,1,data[0].shape[0],data[0].shape[1],data[0].shape[2]).to(dtype=torch.float)
        Y = filter_.detect(x,low_threshold=0.05,high_threshold=0.1)
        titles = ['blurred', 'grad_x', 'grad_y', 'grad_z', 'grad_magnitude', 'cleaned_grad_mag','mask']
        for j in range(0,155,20):
            for i,y in enumerate(Y):
                if i == 0:
                    cmap='bone'
                elif i == 4 or i == 5:
                    cmap='hot'
                elif i == 6:
                    cmap='binary'
                else:
                    cmap='PiYG'
                plt.imshow(y[:,:,:,:,j].squeeze().cpu().detach().numpy().T,cmap = cmap)
                plt.title(titles[i] + ' slice%d'%(j))
                plt.show()
        print('sample_loss = ',loss.compute(torch.from_numpy(data[1]).view(1,1,data[0].shape[0],data[0].shape[1],data[0].shape[2]).to(dtype=torch.float),x))