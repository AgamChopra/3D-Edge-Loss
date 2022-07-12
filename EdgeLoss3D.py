# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 2022

@author: Agam Chopra, achopra4@uw.edu
@affiliation: Department of Mechanical Engineering, University of Washington, Seattle WA
@reference: Thevenot, A. (2022, February 17). Implement canny edge detection from scratch with pytorch. Medium. Retrieved July 10, 2022, from https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed 
"""
import numpy as np
import torch
import torch.nn as nn


def get_gaussian_kernel3D(k=3, mu=0, sigma=1, normalize=True):
    gaussian_1D = np.linspace(-1, 1, k)
    x, y, z = np.meshgrid(gaussian_1D, gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2 + z** 2) ** 0.5

    gaussian_3D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_3D = gaussian_3D / (((2 * np.pi) **(3/2)) * (sigma **3))
    if normalize:
        gaussian_3D = gaussian_3D / np.sum(gaussian_3D)
        
    return gaussian_3D


def get_sobel_kernel3D():
    Sx = np.asarray([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])/4
    Sy = np.asarray([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],[[-2, -4, -2], [0, 0, 0], [2, 4, 2]],[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])/4
    Sz = np.asarray([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],[[0, 0, 0], [0, 0, 0], [0, 0, 0]],[[1, 2, 1], [2, 4, 2], [1, 2, 1]]])/4
    return Sx, Sy, Sz


class GradEdge3D(nn.Module):
    def __init__(self,k_gaussian=3,mu=0,sigma=1,device='cpu'):
        super(GradEdge3D, self).__init__()
        self.device = device
        self.requires_grad = False
    
        gaussian_3D = get_gaussian_kernel3D(k_gaussian, mu, sigma)
            
        self.gaussian_filter = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=k_gaussian,padding=k_gaussian // 2,bias=False)
        self.gaussian_filter.weight.data=torch.from_numpy(gaussian_3D.astype(np.float32)).detach().reshape(1,1,k_gaussian,k_gaussian,k_gaussian)
            
        self.gaussian_filter = self.gaussian_filter.to(device)
            
        k_sobel = 3
        Sx, Sy, Sz = get_sobel_kernel3D()
            
        self.sobel_filter_x = nn.Conv3d(in_channels=1,out_channels=1,stride = 1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        self.sobel_filter_x.weight.data=torch.from_numpy(Sx.astype(np.float32)).detach().reshape(1,1,k_sobel,k_sobel,k_sobel)
    
        self.sobel_filter_y = nn.Conv3d(in_channels=1,out_channels=1,stride = 1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        self.sobel_filter_y.weight.data=torch.from_numpy(Sy.astype(np.float32)).detach().reshape(1,1,k_sobel,k_sobel,k_sobel)
            
        self.sobel_filter_z = nn.Conv3d(in_channels=1,out_channels=1,stride = 1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        self.sobel_filter_z.weight.data=torch.from_numpy(Sz.astype(np.float32)).detach().reshape(1,1,k_sobel,k_sobel,k_sobel)
            
        self.sobel_filter_x = self.sobel_filter_x.to(device)
        self.sobel_filter_y = self.sobel_filter_y.to(device)
        self.sobel_filter_z = self.sobel_filter_z.to(device)

    def forward(self, img):
        B, C, H, W, D = img.shape
        blurred = torch.zeros((B, C, H, W, D), requires_grad = self.requires_grad).to(self.device)
        grad_x = torch.zeros((B, 1, H, W, D), requires_grad = self.requires_grad).to(self.device)
        grad_y = torch.zeros((B, 1, H, W, D), requires_grad = self.requires_grad).to(self.device)
        grad_z = torch.zeros((B, 1, H, W, D), requires_grad = self.requires_grad).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W, D), requires_grad = self.requires_grad).to(self.device)

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])
            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])
            grad_z = grad_z + self.sobel_filter_z(blurred[:, c:c+1])

        grad_x, grad_y, grad_z = grad_x / C, grad_y / C, grad_z / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2 + grad_z ** 2) ** 0.5
        norm_grad_mag = (grad_magnitude - torch.min(grad_magnitude))/(torch.max(grad_magnitude) - torch.min(grad_magnitude))
        mask = torch.where(norm_grad_mag > 0.1, 1, 0)
        cleaned_grad_mag = grad_magnitude * mask

        return blurred, grad_x, grad_y, grad_z, grad_magnitude, cleaned_grad_mag
    

class GMELoss3D(nn.Module):
    def __init__(self, criterion=nn.MSELoss(), k_gaussian=3, mu=0, sigma=1, k_sobel=3, device='cpu'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(k_gaussian, mu, sigma, device)
        self.criterion = criterion        

    def forward(self, y, yp):
        with torch.no_grad():
            _,_,_,_,_,y_edge = self.edge_filter.forward(y)
            _,_,_,_,_,yp_edge = self.edge_filter.forward(yp)
            error = self.criterion(y_edge, yp_edge)
        return error
    
    
if __name__ == "__main__":  
    import matplotlib.pyplot as plt  
    data = np.load('R:/BraTSReg_109.pkl',allow_pickle=True)
    x = torch.from_numpy(data[0]).view(1,1,data[0].shape[0],data[0].shape[1],data[0].shape[2]).to(dtype=torch.float)
    loss = GMELoss3D()
    filter_ = GradEdge3D()
    Y = filter_(x)
    titles = ['blurred', 'grad_x', 'grad_y', 'grad_z', 'grad_magnitude', 'cleaned_grad_mag']
    for j in range(0,155,1):
        for i,y in enumerate(Y):
            if i == 0:
                cmap='bone'
            elif i == 4 or i == 5:
                cmap='hot'
            else:
                cmap='PiYG'
            if i == 4:
                plt.imshow(y[:,:,:,:,j].squeeze().cpu().detach().numpy().T,cmap = cmap)
                plt.title(titles[i] + ' slice%d'%(j))
                plt.show()
    print('sample_loss = ',loss(torch.from_numpy(data[1]).view(1,1,data[0].shape[0],data[0].shape[1],data[0].shape[2]).to(dtype=torch.float),x)) 
