#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
#import torch.nn as nn
import torch.optim as optim
from NN_Model import Model, ResidualMLP_6Layers, ResidualMLP_MBlocks
from torch.optim.swa_utils import AveragedModel, SWALR
#
import numpy as np
from matplotlib import pyplot as plt
#
from time import time
import os


# In[2]:


# Torch version
print('torch version: ', torch.__version__)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print('torch device:', device)


# In[3]:


# Parameters
##########################
# DIM = 2
# COLLOC_POINTS = int(20**DIM)
# LEARNING_RATE = 0.001
# DECAY_RATE    = 0.95
# DECAY_STEPS   = 2000
##gamma         = DECAY_RATE ** (1 / DECAY_STEPS)
# EPOCHS        = 10000
# NN_STRUCTURE  = [DIM,64,64,64,1]
##########################
# DIM = 3
# COLLOC_POINTS = int(20**DIM)
# LEARNING_RATE = 0.001
# DECAY_RATE    = 0.95
# DECAY_STEPS   = 2000
##gamma         = DECAY_RATE ** (1 / DECAY_STEPS)
# EPOCHS        = 10000
# NN_STRUCTURE  = [DIM,64,64,64,1]
##########################
# DIM = 4
# COLLOC_POINTS = int(20**DIM)
# LEARNING_RATE = 0.001
# DECAY_RATE    = 0.95
# DECAY_STEPS   = 2000
# EPOCHS        = 20000
# NN_STRUCTURE  = [DIM,64,64,64,64,1]
##########################
# DIM = 5
# COLLOC_POINTS = int(10**DIM)
# LEARNING_RATE = 0.001
# DECAY_RATE    = 0.95
# DECAY_STEPS   = 3000
# EPOCHS        = 50000
# NN_STRUCTURE  = None #ResidualMLP_6Layers
# WIDTH          = 64
##########################
DIM = 6
COLLOC_POINTS = int(10**5)
LEARNING_RATE = 0.001
DECAY_RATE    = 0.95
DECAY_STEPS   = 4000
EPOCHS        = 200000
# NN_STRUCTURE  = None #ResidualMLP_6Layers
WIDTH          = 32
RES_BLOCKS     = 5
##########################
print('DIMENSION         :',DIM)
print('COLLOCATION POINTS:',COLLOC_POINTS)

# Target function
# def f_rhs(x): #1, x2, x3):
    
#     aux = torch.sin(np.pi*x1)*torch.sin(np.pi*x2)*torch.sin(np.pi*x3)
#     return aux
def f_rhs(x):
    """
    x: tensor of shape (N, DIM)
    returns: tensor of shape (N,)
    f(x) = prod_i sin(pi * x_i)
    """
    return torch.cos(np.pi * x).prod(dim=1).view(-1,1)

def lossRes(x):
    #
    u = model(x) #1,x2,x3)
    #
    f = f_rhs(x) #1,x2,x3)
    #
    #residual = torch.mean((u - f)**2)
    residual = (2**DIM)*torch.mean((u - f)**2)
    return residual

# def random_domain_points(N):
#     x1 = torch.rand(N,1,requires_grad=True)
#     x2 = torch.rand(N,1,requires_grad=True)
#     x3 = torch.rand(N,1,requires_grad=True)
#     return (x1,x2,x3)
# def random_domain_points(N, DIM):
#     aux = tuple(torch.rand(N, 1, requires_grad=True) for _ in range(DIM))
#     return aux #tuple output necessary for PINNs (partial derivatives)
def random_domain_points(N, DIM):
    # Returns a tensor of size (N, DIM) with grad enabled
    x = 2*torch.rand(N, DIM, device= device, requires_grad=True) - 1.
    return x

# model = Model(NN_STRUCTURE).to(device)
# model = ResidualMLP_6Layers(
#     input_dim = DIM,
#     width     = WIDTH,
#     output_dim = 1
# ).to(device)
model = ResidualMLP_MBlocks(
    input_dim = DIM,
    width     = WIDTH,
    output_dim = 1,
    M = RES_BLOCKS  
).to(device)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                      step_size=DECAY_STEPS, 
                                      gamma=DECAY_RATE
)
##############
swa_model = AveragedModel(model)
swa_start = int(0.9*EPOCHS)   # start SWA at 75% of training
swa_scheduler = None #SWALR(optimizer, swa_lr=LEARNING_RATE)
##############

loss_list = []
#
start_time = time()
#
for epoch in range(int(EPOCHS)):
    optimizer.zero_grad() # to make the gradients zero
    #
    #x1, x2, x3 = random_domain_points(COLLOC_POINTS,3)
    x = random_domain_points(COLLOC_POINTS,DIM)
    #
    loss = lossRes(x)
    loss_list.append(loss.item())
    loss.backward() # This is for computing gradients using backward propagation
    optimizer.step() 
    # scheduler.step()
    # --- SWA logic here ---
    if epoch == swa_start:
        # <<< Read the final StepLR LR here >>>
        last_step_lr = optimizer.param_groups[0]["lr"]
        # Now create your SWA scheduler using that LR
        swa_scheduler = SWALR(optimizer, swa_lr=last_step_lr)
        print(f"Starting SWA with learning rate: {last_step_lr:.3e}")
    #    
    if epoch < swa_start:
        scheduler.step()               # normal StepLR phase  
    elif epoch == swa_start:
        # First SWA epoch: update parameters but DO NOT step scheduler yet
        swa_model.update_parameters(model)    
    else:
        swa_model.update_parameters(model)
        swa_scheduler.step()      
    # --- SWA logic here ---  
    if epoch % 500 == 0:
        current_lr = optimizer.param_groups[0]["lr"] #scheduler.get_last_lr()[0]  # Get the current learning rate
        print(f'Epoch: {epoch} - Loss: {loss.item():>1.3e} - Learning Rate: {current_lr:>1.3e}')
current_time = (time() - start_time) / 60        
print('Computing time', current_time, '[min]')        

plt.semilogy(loss_list)
title_str = f"computing time {current_time:.2f} [min]"
plt.title(title_str)
# file_str = 'loss_dim_'+str(DIM)+' (computing time'+str(current_time)+').png'
file_str = f"loss_dim_{DIM:02d}.png"
plt.savefig(file_str, dpi=120)
plt.close()


N_test = 256
idx = 1
x_test = torch.zeros(256,DIM)
x_slice = torch.linspace(-1.,1.,N_test)
x_test[:,idx] = x_slice#.view(-1,1)
#
# u_pred = model(x_test.to(device))
swa_model.eval()
u_pred = swa_model(x_test.to(device))
#
u_exact = f_rhs(x_test)

plt.plot(x_slice,u_pred.cpu().detach().numpy(),label='NN approx')
plt.plot(x_slice,u_exact, label='function')
title_str = 'dim: '+str(DIM)+'; train points: '+str(COLLOC_POINTS)+'; epochs: '+str(EPOCHS)
plt.title(title_str)
plt.legend()
file_str = f'output_dim_{DIM:02d}.png'
plt.savefig(file_str, dpi=120)
plt.close()





