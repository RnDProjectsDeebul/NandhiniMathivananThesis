import torch.nn as nn
import torch.nn.functional as F
import torch


import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as I
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import torch
#import pretrainedmodels


class GraspKeypointModel(nn.Module):
    def __init__(self, freeze_resnet = False):
        super(GraspKeypointModel, self).__init__()
        
     
        self.conv1 = nn.Conv2d( in_channels=3, out_channels=3, kernel_size=(3, 3), stride=1,
                               padding=1, padding_mode='zeros' )
        
        # Resnet Architecture
        self.resnet18 = models.resnet18(pretrained=True)
        if freeze_resnet:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        # replacing last layer of resnet
        # by default requires_grad in a layer is True
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 512) 

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(512, 8) 
        self.variance = nn.Linear(512,8) 
        
    def forward(self, x):
      
        y0 = self.conv1(x)
     
        y1 = self.resnet18(y0)
        
        y_relu = self.relu(y1)
       
        out= self.linear1(y_relu)
        
        var = F.softplus(self.variance(y_relu))
        return out,var

# class GraspKeypointModel(nn.Module):
#     def __init__(self, pretrained, requires_grad):
#         super(GraspKeypointModel, self).__init__()
#         if pretrained == True:
#             self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
#         else:
#             self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
#         if requires_grad == True:
#             for param in self.model.parameters():
#                 param.requires_grad = True
#             print('Training intermediate layer parameters...')
#         elif requires_grad == False:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             print('Freezing intermediate layer parameters...')
#         # change the final layer
#         self.l0 = nn.Linear(2048, 8)
#         self.variance = nn.Linear(2048, 8)
#     def forward(self, x):
#         # get the batch size only, ignore (c, h, w)
#         batch, _, _, _ = x.shape
#         x = self.model.features(x)
#         x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
#         l0 = self.l0(x)
#         var = F.softplus(self.variance(x))
#         return l0, var

# class GraspKeypointModel(nn.Module):
#     def __init__(self):
#         super(GraspKeypointModel, self).__init__()
       
#         #defining maxpool block
#         self.maxpool = nn.MaxPool2d(2, 2)
               
#         #defining dropout block
#         self.dropout = nn.Dropout(p=0.2)
        
#         self.conv1 = nn.Conv2d(3, 32, 5)
        
#         #defining second convolutional layer
#         self.conv2 = nn.Conv2d(32, 64, 3)
        
#         #defining third convolutional layer
#         self.conv3 = nn.Conv2d(64, 128, 3)
        
#         #defining linear output layer
#         self.fc1 = nn.Linear(128*26*26, 8)
#         self.variance = nn.Linear(128*26*26,8) 
        
#     def forward(self, x):
        
#         #passing tensor x through first conv layer
#         x = self.maxpool(F.relu(self.conv1(x)))
     
#         #passing tensor x through second conv layer
#         x = self.maxpool(F.relu(self.conv2(x)))
        
#         #passing tensor x through third conv layer
#         x = self.maxpool(F.relu(self.conv3(x)))
        
#         print(x.size())
#         #flattening x tensor
#         x = x.view(x.size(0), -1)
        
#         #applying dropout
#         x_final = self.dropout(x)
     
#         #passing x through linear layer
#         x = self.fc1(x_final)
#         var = F.softplus(self.variance(x_final))
        
#         #returning x
#         return x,var

