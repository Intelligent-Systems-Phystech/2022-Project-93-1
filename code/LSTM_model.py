import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image

from typing import Tuple, List, Type, Dict, Any
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.1)

class LSTM_Block(nn.Module):
    def __init__(self,  input_size, hidden_size, num_layers,output_size, image_shape):
        super(LSTM_Block, self).__init__()
      
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.output_size = output_size
        self.hidden_size = hidden_size #hidden state
        self.image_shape = image_shape
        self.conv = nn.Sequential(
            torch.nn.Conv2d(1, 4, 2),
            nn.ReLU(),
            torch.nn.Conv2d(4, 8, 2, 3),
            nn.ReLU(),
            torch.nn.Conv2d(8, 16, 2, stride = 2, padding  = 1),
            nn.ReLU(),
        )
        


        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        
        
        self.up_conv =   nn.Sequential(
                                            nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                                            nn.ConvTranspose2d(1,4,3, padding = 1, stride= 1),
                                           
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                                            nn.ConvTranspose2d(4,8,3, padding = 1, stride= 1),
                                           
                                            nn.ReLU(),
                                            nn.Upsample(size=self.image_shape, mode='bilinear'),
                                            nn.ConvTranspose2d(8,self.output_size,3, padding = 1, stride= 1),
                                            
        )

    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE) #internal state
        # Propagate input through LSTM
        batch_size = x.shape[0]
        x = torch.reshape(x, (x.shape[0]*x.shape[1],1 , *x.shape[2:]))
     
        x = self.conv(x)

        x = torch.reshape(x, (batch_size, int(x.shape[0]/batch_size), *x.shape[1:]))
        x = torch.flatten(x,start_dim= 2)

        x, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        x = hn[-1].view(-1, self.hidden_size) 
       
        x = torch.reshape(x, (x.shape[0], 1 , int(np.sqrt(self.hidden_size)), int(np.sqrt(self.hidden_size)) ))
      
        x = self.up_conv(x)
        x = torch.reshape(x, (batch_size, *x.shape[1:]))
      
        return x
