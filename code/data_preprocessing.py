import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index,step):
    indices = range(i-history_size, i)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.squeeze(np.array(data)), np.squeeze(np.array(labels))



class DS(Dataset):
    def __init__(self, data,labels, transform=None):
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)
        self.transform = transform

    def __getitem__(self, index):

        x = self.data[index]
        y = self.labels[index]

    

        return x,y

    def __len__(self):
        return len(self.data)
