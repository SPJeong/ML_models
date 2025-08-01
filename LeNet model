import torch
import torch.nn as nn

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.relu = nn.ReLU()
    self.pool = nn.AvgPool2d(kernel_size= 2, stride= 2)
    self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 6, kernel_size= 5, stride= 1, padding= 0)
    self.conv2 = nn.Conv2d(in_channels= 6, out_channels= 16, kernel_size= 5, stride= 1, padding= 0)
    self.conv3 = nn.Conv2d(in_channels= 16, out_channels= 120, kernel_size= 5, stride= 1, padding= 0)
    self.linear1 = nn.Linear(120*9*9, 84)
    self.linear2 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    #print(x.shape)
    x = self.pool(self.relu(self.conv2(x)))
    #print(x.shape)
    x = self.relu(self.conv3(x)) 
    #print(x.shape)
    x = x.reshape(x.shape[0], -1) 
    x = self.relu(self.linear1(x)) # 120*9*9 -> 84
    x = self.linear2(x)
    return x

# sanity check
#x = torch.randn(2, 1, 64, 64)
#model = LeNet()
#print(model(x).shape)
