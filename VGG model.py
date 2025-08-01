import torch
import torch.nn as nn


VGG_types = {
    'VGG11': [64, 'Max_Pool', 128, 'Max_Pool', 256, 256, 'Max_Pool', 512, 512, 'Max_Pool', 512, 512, 'Max_Pool'],
    'VGG13': [64, 64, 'Max_Pool', 128, 128, 'Max_Pool', 256, 256, 'Max_Pool', 512, 512, 'Max_Pool', 512, 512, 'Max_Pool'],
    'VGG16': [64, 64, 'Max_Pool', 128, 128, 'Max_Pool', 256, 256, 256, 'Max_Pool', 512, 512, 512, 'Max_Pool', 512, 512, 512, 'Max_Pool'],
    'VGG19': [64, 64, 'Max_Pool', 128, 128, 'Max_Pool', 256, 256, 256, 256, 'Max_Pool', 512, 512, 512, 512, 'Max_Pool', 512, 512, 512, 512, 'Max_Pool']
} 

class VGG_net(nn.Module):
  def __init__(self, in_channels= 3, num_classes= 100): # e.g. 100 for num_classes
    super(VGG_net, self).__init__()
    self.in_channels = in_channels
    self.conv_layers = self.create_conv_layers(VGG_types['VGG16']) # VGG11, VGG13, VGG16, VGG19

    self.fc = nn.Sequential(nn.Linear(512*7*7, 4096), # if e.g. original (224x224) -> (7x7) due to five MaxPooling
                            nn.ReLU(),
                            nn.Dropout(p= 0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            nn.Dropout(p= 0.5),
                            nn.Linear(4096, num_classes))

  def forward(self, x):
    x = self.conv_layers(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x

  def create_conv_layers(self, architecture):
    layers = []
    in_channels = self.in_channels

    for x in architecture:
      if type(x) == int:
        out_channels = x
        layers += [nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size=(3, 3), stride= (1, 1), padding= (1, 1)),
                   nn.BatchNorm2d(x),
                   nn.ReLU()]
        in_channels = x
      elif x == 'Max_Pool':
        layers += [nn.MaxPool2d(kernel_size= (2, 2), stride= (2, 2))]

    return nn.Sequential(*layers)
  

#Sanity check
#model = VGG_net(in_channels= 3, num_classes= 100).to(device)
#x = torch.randn(1, 3, 224, 224)
#print(model(x).shape)
