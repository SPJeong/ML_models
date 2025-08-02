import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, channels_img, features_d): # feature_d: discriminator features
    super(Discriminator, self).__init__()
    self.disc = nn.Sequential(nn.Conv2d(channels_img, features_d, kernel_size= 4, stride= 2, padding= 1), # no BatchNorm for the first layer # 128 -> 64
                              nn.LeakyReLU(0.2),
                              self._block(features_d, features_d*2, 4, 2, 1), # half img size # 64 -> 32
                              self._block(features_d*2, features_d*4, 4, 2, 1), # half img size # 32 -> 16
                              self._block(features_d*4, features_d*8, 4, 2, 1), # half img size # 16 -> 8
                              nn.Conv2d(features_d*8, 1, kernel_size= 4, stride= 2, padding=0), # 8 -> 3
                              nn.AdaptiveAvgPool2d((1, 1)),
                              nn.Sigmoid(), )
  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= False),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2),)
  def forward(self, x):
    return self.disc(x)


class Generator(nn.Module):
  def __init__(self, z_dim, channels_img, features_g): # features_g: generator features
    super(Generator, self).__init__()
    self.gen = nn.Sequential(self._block(z_dim, features_g*16, 4, 1, 0), # upscaling
                             self._block(features_g*16, features_g*8, 4, 2, 1), # upscaling
                             self._block(features_g*8, features_g*4, 4, 2, 1), # upscaling
                             self._block(features_g*4, features_g*2, 4, 2, 1), # upscaling
                             self._block(features_g*2, features_g, 4, 2, 1), # upscaling
                             nn.ConvTranspose2d(features_g, channels_img, kernel_size= 4, stride= 2, padding= 1, ), 
                             nn.Tanh(), ) #  [-1 ~ 1]

  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias= False), # Upconvolution
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(),)
                         
  def forward(self, x):
    return self.gen(x)


def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.ConvTranspose2d):
      nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

'''
# sanity check
z_dim = 50
x = torch.randn((2, 3, 128, 128))
disc = Discriminator(channels_img= 3, features_d= 8)
initialize_weights(disc) 
print(disc(x).shape)
gen = Generator(z_dim, channels_img= 3, features_g= 8)
initialize_weights(gen) 
z = torch.randn((N, z_dim, 1, 1))
print(gen(z).shape)
'''

