import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Discriminator(nn.Module):
  def __init__(self, img_dim):
    super(Discriminator, self).__init__() 
    self.discriminator = nn.Sequential(nn.Linear(img_dim, 128),
                              nn.LeakyReLU(0.1),
                              nn.Linear(128, 1), # the last value is real or fake
                              nn.Sigmoid())
  def forward(self, x):
    return self.discriminator(x)

class Generator(nn.Module):
  def __init__(self, z_dim, img_dim): # 
    super().__init__()
    self.generator = nn.Sequential(nn.Linear(z_dim, 256),
                             nn.LeakyReLU(0.1),
                             nn.Linear(256, img_dim), # e.g. MNIST image: img_dim = 28x28 -> 784
                             nn.Tanh(),) # [-1, 1]]
  def forward(self, x):
    return self.generator(x)

'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4 
z_dim = 128
image_dim = 28 * 28 * 1 # 784 MNIST as example
batch_size = 32
num_epochs = 5

discriminator = Discriminator(image_dim).to(device)
generator = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device) 
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

dataset = datasets.MNIST(root='dataset/', transform= transforms, download = True)
loader = DataLoader(dataset, batch_size= batch_size, shuffle= True)
optimizer_disc = optim.Adam(discriminator.parameters(), lr= lr) 
optimizer_gen = optim.Adam(generator.parameters(), lr= lr)
criterion = nn.BCELoss()


for epoch in range(num_epochs):
  for batch_idx, (real, _) in enumerate(loader): # (image, label) # don't need label --->  # GAN is unsupervised learning
    real = real.view(-1, 784).to(device) # (batch, MNIST_img)
    batch_size = real.shape[0]

    # Train Discriminator
    noise = torch.randn(batch_size, z_dim).to(device)
    fake = generator(noise)
    disc_real = discriminator(real).view(-1) # flattern
    lossD_real = criterion(disc_real, torch.ones_like(disc_real)) 

    disc_fake = discriminator(fake).view(-1)
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) 
    
    lossD = (lossD_real + lossD_fake) / 2
    
    optimizer_disc.zero_grad() 
    lossD.backward(retain_graph= True)
    optimizer_disc.step()

    # Train Generator
    output = discriminator(fake).view(-1)
    lossG = criterion(output, torch.ones_like(output))
    
    optimizer_gen.zero_grad()
    lossG.backward()
    optimizer_gen.step()
'''
