import torch
from torch import nn

# Input img -> Hidden_dim -> mean, std -> Parametrization trick -> Decoder -> Output img
class VariationalAutoEncoder(nn.Module):
  def __init__(self, input_dim, h_dim= 100, z_dim= 10):
    super().__init__()
    # for encoder
    self.img_2hid = nn.Linear(input_dim, h_dim)
    self.hid_2mu = nn.Linear(h_dim, z_dim)
    self.hid_2sigma = nn.Linear(h_dim, z_dim)

    # for decoder
    self.z_2hid = nn.Linear(z_dim, h_dim)
    self.hid_2img = nn.Linear(h_dim, input_dim)

    self.relu = nn.ReLU()

  def encode(self, x):
    h =  self.relu(self.img_2hid(x))
    #print("h shape: " + str(h.shape))
    mu = self.hid_2mu(h)
    #print("mu shape: " + str(mu.shape))
    sigma = self.hid_2sigma(h)
    #print("sigma shape: " + str(sigma.shape))
    return mu, sigma

  def decoder(self, z):
    h = self.relu(self.z_2hid(z))
    #print("h shape: " + str(h.shape))
    return torch.sigmoid(self.hid_2img(h)) # sigmoid() if img is normalized [0, 1] before encoding

  def forward(self, x):
    mu, sigma =self.encode(x)
    epsilon = torch.randn_like(sigma) # Gaussian-like distribution
    #print("epsilon shape: " + str(epsilon.shape))
    z_reparameterized = mu + sigma*epsilon
    #print("z_reparameterized shape: " + str(z_reparameterized.shape))
    x_reconstructed = self.decoder(z_reparameterized)
    #print("x_reconstructed shape: " + str(x_reconstructed.shape))
    return x_reconstructed, mu, sigma
    

# sanity check
x = torch.randn(4, 64*64) # e.g.
vae = VariationalAutoEncoder(input_dim= 64*64) 
x_reconstructed, mu, sigma = vae(x)
print(x_reconstructed.shape) # e.g. torch.Size([4, 4096])
print(mu.shape) # e.g. torch.Size([4, 10])
print(sigma.shape) # e.g. torch.Size([4, 10])


'''
# General loss term
reconstructed_loss = loss_fn(x_reconstructed, x)
kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
total_loss = reconstructed_loss + kl_div
'''
