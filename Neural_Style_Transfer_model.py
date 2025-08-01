import torch
import torch.nn as nn
import torchvision.models as models


model = models.vgg19(pretrained= True).features
#print(model)

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.chosen_features = ['0', '5', '10', '19', '28'] # used each first conv layer
    self.model = models.vgg19(pretrained= True).features[:29] 

  def forward(self, x):
    features = []
    for layer_num, layer in enumerate(self.model):
      x = layer(x)

      if str(layer_num) in self.chosen_features:
        features.append(x)

    return features

'''
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_file):
  #image = Image.open(image_name)
  image = transforms.Compose([transforms.Resize((256, 256)), # e.g. image_size = 356
                              transforms.ToTensor()])(image)
  image = image.unsqueeze(0)# additional dimension for batch or sequence
  return image.to('cpu')

original_img = load_image("img_file")
style_img = load_image("img_file") 

# hyperparameters
total_steps = 1000
learning_rate = 0.001
alpha = 0.5 # content loss weight
beta = 0.05 # style loss weight

model = VGG().to('cpu')
generated = original_img.clone().requires_grad_(True) # typically, original img used
optimizer = torch.optim.Adam([generated], lr= learning_rate) 

for step in range(total_steps): 
  generated_features = model(generated)
  original_img_features = model(original_img)
  style_features = model(style_img)
  style_loss = original_loss = 0

  for gen_feature, orig_feature, style_feature in zip(generated_features,,original_img_features,,style_features): 
    bathch_size, channel, height, width = gen_feature.shape

    # Content Loss
    original_loss += torch.mean((gen_feature - orig_feature)**2) # MSE loss

    # Style Loss
    G = gen_feature.view(channel, height*width).mm(gen_feature.view(channel, height*width).t()) # matmul reason: get correlation
    A = style_feature.view(channel, height*width).mm(style_feature.view(channel, height*width).t()) # (channel x pixel) -> (channel x channel)
    style_loss += torch.mean((G-A)**2)

  total_loss = alpha*original_loss + beta*style_loss
'''
