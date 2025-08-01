Image Captioning_Seq2Seq 
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
  def __init__(self, embed_size, train_CNN= False):
    super(EncoderCNN, self).__init__()
    self.train_CNN = train_CNN
    self.inception = models.inception_v3(pretrained= True, aux_logits= True) # inception_v3 model # aux_logits= True (default value)
    self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size) # inception_v3's FC layer change into embed size
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)

  def forward(self, images):
    features, aux_features = self.inception(images) # get features only # get embedding from image
    for name, param in self.inception.named_parameters():
      if "fc.weight" in name or "fc.bias" in name:
        param.requires_grad = True # last FC layer can be trainable
      else:
        param.requires_grad= self.train_CNN # maintaing weights except for last FC layers

    return self.dropout(self.relu(features))


class DecoderRNN(nn.Module): 
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
    super(DecoderRNN, self).__init__()
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
    self.linear = nn.Linear(hidden_size, vocab_size) 
    self.dropout = nn.Dropout(0.5)

  def forward(self, features, captions): 
    embeddings = self.dropout(self.embed(captions)) # if batch_first = False -> (seq_len, batch_size, embed_size)
    embeddings = torch.cat((features.unsqueeze(0), embeddings), dim= 0) # embedding shape (sequence_length + 1, batch_size, hidden_size)
    hiddens, _ = self.lstm(embeddings)  
    outputs = self.linear(hiddens) # softmax will be needed in future
    return outputs


class CNNtoRNN(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
    super(CNNtoRNN, self).__init__()
    self.encoderCNN = EncoderCNN(embed_size)
    self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

  def forward(self, images, captions):
    features = self.encoderCNN(images)
    outputs = self.decoderRNN(features, captions)
    return outputs

  def caption_image(self, image, vocabulary, max_length= 50):
    result_caption = []

    with torch.no_grad():
      x = self.encoderCNN(image).unsqueeze(0) # `image`: (1, C, H, W) -> (1, embed_size) -> (1, 1, embed_size)
      state = None
      
      for _ in range(max_length):
        hiddens, states = self.decoderRNN.lstm(x, state) # `hiddens`: (1, batch_size, hidden_size)
        output = self.decoderRNN.linear(hiddens.squeeze(0)) # (1, batch_size, hidden_size) -> (batch_size, hidden_size) -> (batch_size, vocab_size)
        predicted = output.argmax(1)

        # print(predicted.shape) # sanity check

        result_caption.append(predicted.item())
        x = self.decoderRNN.embed(predicted).unsqueeze(0) # input for seq to seq

        if vocabulary.itos[predicted.item()] == "<EOS>":
          break

    return [vocabulary.itos[idx] for idx in result_caption]


# sanity check
embed_size = 64
hidden_size = 128
vocab_size = 100
seq_len = 50 # max seq length as an example
batch_size = 4
captions = torch.randint(0, vocab_size, (seq_len, batch_size)).to('cpu')
num_layers = 2

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to('cpu')
x = torch.randn((batch_size, 3, 299, 299)).to('cpu') # inception_v3 input img size (299x299)
print(model(x, captions).shape) # torch.Size([50+1, 8, 100])



