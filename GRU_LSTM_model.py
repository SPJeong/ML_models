'''
GRU (faster, but less accuracy)
LSTM(slower, but high accuracy)
'''

import torch
import torch.nn as nn

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_layers = 256 # hidden_layers = hidden_size = hidden_state
num_classes = 10 # final destination
learning_rate = 0.001
batch_size = 64
num_epochs = 1

########## GRU_model ##########
class GRU(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(GRU, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first= True) # batch_first = True -> N x time_sequece x features
    self.fc = nn.Linear(sequence_length * hidden_size, num_classes)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cpu') # (num_layers, batch_size, hidden_size)
    out, _ = self.gru(x, h0)
    out = out.reshape(out.shape[0], -1) # this is the case for only one final target
    out = self.fc(out)
    return out

########## LSTM_model ##########
class lstm(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(lstm, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bi_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True, bidirectional= True) # in case of bidirectional = True
    self.fc = nn.Linear(hidden_size*2, num_classes)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to('cpu')
    c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to('cpu')

    out, _ = self.bi_lstm(x, (h0, c0)) # out shape: (batch_size, sequence_length, hidden_size)
    out = self.fc(out[:, -1, :]) # just take last hidden state for fast training
    return out

'''
# sanity check
model = lstm(input_size, hidden_layers, num_layers, num_classes).to('cpu')
x = torch.randn((2, 28, 28)).to('cpu')
print(model(x).shape) # torch.Size([2, 10])
'''
