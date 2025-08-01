
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout): # input_size: vocab_size, dropout: dropout probability
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout= dropout)

  def forward(self, x):
    embedding = self.dropout(self.embedding(x)) # (seq_length, batch) -> (seq_length, batch, embedding_size)
    outputs, (hidden, cell) = self.rnn(embedding) # context vector (hidden, cell) needed for language translation
    return hidden, cell

class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout) 
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout= dropout)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden, cell):
    x = x.unsqueeze(0) # to make (seq_length, batch) shape
    embedding = self.dropout(self.embedding(x))
    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
    predictions = self.fc(outputs)
    predictions = predictions.squeeze(0)

    return predictions, hidden, cell

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, source, target, teacher_force_ratio= 0.5):
    batch_size = source.shape[1] # source: (seq_len, batch_size)
    target_len = target.shape[0] # target: (seq_len, batch_size)`
    target_vocab_size = 1000
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    hidden, cell = self.encoder(source)
    
    x = target[0] 
    for t in range(1, target_len):
      output, hidden, cell = self.decoder(x, hidden, cell) 
      outputs[t] = output 
      best_guess = output.argmax(1)
      x = target[t] if random.random() < teacher_force_ratio else best_guess
    return outputs

'''
# Training hyperparameters
num_epochs = 3
learning_rate = 0.001
batch_size = 16

# Model hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = 1000
input_size_decoder = 1000
output_size = 1000
encoder_embedding_size = 128
decoder_embedding_size = 128
num_layers = 2
hidden_size = 256
enc_dropout = 0.5
dec_dropout = 0.5

# Model initiation
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
criterion = nn.CrossEntropyLoss()

# training
for epoch in range(num_epochs):
  print(f'Epoch[{epoch}/ {num_epochs}]')

  inp_data = torch.randint(0, 10, (15, 32)).to(device)
  target = torch.randint(0,2, (20, 32)).to(device)

  output = model(inp_data, target)
  print(output.shape) # output shape: (trg_len, batch_size, output_dim)

  output = output[1:].reshape(-1, output.shape[2]) # remove start token
  target = target[1:].reshape(-1)
  print(target.shape) #target shape: (trg_len-1 * batch_size)

  optimizer.zero_grad()
  loss = criterion(output, target)

  loss.backward()
  optimizer.step()
  print(loss.item())
'''
