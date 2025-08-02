import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout): 
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional= True, dropout= dropout) 
    self.fc_hidden = nn.Linear(hidden_size*2, hidden_size) 
    self.fc_cell = nn.Linear(hidden_size*2, hidden_size) 

  def forward(self, x):
    embedding = self.dropout(self.embedding(x))
    encoder_states, (hidden, cell) = self.rnn(embedding) # encoder_states: (seq_len, batch, h_size*2) # h or c: (num_layers*2, batch, h_size) due to bidrectional
    hidden = self.fc_hidden(torch.cat((hidden[0:self.num_layers], hidden[self.num_layers:]), dim= 2)) # forward hidden + backward hidden # dim = 2 (h_size)
    cell = self.fc_cell(torch.cat((cell[0:self.num_layers], cell[self.num_layers:]), dim=2)) # forward cell + backward cell # dim = 2 (h_size)
    return encoder_states, hidden, cell # encoder_states for attention


class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout): # input_size (vocab) 
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout= dropout) # (hidden_size*2 from bidrectional encoder) + (embed from vocab)
    self.energy = nn.Linear(hidden_size*3, 1) # hidden_size*3: concat (`h_reshaped` (hidden_size) + `encoder_states` (hidden_size*2))
    self.softmax = nn.Softmax(dim= 0)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, encoer_states, hidden, cell):
    x = x.unsqueeze(0)
    embedding = self.dropout(self.embedding(x)) # (1, N, embedding_size)
    sequence_length = encoder_states.shape[0]
    h_reshaped = hidden.repeat(sequence_length, 1, 1) # h_reshaped: (sequence_length, batch_size, hidden_size)
    energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim= 2))) # (seq_len, batch, h_size + h_size*2) -> (seq_len, batch, h_size*3)
    attention = self.softmax(energy) # (seg_length, N, 1)
    attention = attention.permute(1, 2, 0) # (N, 1, seg_length)
    encoder_states = encoder_states.permute(1, 0, 2) # (N, seg_length, hidden_size*2)
    context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2) # (N, 1, hidden_size*2) after bmm -> (1, N, hidden_size*2) after permute
    rnn_input = torch.cat((context_vector, embedding), dim= 2)
    outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell)) # outputs shape: (1, N, hidden_size) 
    predictions = self.fc(outputs) # (1, N, hidden_size) -> (1, N, output_size)
    predictions = predictions.squeeze(0)
    return predictions, hidden, cell

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, total_target_vocab_size):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.target_vocab_size = total_target_vocab_size

  def forward(self, source, target, teacher_force_ratio= 0.5):
    batch_size = source.shape[1] 
    target_len = target.shape[0] 
    target_vocab_size = self.target_vocab_size
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    encoder_states, hidden, cell = self.encoder(source)

    x = target[0]
    for t in range(1, target_len):
      output, hidden, cell = self.decoder(x, encoder_states, hidden, cell) 
      outputs[t] = output 
      best_guess = output.argmax(1)
      x = target[t] if random.random() < teacher_force_ratio else best_guess
    return outputs
