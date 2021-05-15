import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

os.getcwd()
path = '../Dataset/'
train_image_path = path + 'images/images_train/'
validation_image_path = path + 'images/images_validation/'
test_image_path = path + 'images/images_test/'
train_formula_path = path + 'formulas/train_formulas.txt'
validation_formula_path = path + 'formulas/validation_formulas.txt'

n_train, n_validation, n_test = 66509, 7391, 8250

images_test = []

for i in range(n_test):
  img = np.asarray(Image.open(test_image_path + '{}.png'.format(i)))
  images_test.append(img)

image = np.asarray(Image.open(test_image_path + '1.png'))

# print(image.shape)

images_test = np.array(images_test)
images_test = images_test.reshape(n_test, 1, 60, 400)

bos_token = '<BOS>'
eos_token = '<EOS>'
pad_token = '<PAD>'
unk_token = '<UNK>'


class MyReshape(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), input.size(1)*input.size(2), input.size(3)).transpose(1,2)

class AttnDecoder(nn.Module):
  def __init__(self, num_tokens, embedding_dim, hidden_dim, output_dim, attn_dim):
    super(AttnDecoder, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.attn_dim = attn_dim # dimension e Wh o Wv o Beta
    
    self.embedding = nn.Embedding(num_tokens, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim+output_dim, hidden_dim, 1, batch_first=True)
    self.Wh = nn.Linear(hidden_dim, attn_dim)
    self.Wv = nn.Linear(hidden_dim, attn_dim)
    self.tanh = nn.Tanh()
    self.beta = nn.Linear(attn_dim,1)
    self.Wc1 = nn.Linear(hidden_dim, output_dim)
    self.Wc2 = nn.Linear(hidden_dim, output_dim)
    self.o_tanh = nn.Tanh()
    self.Wout = nn.Linear(output_dim, num_tokens)
    self.nll = nn.NLLLoss(reduction='sum')
    
    
  def forward(self, last_y, last_o, last_h, last_c_memory, v, ):
    
    linear_trans = self.Wh(last_h).unsqueeze(1) + self.Wv(v) # Wh * h_t-1 + Wv * v_t
    e_i = self.beta(self.tanh(linear_trans)).squeeze(2) # beta_T * tanh(linear_trans)
    alpha_i = F.softmax(e_i, dim=1)
    c_i = (alpha_i.unsqueeze(2) * v).sum(dim=1)
    
    embedded = self.embedding(last_y)
    ltsm_input = torch.cat((embedded, last_o), dim=1)
    ltsm_input = ltsm_input.unsqueeze(1)
    h_t, (_, c_memory_t) = self.lstm(ltsm_input, (last_h.unsqueeze(0), last_c_memory.unsqueeze(0)))
    h_t = h_t.squeeze(1)
    c_memory_t = c_memory_t.squeeze(0)
    
    o_t = self.o_tanh(self.Wc1(h_t) + self.Wc2(c_i))
    prob = F.log_softmax(self.Wout(o_t), dim=1)
    
    return prob, o_t, h_t, c_memory_t,
    
  def train(self, v_tilda, sequence, max_len, teacher_forcing_ratio=1):
    batch_num = v_tilda.shape[0]
    h_0 = v_tilda[:,-1,:].contiguous()
    c_0 = torch.zeros_like(h_0).cuda()
    o_0 = torch.zeros(batch_num, self.output_dim).cuda()
    y_0 = sequence[:, 0].cuda()
    
    loss = 0
    for i in range(max_len-1):
#       y_0 = sequence[:, i].cuda()
      prob, o_0, h_0, c_0 = self.forward(y_0, o_0, h_0, c_0, v_tilda)
      loss += self.nll(prob, sequence[:, i+1].cuda())
      
      y_0 = sequence[:, i+1]
#       eps = random.random()
#       if eps > teacher_forcing_ratio:
#         y_0 = torch.distributions.Categorical(prob).sample()
#       else:
#         y_0 = sequence[:, i+1]
      
    return loss
  
      

# attn_decoder = AttnDecoder(num_tokens, embedding_dim=80, hidden_dim=512, output_dim=600, attn_dim=512).cuda()
    
encoder_hidden_size = 256
num_layers = 1

# lstm_encoder = nn.LSTM(feature_num, encoder_hidden_size, num_layers, bidirectional=True, batch_first=True).cuda()
cnn_encoder_loaded = nn.Sequential(
  nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
  nn.ReLU(),
  
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
  
  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
  nn.ReLU(),
  
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
  
  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256),
  nn.ReLU(),
  
  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
  nn.ReLU(),
  
  nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
  
  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(512),
  nn.ReLU(),
  
  nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)),
  
  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(512),
  nn.ReLU(),
  
  MyReshape()
    
)
import pickle
with open('token_idx.pkl', 'rb') as handle:
    token_idx = pickle.load(handle)

import numpy as np
all_tokens = np.load('all_tokens.npy')
all_tokens = all_tokens.tolist()
num_tokens = len(all_tokens)
cnn_encoder_loaded.load_state_dict(torch.load('cnn_train_0_epoch14.pth'))
cnn_encoder_loaded.cuda()
cnn_encoder_loaded.eval()


dummy_v = cnn_encoder_loaded(torch.Tensor(images_test[0:10]).cuda())
# print(dummy_v.shape)
_, seq_len, feature_num = dummy_v.shape

lstm_encoder_loaded = nn.LSTM(feature_num, encoder_hidden_size, num_layers, bidirectional=True, batch_first=True)
lstm_encoder_loaded.load_state_dict(torch.load('lstm_train_0_epoch14.pth'))
lstm_encoder_loaded.cuda()
lstm_encoder_loaded.eval()

attn_decoder_loaded = AttnDecoder(num_tokens, embedding_dim=80, hidden_dim=512, output_dim=600, attn_dim=512)
attn_decoder_loaded.load_state_dict(torch.load('decoder_train_0_epoch14.pth'))
attn_decoder_loaded.cuda()
# attn_decoder_loaded.eval()

def predict(image, cnn_encoder, lstm_encoder, attn_decoder, max_len=178):
  v = cnn_encoder(image.unsqueeze(0))
  v_tilda, _ = lstm_encoder(v)
  
  h_0 = v_tilda[0, -1, :].unsqueeze(0)
  c_0 = torch.zeros_like(h_0).cuda()
  o_0 = torch.zeros(attn_decoder.output_dim).unsqueeze(0).cuda()
  
  sequence = [token_idx[bos_token]]
  
  for i in range(max_len-1):
    y_0 = torch.LongTensor([sequence[-1]]).cuda()
    prob, o_0, h_0, c_0 = attn_decoder(y_0, o_0, h_0, c_0, v_tilda)
    prob = prob.squeeze(0)
    max_idx = torch.argmax(prob).item()
#     print(2.71 ** prob[max_idx].item())
    sequence.append(max_idx)
    if max_idx == token_idx[eos_token] or max_idx == token_idx[pad_token]:
      return sequence
    
  return sequence
      
  # a random image from test set is chosen ands the model is run on it
index = 74
test_image = torch.Tensor(images_test[index]).cuda()
print("test size",test_image.device)
test_image = image

test_image = torch.Tensor(image).cuda()
test_image -= 128
test_image /= 128
formula = predict(test_image, cnn_encoder_loaded, lstm_encoder_loaded, attn_decoder_loaded)
for i in range(len(formula)):
    formula[i] = all_tokens[formula[i]]
# print(formula)
sentence = ""
for i in range(1, len(formula)-1):
    sentence += formula[i]
    sentence += " "
# print("here")


# formula = outer_predict(test_image)
# print(formula)
# test_image = torch.Tensor(images_test[index].reshape(60,400))
# print("here",test_image.shape)


