import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --------------------
# Step 1: Data Preprocessing
# --------------------

# Define paths and load data
path = '../Dataset/'
train_image_path = path + 'images/images_train/'
validation_image_path = path + 'images/images_validation/'
test_image_path = path + 'images/images_test/'
train_formula_path = path + 'formulas/train_formulas.txt'
validation_formula_path = path + 'formulas/validation_formulas.txt'

n_train, n_validation, n_test = 66509, 7391, 8250
images_test = [np.asarray(Image.open(test_image_path + f'{i}.png')) for i in range(n_test)]
images_test = np.array(images_test).reshape(n_test, 1, 60, 400)

bos_token = '<BOS>'
eos_token = '<EOS>'
pad_token = '<PAD>'
unk_token = '<UNK>'
# --------------------
# Step 2: Model Definition
# --------------------
class CustomReshape(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, input.size(3)).transpose(1, 2)


encoder_hidden_size = 256
num_layers = 1
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
    
    CustomReshape()
    
)

class AttnDecoder(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, output_dim, attn_dim):
        super(AttnDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attn_dim = attn_dim
        
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
        
    def forward(self, last_y, last_o, last_h, last_c_memory, v):
        linear_trans = self.Wh(last_h).unsqueeze(1) + self.Wv(v)
        e_i = self.beta(self.tanh(linear_trans)).squeeze(2)
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
        
        return prob, o_t, h_t, c_memory_t
        
    def train(self, v_tilda, sequence, max_len, teacher_forcing_ratio=1):
        batch_num = v_tilda.shape[0]
        h_0 = v_tilda[:,-1,:].contiguous()
        c_0 = torch.zeros_like(h_0).cuda()
        o_0 = torch.zeros(batch_num, self.output_dim).cuda()
        y_0 = sequence[:, 0].cuda()
        
        loss = 0
        for i in range(max_len-1):
            prob, o_0, h_0, c_0 = self.forward(y_0, o_0, h_0, c_0, v_tilda)
            loss += self.nll(prob, sequence[:, i+1].cuda())
            
            y_0 = sequence[:, i+1]
            eps = random.random()
            if eps > teacher_forcing_ratio:
                y_0 = torch.distributions.Categorical(prob).sample()
            else:
                y_0 = sequence[:, i+1]
        
        return loss

with open('token_idx.pkl', 'rb') as handle:
    token_idx = pickle.load(handle)

all_tokens = np.load('all_tokens.npy')
all_tokens = all_tokens.tolist()
num_tokens = len(all_tokens)
cnn_encoder_loaded.load_state_dict(torch.load('cnn_train_0_epoch14.pth'))
cnn_encoder_loaded.cuda()
cnn_encoder_loaded.eval()

dummy_v = cnn_encoder_loaded(torch.Tensor(images_test[0:10]).cuda())

_, seq_len, feature_num = dummy_v.shape

lstm_encoder_loaded = nn.LSTM(feature_num, encoder_hidden_size, num_layers, bidirectional=True, batch_first=True)
lstm_encoder_loaded.load_state_dict(torch.load('lstm_train_0_epoch14.pth'))
lstm_encoder_loaded.cuda()
lstm_encoder_loaded.eval()

attn_decoder_loaded = AttnDecoder(num_tokens, embedding_dim=80, hidden_dim=512, output_dim=600, attn_dim=512)
attn_decoder_loaded.load_state_dict(torch.load('decoder_train_0_epoch14.pth'))
attn_decoder_loaded.cuda()

# --------------------
# Step 3: Training
# --------------------
class ImageFormulaDataset(Dataset):
    def __init__(self, image_paths, formula_paths, transform=None):
        self.image_paths = image_paths
        self.formula_paths = formula_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        with open(self.formula_paths[idx], 'r') as file:
            formula = file.read().strip()
        if self.transform:
            image = self.transform(image)
        return image, formula

def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Assuming the paths are defined as shown at the top of the file
    train_image_paths = [train_image_path + f'{i}.png' for i in range(n_train)]
    validation_image_paths = [validation_image_path + f'{i}.png' for i in range(n_validation)]
    test_image_paths = [test_image_path + f'{i}.png' for i in range(n_test)]

    # Assuming formulas are stored in a way that they can be mapped directly from the image paths
    train_formula_paths = [train_formula_path for _ in range(n_train)]
    validation_formula_paths = [validation_formula_path for _ in range(n_validation)]
    test_formula_paths = [test_formula_path for _ in range(n_test)]

    train_dataset = ImageFormulaDataset(train_image_paths, train_formula_paths, transform)
    validation_dataset = ImageFormulaDataset(validation_image_paths, validation_formula_paths, transform)
    test_dataset = ImageFormulaDataset(test_image_paths, test_formula_paths, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader

def train_model():
    train_loader, validation_loader, _ = get_data_loaders()

    parameters = list(cnn_encoder_loaded.parameters()) + list(lstm_encoder_loaded.parameters()) + list(attn_decoder_loaded.parameters())
    optimizer = optim.Adam(parameters, lr=0.001)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        cnn_encoder_loaded.train()
        lstm_encoder_loaded.train()
        attn_decoder_loaded.train()

        running_loss = 0.0
        for images, formulas in train_loader:
            images = images.cuda()
            formulas = formulas.cuda()

            optimizer.zero_grad()

            v = cnn_encoder_loaded(images)
            v_tilda, _ = lstm_encoder_loaded(v)
            loss = attn_decoder_loaded.train(v_tilda, formulas, max_len=formulas.size(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

        with torch.no_grad():
            cnn_encoder_loaded.eval()
            lstm_encoder_loaded.eval()
            attn_decoder_loaded.eval()

            validation_loss = 0.0
            for images, formulas in validation_loader:
                images = images.cuda()
                formulas = formulas.cuda()

                v = cnn_encoder_loaded(images)
                v_tilda, _ = lstm_encoder_loaded(v)
                loss = attn_decoder_loaded.train(v_tilda, formulas, max_len=formulas.size(1))

                validation_loss += loss.item()

            print(f'Validation Loss: {validation_loss/len(validation_loader)}')

# --------------------
# Step 4: Prediction
# --------------------
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

        sequence.append(max_idx)
        if max_idx == token_idx[eos_token] or max_idx == token_idx[pad_token]:
            return sequence
        
    return sequence
        
# Example usage
index = 74
test_image = torch.Tensor(images_test[index]).cuda()
formula = predict(test_image, cnn_encoder_loaded, lstm_encoder_loaded, attn_decoder_loaded)
# Convert formula indices to tokens
for i in range(len(formula)):
    formula[i] = all_tokens[formula[i]]

sentence = ""
for i in range(1, len(formula)-1):
    sentence += formula[i]
    sentence += " "

# --------------------
# Step 5: Saving Models
# --------------------
def save_models():
    torch.save(cnn_encoder_loaded.state_dict(), 'deploy/cnn_train.pth')
    torch.save(lstm_encoder_loaded.state_dict(), 'deploy/lstm_train.pth')
    torch.save(attn_decoder_loaded.state_dict(), 'deploy/decoder_train.pth')

# Call save_models at the end of training
save_models()
