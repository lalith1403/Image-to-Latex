import os
import numpy as np
os.getcwd()
path = '../Dataset/'
train_image_path = path + 'images/images_train/'
validation_image_path = path + 'images/images_validation/'
test_image_path = path + 'images/images_test/'
train_formula_path = path + 'formulas/train_formulas.txt'
validation_formula_path = path + 'formulas/validation_formulas.txt'

train_formulas = open(train_formula_path, 'r').readlines()
validation_formulas = open(validation_formula_path, 'r').readlines()
bos_token = '<BOS>'
eos_token = '<EOS>'
pad_token = '<PAD>'
unk_token = '<UNK>'
n_train, n_validation, n_test = 66509, 7391, 8250

MAX_LENGTH = 0

for i in range(n_train):
  train_formulas[i] = [bos_token] + train_formulas[i].split() + [eos_token]
  MAX_LENGTH = max(MAX_LENGTH, len(train_formulas[i]))

for i in range(n_validation):
  validation_formulas[i] = [bos_token] + validation_formulas[i].split() + [eos_token]
  MAX_LENGTH = max(MAX_LENGTH, len(validation_formulas[i]))

train_formula_lengths = np.zeros((n_train,))
validation_formula_lengths = np.zeros((n_validation,))
# print(MAX_LENGTH)

for i in range(n_train):
  length = len(train_formulas[i])
  train_formula_lengths[i] = length
  train_formulas[i] += [pad_token for _ in range(MAX_LENGTH - length)]

for i in range(n_validation):
  length = len(validation_formulas[i])
  validation_formula_lengths[i] = length
  validation_formulas[i] += [pad_token for _ in range(MAX_LENGTH - length)]


# print(len(train_formulas))
# print(len(validation_formulas))

# print(train_formulas[0])

train_formula_lengths = train_formula_lengths.astype(int)
validation_formula_lengths = validation_formula_lengths.astype(int)

all_tokens = [unk_token]
for i in range(n_train):
  for j in range(len(train_formulas[i])):
    if train_formulas[i][j] not in all_tokens:
      all_tokens.append(train_formulas[i][j])
#   all_tokens += train_formulas[i]

print(type(all_tokens))
all_tokens_numpy = np.array(all_tokens)
np.save('all_tokens.npy',all_tokens_numpy)
# all_tokens = list(set(all_tokens))
num_tokens = len(all_tokens)
# print(num_tokens)
# print(all_tokens)
# print('\\over' in all_tokens)
# print('\\' in all_tokens)
token_idx = {}
for i in range(num_tokens):
  token_idx[all_tokens[i]] = i
# print(token_idx['<BOS>'])
# print(token_idx[bos_token])
# print(token_idx['{'])
print(len(token_idx))
import pickle

# pickle_file = open("token_idx.pkl","wb")
# pickle.dumps(token_idx,pickle_file)
# pickle_file.close()

with open('token_idx.pkl', 'wb') as handle:
    pickle.dump(token_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)