# Image to Latex

### Installations

The following need to be installed as a part of the project:
1. Perl
2. Node.js 

### Preprocessing Scripts
[Im2latex](https://sujayr91.github.io/Im2Latex/), is a dataset that comprises of handwritten math equations and their corresponding latex code. Using the preprocessing scripts available with the dataset, the data is preprocessed and split into train, test and validation sets.

### Model Architecture
A CNN + LSTM based model is chosen to encode the images and then decode the latex code. The encoder is a CNN that takes the image as input and outputs a vector representation of the image. The vector representation is then passed through the LSTM layer. 

The LSTM layer takes the vector representation and the previous hidden state as input and outputs the next hidden state and the next cell state. The next hidden state and the next cell state are then passed through the decoder which is a LSTM layer followed by a linear layer. 

The linear layer takes the next hidden state and the next cell state as input and outputs the probabilities of the next token. The token with the highest probability is chosen as the next token. This process is repeated until the end of the sequence is reached.

Architecture Inspired by: https://cs231n.stanford.edu/reports/2017/pdfs/815.pdf

### Training

Using the architecture defined in train/model.py, the architecture of LSTM+CNN encoder combined with Attention Based Decoder is trained on the Im2latex dataset.

Custom data loaders are written to enable the training of the model. The data loaders are defined in train/model.py.

### App Deployment

Using streamlit, the model is deployed as a web app. The web app takes an image of a handwritten math equation as input and returns the predicted latex code for the equation.

To run the app, use the following command:
```
streamlit run app.py
```
### Dataset Link

Link: https://zenodo.org/records/56198#.YJjuCGZKgox
### Demo Link

Loom Video Demo: https://www.loom.com/share/ee3c3913815e4bc5b84bf3bec3bc8ea0
