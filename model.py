import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # load in resnet pretrained model
        resnet = models.resnet50(pretrained=True)
        # freeze the features from being trained 
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        print('modules ='+str(modules))
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size ,batch_size=32, num_layers=2,drop_out = 0.15):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers= self.num_layers,
                            batch_first = True, dropout=self.drop_out)
        
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        
        # initialize the hidden state 
        self.hidden = self.init_hidden(self.batch_size)
        
    def init_hidden(self, batch_size):
        # The axes dimensions are (num_layers, batch_size, hidden_size). batch_size explicitly made = 1
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
    
        
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embed = self.embedding_layer(captions)
        #reshape features
        #features :(batch_size, 1, embed)
        features = features.view(self.batch_size, 1, -1)
        #combine features and embed
        #embed is the input tensor
        #embed shape : (batch_size, seq_len, embed)
        embed = torch.cat((features, embed), dim =1)
        # lstm_outputs shape : (batch_size, seq_len, hidden_size)
        lstm_outputs, self.hidden = self.lstm(embed, self.hidden)
        #lstm_outputs = self.dropout(lstm_outputs)
        lstm_outputs_shape = lstm_outputs.shape
        lstm_outputs_shape = list(lstm_outputs_shape)
        lstm_outputs = lstm_outputs.reshape(lstm_outputs.size()[0]*lstm_outputs.size()[1], -1)
        #get the probability for the next word
        #vocab outputs shape ; (batch_size*seq, vocab_size)
        vocab_outputs = self.linear(lstm_outputs)
        # new vocab outputs shape :(batch_size, seq, vocab_size)
        vocab_outputs = vocab_outputs.reshape(lstm_outputs_shape[0], lstm_outputs_shape[1], -1)
        
        return vocab_outputs
    
    
    
    def sample(self, inputs, states=None, max_len=20):
        output = []
        batch_size = inputs.shape[0]
        hidden = (torch.randn(1, 1, 512).to(inputs.device),
              torch.randn(1, 1, 512).to(inputs.device))
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_pred_index = torch.max(outputs, dim = 1)
            output.append(max_pred_index.cpu().numpy()[0].item())
            if (max_pred_index == 1):
                break
            inputs = self.embedding_layer(max_pred_index)
            inputs = inputs.unsqueeze(1)
        return output