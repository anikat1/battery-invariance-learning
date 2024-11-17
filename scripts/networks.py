import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import copy
import pdb
from utils import *
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''
This script is partly copied and adapted from paper: 
Meta-learning the invariant representation for domain generalization
gihub: https://github.com/jiachenwestlake/MLIR
'''
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=False)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=False)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class ContextNet(nn.Module) : # we will use this for masked input of invariance model
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)

class Embedding(nn.Module): #this is for exogenous encoding of regressor
    def __init__(self, in_features, out_features, layer_size=[8,16,32]):
        super().__init__()
        self.in_feature = in_features
        
        dim = [in_features] + layer_size + [out_features]
        #self.activation = nn.LeakyReLU()
        self.activation = nn.ReLU()
        layers = list()
        self.num_dim=len(dim)
        for i in range(1,self.num_dim-1):
            layers.append(nn.Linear(dim[i-1],dim[i]))
            layers.append(self.activation)
        
        layers.append(nn.Linear(dim[i],dim[i+1]))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)

        return out

class MLP(nn.Module): #will use MLP for meta data and material encoding
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x) #use leaky relu
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x) #use leaky relu
        x = self.output(x)
        return x

class InvariantLSTM(nn.Module):
    def __init__(self, hparams, out_invar, mask_layer = [128, 64, 32]):
        super(InvariantLSTM, self).__init__()
        self.n_features = hparams['n_features'] #2 load, voltage
        self.seq_len = hparams['history_size']  #input seq length
        self.dim_hidden = hparams['dim_hidden']
        self.n_hidden = self.dim_hidden[0] # number of hidden states in a list form
        self.n_layers = hparams['num_layers'] # number of LSTM layers (stacked)
        self.out_seq = hparams['out_seq_len']
        self.out_features= hparams['out_features']
        self.out_invar = out_invar
        self.enc_dim = hparams['enc_dim']
        self.pos_enc = hparams['pos_enc'] 
        self.mask_dim = hparams['mask_dim'] # max_seqeunce_length after masking
        self.total_feat = self.n_features + 2*self.out_invar+ 2 
        # masking features (2), 2 invariance features (50 hidden dim each) + meta_data (1) + mat_enc(1)
        self.final_in_dim = self.n_hidden*self.seq_len+self.total_feat
        self.ffn1 = Embedding(self.enc_dim, 1)
        self.ffn2 = Embedding(self.pos_enc, 1)
        self.ffn3 = Embedding(self.n_features*self.mask_dim, self.n_features, layer_size=mask_layer)
        
        self.l_lstm = torch.nn.LSTM(input_size = self.n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        self.l_linear = torch.nn.Linear(self.final_in_dim, self.out_features*self.out_seq)
        self.activation = nn.Sigmoid()
        
    
    def init_hidden(self, x, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden, device = x.device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden, device = x.device)
        self.hidden = (hidden_state, cell_state)

        return self.hidden
    
    def forward(self, x, tgt, enc, pos_enc, masks, mat_inv, soc_inv):   
           
        batch_size, seq_len, _ = x.size() 
        
        hidden = self.init_hidden(x, batch_size) 
        lstm_out, hidden = self.l_lstm(x, hidden)
        x = lstm_out.contiguous().view(batch_size,-1)
        
        #all exog+invariance
        masks = masks.view(-1, self.n_features*self.mask_dim)
        #pdb.set_trace()
        enc_logits = self.ffn1(enc)
        mat_logits = self.ffn2(pos_enc)
        mask_logits = self.ffn3(masks)
        out_cat = torch.cat((x, enc_logits, mat_logits, mask_logits, mat_inv, soc_inv), 1)
        
        x_logits = self.l_linear(out_cat)
        x = self.activation(x_logits)
        
        out = x.view(-1,self.out_seq)
        
        return x_logits, out

class InvariantTransformer(nn.Module):
    def __init__(self, hparams, out_invar, mask_layer = [128, 64, 32]):
        super(InvariantTransformer, self).__init__()
        self.n_features = hparams['n_features'] #2 load, voltage
        self.seq_len = hparams['history_size']  #input seq length
        self.dim_hidden = hparams['dim_hidden']
        self.n_hidden = self.dim_hidden[0] # number of hidden states in a list form
        self.n_layers = hparams['num_layers'] # number of transformer layers
        self.out_seq = hparams['out_seq_len']
        self.out_features= hparams['out_features']
        self.out_invar = out_invar
        self.input_dim = hparams['input_dim']
        self.enc_dim = hparams['enc_dim']
        self.pos_enc = hparams['pos_enc'] 
        self.mask_dim = hparams['mask_dim'] # max_seqeunce_length after masking
        self.mask_features = 8
        self.total_feat = self.mask_features + 2*self.out_invar + 2
        self.n_heads = hparams['n_heads']
        
        # masking features (2), 2 invariance features (50 hidden dim each) + meta_data (1) + mat_enc(1)
        self.final_in_dim = self.input_dim+self.total_feat
        self.input_emb = nn.Linear(self.n_features*self.seq_len, self.input_dim)
        self.target_emb = nn.Linear(self.out_seq, self.input_dim)
        self.ffn1 = Embedding(self.enc_dim, 1)
        self.ffn2 = Embedding(self.pos_enc, 1)
        self.ffn3 = Embedding(self.n_features*self.mask_dim, self.mask_features, layer_size=mask_layer)
        
        self.transformer = nn.Transformer(d_model = self.input_dim, nhead = self.n_heads, 
                                          num_encoder_layers = self.n_layers, 
                                          num_decoder_layers = self.n_layers, 
                                          dim_feedforward = self.n_hidden,
                                          batch_first = True)
        
        
        self.l_linear = torch.nn.Linear(self.final_in_dim, self.out_features*self.out_seq)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x, tgt, enc, pos_enc, masks, mat_inv, soc_inv):   
           
        batch_size, seq_len, _ = x.size() 
        in_x = self.input_emb(x.view(batch_size, -1))
        tgt_x = self.target_emb(tgt)
        x = self.transformer(in_x, tgt_x)
        #print(in_x.size(), tgt_x.size(), x.size())
        #all exog+invariance
        masks = masks.view(-1, self.n_features*self.mask_dim)
        
        enc_logits = self.ffn1(enc)
        mat_logits = self.ffn2(pos_enc)
        mask_logits = self.ffn3(masks)
        out_cat = torch.cat((in_x, enc_logits, mat_logits, mask_logits, mat_inv, soc_inv), 1)
        
        x_logits = self.l_linear(out_cat)
        x = self.activation(x_logits)
        
        out = x.view(-1,self.out_seq)
        
        return x_logits, out

class InvariantModel(nn.Module):
    def __init__(self, hparams):
        super(InvariantModel, self).__init__()
        '''
        for this we need to process n inputs in different ways:
        1. Input history seq: L, V
        2. Material enc
        3. meta_data
        4. masked seq, L, V
        #n_inputs is a array of input shapes for all input features 
        n_outputs is array of output shapes to receive 
        '''
        n_inputs = hparams['n_inputs']
        n_outputs = hparams['n_outputs'] 
        self.input_enc = MLP(n_inputs[0], n_outputs[0], hparams)
        self.pos_embedding = MLP(n_inputs[1], n_outputs[1], hparams)
        self.meta_embedding = MLP(n_inputs[2], n_outputs[2], hparams) 
        self.masked_model = ContextNet([n_inputs[3]]) #
        self.in_shape = torch.sum(torch.tensor(n_outputs[:4])) #ignore the last output which is the output shape of last linear layer
        self.out_shape = n_outputs[4] #final output
        self.linear = torch.nn.Linear(self.in_shape, self.out_shape)
    
    def forward(self, x, mat, exog, mask):
        batch_size = x.size()[0]
        x = x.reshape(batch_size, -1)
        mask = mask.permute((0,2,1))
        mask= mask[:,:,:, None]
        
        out1 = self.input_enc(x)
        #mat.requires_grad=True
        out2 = self.pos_embedding(mat)
        #exog.requires_grad = True
        out3 = self.meta_embedding(exog)
        out4 = self.masked_model(mask)
        out4 = out4.reshape(batch_size, -1) #won't work for concatenation if batch size=1

        out_x = torch.cat((out1, out2, out3, out4), dim =1)
        
        inv_x = self.linear(out_x)
         
        #material
        ''' 
        response = {"disc_material_enc":mat.cpu().data.numpy().ravel().tolist(),
                 "cont_material_enc":out2.item(),
                 "cont_enc_sens":jac_calc(inv_x[0],out2),
                 "disc_enc_sens": jac_calc2(inv_x[0], mat)}
         
        #soc
        response = {"disc_soc_enc": exog[0,1].item(),
                 "cont_soc_enc":out3.item(),
                 "cont_enc_sens":jac_calc(inv_x[0],out3),
                 "disc_enc_sens": jac_calc3(inv_x[0], exog)} 
        '''
        return inv_x#, response

def AdvClassifier(in_features, out_features, in_nonlinear=False):
    if in_nonlinear:
        return torch.nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_features, in_features // 2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Linear(in_features // 2, out_features))
        )
    else:
        return torch.nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU() 
            )
