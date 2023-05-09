import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#normalize
def data_normal(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data +=torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    dst = d_max - d_min
    norm_data = (orign_data-d_min).true_divide(dst)
    return norm_data

##############################position encoding###############################
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=81):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


##############self-attn########################################################    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.gamma=nn.Parameter(torch.tensor(100.0))

    def forward(self, q, k, v):

        attn1 = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn1, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn    
    
##############multihead########################################################    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self,d_model, nhead,dropout):
        super().__init__()

        self.n_head = nhead
        self.d_k = d_model // nhead
        self.d_v = d_model // nhead

        self.w_qs = nn.Linear(d_model, nhead * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, nhead * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, nhead * self.d_v, bias=False)
        self.fc = nn.Linear(nhead * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, src):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = src.size(0), src.size(1), src.size(1), src.size(1)

        residual = src

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(src).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(src).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(src).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)
        #print(attn.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #print(q.shape)
        #exit()
        # q = self.dropout(self.fc(q))
        # q += residual

        #q = self.layer_norm(q)

        return q, attn     
#################################conv-layer##################################                
class ConvLayer(nn.Module):

    def __init__(self, d_model):
        super(ConvLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=3 // 2)
        self.conv2 = nn.Conv1d(d_model, d_model, 5, padding=5 // 2)
        self.conv3 = nn.Conv1d(d_model, d_model, 7, padding=7 // 2)
        
    def forward(self, src):
        src = self.activation(self.norm1(self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)))
        src = self.activation(self.norm2(self.conv2(src.permute(0, 2, 1)).permute(0, 2, 1)))
        src = self.conv3(src.permute(0, 2, 1)).permute(0, 2, 1)
        return src


######################################SparseLayer####################################################
class SparseLayer(nn.Module):

    def __init__(self, d_model, nhead, d_feedfd, dropout=0.1):
        super(SparseLayer, self).__init__()
        self.sparse_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_feedfd)
        self.linear2 = nn.Linear(d_feedfd, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, src):
        res = src
        src, attn = self.sparse_attn(src)  # [bz,len,d_model]
        src = res + self.dropout(src)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attn


###########################################main model#################################################
class Spliformer_motif(nn.Module):

    def __init__(self, ntoken, nclass, d_model, nhead, d_feedfd, nlayers, dropout=0.1):
        super(Spliformer_motif, self).__init__()
        self.src_emb = nn.Embedding(ntoken, d_model)  # 4,32
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.convlayer = ConvLayer(d_model)
        self.sparse_attn_layer = []
        for i in range(nlayers):
            self.sparse_attn_layer.append(SparseLayer(d_model, nhead, d_feedfd, dropout))
        self.sparse_attn_layer = nn.ModuleList(self.sparse_attn_layer)

        self.decoder = nn.Linear(d_model, nclass)  # （32，3）

    def forward(self, src):
        B, L, _ = src.shape
        src = self.src_emb(src).reshape(B, L, -1)  # [bz,seqlen,d_model] #word embedding        
        src = self.convlayer(src)
        src = src[:,6:-6,:]
        src = self.pos_emb(src.permute(1,0,2)).permute(1,0,2) 
        for layer in self.sparse_attn_layer:

            src, attention_weights = layer(src)
        sum_attn=torch.sum(attention_weights,dim=1)          
        sum_attn1=np.array(sum_attn[0,:,:].cpu())#attention weight

        output = self.decoder(src)

        return sum_attn1
