import math
import torch
import torch.nn as nn
import torch.nn.functional as F


##############################position encoding###############################
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=20001):
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

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn    
    
##############multihead########################################################    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, neighbour,rate,d_model, nhead,dropout):
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





########################sliding window for local attn###########################    
def extract_seq_patches(x, kernel_size, rate):
    """x.shape=[bz,seq_len,seq_dim]"""
    seq_dim = x.shape[-1]
    seq_len = x.shape[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = F.pad(x, [0, 0, p_left, p_right])  # pad p_right*0 at the start and end of seq
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = torch.cat(xs, 2)
    return x.view(-1, seq_len, kernel_size, seq_dim)


#######################sparse-selfattn#########################################
class SparseSelfAttention(nn.Module):
    def __init__(self, neighbour, rate, d_model, nhead, dropout):  # dropout=dropout
        super(SparseSelfAttention, self).__init__()

        assert rate != 1, u'if rate=1, please use SelfAttention directly'
        self.rate = rate
        self.neighbors = rate - 1
        #self.neighbors = neighbour
        self.key = 0
        self.nhead = nhead
        self.d_k = d_model // nhead  # 32//2=16
        self.d_v = d_model // nhead
        self.d_model = d_model

        self.w_qs = nn.Linear(self.d_model, self.nhead * self.d_k, bias=False)  # 32>128
        self.w_ks = nn.Linear(self.d_model, self.nhead * self.d_k, bias=False)
        self.w_vs = nn.Linear(self.d_model, self.nhead * self.d_v, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.nhead * self.d_k, d_model, bias=False)

    def forward(self, src):
        seq_dim = src.shape[-1] 
        seq_len = src.shape[1]  
        pad_len = self.rate - seq_len % self.rate 
        src = F.pad(src, [0, 0, 0, pad_len])  
        new_seq_len = src.shape[1]  
        src = src.view(-1, new_seq_len, seq_dim)  

        # Pass through the pre-attention projection: b x lq x (n*dv)

        qw = self.w_qs(src)  # .view(sz_b, len_q, n_head, d_k) #checkpoint
        kw = self.w_ks(src)  # .view(sz_b, len_k, n_head, d_k) #checkpoint
        vw = self.w_vs(src)  # .view(sz_b, len_v, n_head, d_v) #checkpoint

        # extract local pattern
        kernel_size = 1 + 2 * self.neighbors  # 3
#         kwp = extract_seq_patches(kw, kernel_size, self.rate) #[bz,seq_len,kernel_size,dim]
#         vwp = extract_seq_patches(vw, kernel_size, self.rate)

        kwp = extract_seq_patches(kw, kernel_size, 1)  # [bz,seq_len,kernel_size,dim]
        vwp = extract_seq_patches(vw, kernel_size, 1)  # 

        # Separate different heads: b x lq x r x n x dv
        qw = qw.view(-1, new_seq_len // self.rate, self.rate, self.nhead, self.d_k)
        kw = kw.view(-1, new_seq_len // self.rate, self.rate, self.nhead, self.d_k)
        vw = vw.view(-1, new_seq_len // self.rate, self.rate, self.nhead, self.d_v)
        kwp = kwp.view(-1, new_seq_len // self.rate, self.rate, kernel_size, self.nhead, self.d_k)
        vwp = vwp.view(-1, new_seq_len // self.rate, self.rate, kernel_size, self.nhead, self.d_v)

        # Transpose for attention dot product: b x n x r x lq x dv
        qw = qw.permute(0, 3, 2, 1, 4)  # [bz,nhead,r,seqlen//r, dim]
        kw = kw.permute(0, 3, 2, 1, 4)
        vw = vw.permute(0, 3, 2, 1, 4)
        qwp = qw.unsqueeze(4)  # [bz,nhead,r,len//r,1,dim]
        kwp = kwp.permute(0, 4, 2, 1, 3, 5)  ## shape=[bz, nhead, r, seq_len // r, kernel_size, dim]
        vwp = vwp.permute(0, 4, 2, 1, 3, 5)

        # Atrous attention
        atrattn = torch.matmul(qw, kw.transpose(-2, -1)) / self.d_k ** 0.5  # [bz,head,r,len//r,len//r]

        lpattn = torch.matmul(qwp, kwp.transpose(-2, -1)) / self.d_k ** 0.5  # [bz,head,r,seq_len//r,1,kernel_size]
        lpattn = lpattn[..., 0, :]  # [bz,head,r,seq_len//r,kernel_size]

        # merge two atten
        allattn = torch.cat([atrattn, lpattn], -1)
        allattn = self.dropout(F.softmax(allattn, dim=-1))
        atrattn, lpattn = allattn[..., : atrattn.shape[-1]], allattn[..., atrattn.shape[-1]:]

        # multiply v
        atr_out = torch.matmul(atrattn, vw)  ##[bz,head,r,len//r,dim]

        lpattn = lpattn.unsqueeze(-2)  # [bz,nhead,r,seq_len//r,1,kernel_size]
        lp_out = torch.matmul(lpattn, vwp)
        lp_out = lp_out[..., 0, :]  # [bz,nhead,r,seq_len//r,dim]

        # atr + lp
        all_out = atr_out + lp_out
        all_out = all_out.permute(0, 3, 2, 1, 4)  # [bz,len//r,r,nhead,dim]
        all_out = all_out.contiguous().view(-1, new_seq_len, self.nhead * self.d_k)  # (bz,len,nhead*dim)
        all_out = all_out[:, : - pad_len]

        return all_out, allattn


    
    
class residualunit1(nn.Module):
    def __init__(self, d_model, con_size):
        super(residualunit1, self).__init__()

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(d_model, d_model, con_size[0], padding=con_size[0] // 2)
        self.conv2 = nn.Conv1d(d_model, d_model, con_size[0], padding=con_size[0] // 2)

    def forward(self, src):
        res = src
        src = self.activation(self.norm0(src))
        src = self.activation(self.norm1(self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)))
        src = self.conv2(src.permute(0, 2, 1)).permute(0, 2, 1)
        
        src = res + src
        return src

class residualunit2(nn.Module):
    def __init__(self, d_model, con_size):
        super(residualunit2, self).__init__()

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(d_model, d_model, con_size[0], dilation=4,padding=4*(con_size[0] // 2))
        self.conv2 = nn.Conv1d(d_model, d_model, con_size[0], dilation=4,padding=4*(con_size[0] // 2))

    def forward(self, src):
        res = src
        src = self.activation(self.norm0(src))

        src = self.activation(self.norm1(self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)))
        src = self.conv2(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = res + src
        return src
    
class residualunit3(nn.Module):
    def __init__(self, d_model, con_size):
        super(residualunit3, self).__init__()

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(d_model, d_model, con_size[1], dilation=10,padding=10*(con_size[1] // 2))
        self.conv2 = nn.Conv1d(d_model, d_model, con_size[1], dilation=10,padding=10*(con_size[1] // 2))

    def forward(self, src):
        res = src
        src = self.activation(self.norm0(src))

        src = self.activation(self.norm1(self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)))
        src = self.conv2(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = res + src
        return src
    
class residualunit4(nn.Module):
    def __init__(self, d_model, con_size):
        super(residualunit4, self).__init__()

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(d_model, d_model, con_size[2], dilation=25,padding=25*(con_size[2] // 2))
        self.conv2 = nn.Conv1d(d_model, d_model, con_size[2], dilation=25,padding=25*(con_size[2] // 2))

    def forward(self, src):
        res = src
        src = self.activation(self.norm0(src))

        src = self.activation(self.norm1(self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)))
        src = self.conv2(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = res + src
        return src
    
class ConvLayer4(nn.Module):

    def __init__(self, d_model, con_size):
        super(ConvLayer4, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv0 = nn.Conv1d(d_model, d_model, 1)
        self.convskip = nn.Conv1d(d_model, d_model, 1)
        self.resunit = residualunit4(d_model,con_size)
        self.conv1 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, src):
        res = self.convskip(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.conv0(src.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(4): 
            src = self.resunit(src)
            
        src = self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)        
        src = res + src
        src = self.norm(src)

        return src
    
    
class ConvLayer3(nn.Module):

    def __init__(self, d_model, con_size):
        super(ConvLayer3, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv0 = nn.Conv1d(d_model, d_model, 1)
        self.convskip = nn.Conv1d(d_model, d_model, 1)
        self.resunit = residualunit3(d_model,con_size)
        self.conv1 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, src):
        res = self.convskip(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.conv0(src.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(4): 
            src = self.resunit(src)
            
        src = self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)        
        src = res + src
        #src = self.norm(src)

        return src
    
class ConvLayer2(nn.Module):

    def __init__(self, d_model, con_size):
        super(ConvLayer2, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv0 = nn.Conv1d(d_model, d_model, 1)
        self.convskip = nn.Conv1d(d_model, d_model, 1)
        self.resunit = residualunit2(d_model,con_size)
        self.conv1 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, src):
        res = self.convskip(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.conv0(src.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(4):
            src = self.resunit(src)
            
        src = self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)        
        src = res + src
        return src    
#################################conv-layer##################################                
class ConvLayer(nn.Module):

    def __init__(self, d_model, con_size):
        super(ConvLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.conv0 = nn.Conv1d(d_model, d_model, 1)
        self.convskip = nn.Conv1d(d_model, d_model, 1)
        self.resunit = residualunit1(d_model,con_size)
        self.conv1 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, src):
        res = self.convskip(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.conv0(src.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(4): 
            src = self.resunit(src)
          
        src = self.conv1(src.permute(0, 2, 1)).permute(0, 2, 1)        
        src = res + src

        return src


######################################SparseLayer####################################################
class SparseLayer(nn.Module):

    def __init__(self, neighbour, rate, d_model, nhead, d_feedfd, dropout=0.1):
        super(SparseLayer, self).__init__()
        self.sparse_attn = SparseSelfAttention(neighbour, rate, d_model, nhead, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_feedfd)
        self.linear2 = nn.Linear(d_feedfd, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, src):
        res = src
        src, attn = self.sparse_attn(src)  
        src = res + self.dropout(src)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attn


###########################################main model#################################################
class Spliformer(nn.Module):

    def __init__(self, ntoken, nclass, d_model, nhead, d_feedfd, nlayers, rate, neighbour, con_size, dropout=0.1):
        super(Spliformer, self).__init__()
        self.src_emb = nn.Embedding(ntoken, d_model)  # 4,32
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.convlayer = ConvLayer(d_model, con_size)
        self.convlayer2 = ConvLayer2(d_model, con_size)
        self.convlayer3 = ConvLayer3(d_model, con_size)
        self.convlayer4 = ConvLayer4(d_model, con_size)
        self.sparse_attn_layer = []
        for i in range(nlayers):
            self.sparse_attn_layer.append(SparseLayer(neighbour, rate, d_model, nhead, d_feedfd, dropout))
        self.sparse_attn_layer = nn.ModuleList(self.sparse_attn_layer)



        self.decoder = nn.Linear(d_model, nclass)  # （32，3）

    def forward(self, src):
        B, L, _ = src.shape
        src = self.src_emb(src).reshape(B, L, -1)  # [bz,seqlen,d_model] #word embedding        
        src = self.convlayer(src)
        src = self.convlayer2(src)
        src = self.convlayer3(src)
        src = self.convlayer4(src)
        
        src = src[:,5000:-5000,:]

        src = self.pos_emb(src.permute(1,0,2)).permute(1,0,2)  # pos embedding #maybe can remove???
        for layer in self.sparse_attn_layer:

            src, attention_weights = layer(src)
            
        output = self.decoder(src)
        softmax1=nn.Softmax(dim=2)
        output = softmax1(output)
        return output
