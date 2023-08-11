import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable    
from einops import rearrange
    

class MaskedCasualAttention(nn.Module):
    def __init__(self, n_heads, h_dim, p_drop):
        super().__init__()

        self.n_heads = n_heads
        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(p=p_drop)
        self.proj_drop = nn.Dropout(p=p_drop)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # batch_size, seq_length, h_dim * n_heads
        N, D = self.n_heads, C // self.n_heads  # n_heads, attention_dim

        # rearrange q,k,v as (B,N,T,D)
        q = self.q_net(x)
        q = rearrange(q, 'b t (n d) -> b n t d', b=B, t=T, n=N, d=D)
        k = self.k_net(x)
        k = rearrange(k, 'b t (n d) -> b n t d', b=B, t=T, n=N, d=D)
        v = self.v_net(x)
        v = rearrange(v, 'b t (n d) -> b n t d', b=B, t=T, n=N, d=D)

        # weights (B,N,T,T)
        weights = (q @ k.transpose(2,3)).transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights 
        if mask is not None: 
            weights = weights.masked_fill(mask==0, float('-inf'))
        # normalize weights, all -inf -> after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B,N,T,D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B,N,T,D) -> (B,T,N*D)
        out = attention.transpose(1,2).contiguous().view(B,T,N*D)
        out = self.proj_drop(self.proj_net(out))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, h_dim, p_drop):
        super().__init__()
        self.n_heads = n_heads
        self.h_dim = h_dim

        self.attention = MaskedCasualAttention(n_heads=n_heads, h_dim=h_dim, p_drop=p_drop)
        self.feed_forward = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(p_drop)
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x, mask):
        x = x + self.attention(self.ln1(x), mask=mask)  
        x = x + self.feed_forward(self.ln2(x))
        return x


# self.t2vec_embedding(torch.tensor([[1.0],[3.0]]))
class Time2vecEncoding(nn.Module):
    def __init__(self, h_dim, scale=1):
        super(Time2vecEncoding, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(1, h_dim-1))
        self.b = nn.parameter.Parameter(torch.randn(h_dim-1))
        self.f = torch.sin
        self.scale = scale

    def t2v(self, time):
        time = time / self.scale        
        v1 = torch.matmul(time, self.w0) + self.b0
        v2 = self.f(torch.matmul(time, self.w) + self.b)
        v = torch.cat([v1,v2], 2)
        return v

    def forward(self, time):
        time_embedding = self.t2v(time)
        return time_embedding
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, h_dim, max_seq_len=200):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, h_dim)

        position = torch.arange(0,max_seq_len).unsqueeze(1)
        base = torch.ones(h_dim//2).fill_(10000)
        pow_term = torch.arange(0, h_dim, 2) / torch.tensor(h_dim,dtype=torch.float32)
        div_term = torch.pow(base,pow_term)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)

        # register_buffer: set as non-trainable layer but can check in state_dict
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        return x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)

