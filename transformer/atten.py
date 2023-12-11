'''
Author: wenjun-VCC
Date: 2023-12-11 18:05:55
LastEditors: wenjun-VCC
LastEditTime: 2023-12-12 01:05:45
FilePath: atten.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2023 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048) -> None:
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def forward(self, x):
        return self.ff(x)

# We can use nn.MultiheadAttention()

class MHAttention(nn.Module):
    def __init__(self, d_model, n_heads=6, qk_bias=False, dropout=0.1) -> None:
        super(MHAttention, self).__init__()
        assert d_model // n_heads == 0, "d_model//n_heads != 0!"
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.Qw = nn.Linear(d_model, d_model, bias=qk_bias)
        self.Kw = nn.Linear(d_model, d_model, bias=qk_bias)
        self.Vw = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(self.d_k)
        
    @staticmethod
    def get_score(query, key):
        '''
            query   : [seq_len, batch_size, n_heads, d_k(d_model//n_heads)]
            key     : [seq_len, batch_size, n_heads, d_k(d_model//n_heads)]
        '''
        return torch.einsum('ibhd,jbhd->ijbh', query, key)
    
    def forward(self, query, key, value, mask=None):
        seq_len, batch_size, _ = query.shape
        query = self.Qw(query)
        key = self.Kw(key)
        value = self.Vw(value)
        score = self.get_score(query, key)
        score *= self.scale
        
        
        
class MaskedMHAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MaskedMHAttention ,self).__init__(*args, **kwargs)
        
        
if __name__ == '__main__':
    q = torch.randn(81, 32, 8, 128).to('cuda')
    k = torch.randn(36, 32, 8, 128).to('cuda')
    score = torch.einsum('ibhd,jbhd->ijbh', q, k)
    print(score.shape)