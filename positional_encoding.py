'''
Author: wenjun-VCC
Date: 2023-12-11 14:55:26
LastEditors: wenjun-VCC
LastEditTime: 2023-12-11 16:30:53
FilePath: positional_encoding.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2023 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# input [bs, seqs, d_model]

# Transformer
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
def pe_constant(seq_len, d_model):
    assert d_model % 2 == 0, "Wrong dimension!"
    seq_len_vec = torch.arange(seq_len, dtype=torch.float32)
    position_embedding = torch.zeros(seq_len, d_model, dtype=torch.float32)
    omega = torch.arange(d_model//2)
    omega = omega / (d_model / 2)
    omega = 1. / (10000**omega)
    out = seq_len_vec[:, None] @ omega[None, :]
    embed_sin = torch.sin(out)
    embed_cos = torch.cos(out)
    position_embedding[:,0::2] = embed_sin
    position_embedding[:,1::2] = embed_cos
    return position_embedding


# VIT
def learnable_pe(seq_len, d_model):
    
    pass

if __name__ == '__main__':
    seq_len = 64
    dim = 768
    pe = pe_constant(seq_len, dim)
    # print(pe)
    tensor_np = pe.numpy()
    # 使用 Matplotlib 创建热图
    plt.imshow(tensor_np, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 显示颜色条
    plt.show()
    
