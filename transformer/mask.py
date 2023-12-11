'''
Author: wenjun-VCC
Date: 2023-12-11 23:17:24
LastEditors: wenjun-VCC
LastEditTime: 2023-12-11 23:49:31
FilePath: mask.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2023 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn

def pad_mask(seqs, max_len):
    '''
    seqs: [bs, seq_len]
        bs      : batch size
        seq_len : 一句话单词个数
    max_len: 输入的最多单词数，用于求 self attention
    '''
    ...
    