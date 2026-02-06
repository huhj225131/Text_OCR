import torch
import torch.nn as nn
import math
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model , max_h = 500, max_w= 1000):
        super().__init__()
        pe = torch.zero(max_h, max_w, d_model)
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * 
                            -(math.log(10000.0) / d_model_half))
        
        pos_h = torch.arange(0, max_h).unsqueeze(1)
        pos_w = torch.arange(0, max_w).unsqueeze(1)
        
        pe[:, :, 0:d_model_half:2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, 1:d_model_half:2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, d_model_half::2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(max_h, 1, 1)
        pe[:, :, d_model_half+1::2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(max_h, 1, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x, h, w):
        # x: [B, seq_len, d_model] where seq_len = h*w
        pe_flat = self.pe[:h, :w].reshape(-1, self.pe.size(-1))  # [h*w, d_model]
        return x + pe_flat.unsqueeze(0)
    
class PositionalEncoding1D(nn.Module):
    """1D positional encoding for decoder (sequence position)"""
    def __init__(self, d_model, max_len=200):  # INCREASED: IAM has longer words (up to 30+ chars)
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        # x: [B, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]