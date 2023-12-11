import torch
import torch.nn as nn
import torch.nn.functional as F
import parser
import math
args = parser.parse_args()

class Fusion(nn.Module):
    def __init__(self, d_model, num_heads, nlayers=6):
        super(Fusion, self).__init__()
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.seq_len = args.seq_len
    
    def forward(self, src):
        
        src = src.reshape(src.shape[0]//self.seq_len, self.seq_len, -1)
        
        out = self.transformer_encoder(src)
        out = out.mean(dim=1)
        return out
    
