# Custom Models with SE (Squeeze-and-Excitation) Attention
# SE attention adaptively recalibrates channel-wise feature responses

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class I3DFeatureExtractor(nn.Module):
    """Wrapper for I3D to extract features without final classification layers"""
    def __init__(self, i3d_model):
        super(I3DFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            OrderedDict([
                (k, i3d_model._modules[k]) for k in list(i3d_model.end_points.keys())
            ])
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x


class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention Module"""
    def __init__(self, channels, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze: global spatial pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: channel-wise attention
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Scale: apply attention weights
        return x * y.expand_as(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    """Transformer for temporal modeling"""
    def __init__(self, d_model=1024, nhead=8, num_layers=6, num_classes=2000):
        super(SignLanguageTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Mean pooling over the sequence length
        x = self.classifier(x)
        return x


class SignLanguageRecognitionModel(nn.Module):
    """Complete model: I3D + SE Attention + Transformer"""
    def __init__(self, i3d_feature_extractor, num_classes):
        super(SignLanguageRecognitionModel, self).__init__()
        self.feature_extractor = i3d_feature_extractor
        
        # SE attention module
        # I3D Mixed_5c outputs 1024 channels
        self.se_attention = SEAttention(channels=1024, reduction_ratio=16)
        
        # Transformer for temporal modeling
        self.transformer = SignLanguageTransformer(
            d_model=1024, 
            nhead=8, 
            num_layers=6, 
            num_classes=num_classes
        )
        
        print("Initialized model with SE (Squeeze-and-Excitation) Attention")

    def forward(self, x):
        # x shape: [batch_size, C, T, H, W]
        features = self.feature_extractor(x)
        # features shape: [batch_size, 1024, frames, height, width]
        
        # Apply SE attention to enhance channel features
        features = self.se_attention(features)
        
        batch_size, channels, frames, height, width = features.shape
        
        # Spatial pooling
        features = F.adaptive_avg_pool3d(features, (frames, 1, 1))
        features = features.view(batch_size, channels, frames)  # [batch_size, channels, frames]
        features = features.permute(2, 0, 1)  # [frames, batch_size, channels]
        
        # Temporal modeling and classification
        logits = self.transformer(features)
        return logits
