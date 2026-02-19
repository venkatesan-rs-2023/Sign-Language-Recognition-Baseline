import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_models import SignLanguageTransformer, I3DFeatureExtractor

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAttention()
            
    def forward(self, x):
        x_out = self.ChannelGate(x) * x
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out) * x_out
        return x_out

class SignLanguageRecognitionModelCBAM(nn.Module):
    def __init__(self, i3d_feature_extractor, num_classes):
        super(SignLanguageRecognitionModelCBAM, self).__init__()
        self.feature_extractor = i3d_feature_extractor
        # I3D Mixed_5c output has 1024 channels
        self.cbam = CBAM(gate_channels=1024) 
        self.transformer = SignLanguageTransformer(d_model=1024, nhead=8, num_layers=6, num_classes=num_classes)

    def forward(self, x):
        # x shape: [batch_size, C, T, H, W]
        features = self.feature_extractor(x)
        
        # Apply CBAM Attention
        features = self.cbam(features)
        
        # Spatial pooling similar to baseline
        batch_size, channels, frames, height, width = features.shape
        features = F.adaptive_avg_pool3d(features, (frames, 1, 1))
        features = features.view(batch_size, channels, frames)
        features = features.permute(2, 0, 1) # [frames, batch_size, channels]
        
        logits = self.transformer(features)
        return logits