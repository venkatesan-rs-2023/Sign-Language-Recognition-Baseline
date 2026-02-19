"""
train_local_attention_100.py

Self-contained training script for Sign Language Recognition with Local Attention.
This file includes all necessary model definitions and can be run directly with your existing repository.

Usage:
    python train_local_attention_100.py --window_size 4
    python train_local_attention_100.py --window_size 8 --num_layers 6
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import videotransforms
import numpy as np
from collections import OrderedDict

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset
from torch.utils.data import WeightedRandomSampler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))


# ============================================================================
# LOCAL ATTENTION MODEL DEFINITIONS
# ============================================================================

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


def create_local_attention_mask(seq_len, window_size, device):
    """
    Creates a local attention mask where each position can only attend to 
    positions within a local window.
    
    Args:
        seq_len: Length of the sequence
        window_size: Size of the local attention window (positions on each side)
        device: Device to create the mask on
    
    Returns:
        mask: Boolean mask of shape [seq_len, seq_len] where True means masked
    """
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    positions_t = positions.transpose(0, 1)
    distance = torch.abs(positions - positions_t)
    mask = distance > window_size
    return mask


class LocalAttentionTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with local attention"""
    def __init__(self, d_model, nhead, window_size, dim_feedforward=2048, dropout=0.1):
        super(LocalAttentionTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
        self.window_size = window_size
        
    def forward(self, src, src_mask=None):
        if src_mask is None:
            seq_len = src.size(0)
            device = src.device
            src_mask = create_local_attention_mask(seq_len, self.window_size, device)
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class SignLanguageTransformer(nn.Module):
    """Transformer with local attention for sign language recognition"""
    def __init__(self, d_model=1024, nhead=8, num_layers=6, num_classes=2000, window_size=4, dropout=0.1):
        super(SignLanguageTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            LocalAttentionTransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                window_size=window_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)
        self.window_size = window_size

    def forward(self, x):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x


class SignLanguageRecognitionModel(nn.Module):
    """Complete model: I3D feature extractor + Local Attention Transformer"""
    def __init__(self, i3d_feature_extractor, num_classes, window_size=4, d_model=1024, nhead=8, num_layers=6):
        super(SignLanguageRecognitionModel, self).__init__()
        self.feature_extractor = i3d_feature_extractor
        self.transformer = SignLanguageTransformer(
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            num_classes=num_classes,
            window_size=window_size
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        batch_size, channels, frames, height, width = features.shape
        features = F.adaptive_avg_pool3d(features, (frames, 1, 1))
        features = features.view(batch_size, channels, frames)
        features = features.permute(2, 0, 1)
        logits = self.transformer(features)
        return logits


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy


def run(configs, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', 
        save_model='', pretrained_i3d_weights=None, window_size=4, num_layers=6):
    
    # Data transforms
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # Load datasets
    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)

    # Calculate class weights for balanced training
    all_labels = [label for _, label, _ in dataset]
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels, minlength=dataset.num_classes)
    epsilon = 1e-6
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    print(f"Class Weights: {class_weights_tensor}")

    # Create weighted sampler
    sample_weights = class_weights[all_labels]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor, 
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

    print(f"Sampler created with {len(sampler)} samples")

    # Create dataloaders
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=configs.batch_size, 
        sampler=sampler, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=configs.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=False
    )
    
    dataloaders = {'train': dataloader, 'test': val_dataloader}

    # Load pretrained I3D and create model
    i3d = InceptionI3d(100, in_channels=3)
    i3d.load_state_dict(torch.load(pretrained_i3d_weights, map_location=torch.device('cpu'), weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    num_classes = dataset.num_classes

    model = SignLanguageRecognitionModel(
        feature_extractor, 
        num_classes,
        window_size=window_size,
        d_model=1024,
        nhead=8,
        num_layers=num_layers
    )

    # Freeze I3D feature extractor
    for param in model.feature_extractor.feature_extractor.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Local attention window size: {window_size}")
    print(f"Number of transformer layers: {num_layers}\n")

    model = model.to(device)
    model = nn.DataParallel(model)

    # Setup training
    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.module.transformer.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 1
    patience = 5

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_accuracy = 0
    epochs_no_improve = 0
    early_stop = False

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        total_batches = 0

        for batch_idx, (inputs, labels, vids) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(outputs, labels)
            running_loss += loss.item()
            running_accuracy += accuracy
            total_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloaders['train'])}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

        epoch_loss = running_loss / total_batches
        epoch_accuracy = running_accuracy / total_batches

        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, "
              f"Training Accuracy: {epoch_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        val_total_batches = 0

        with torch.no_grad():
            for val_inputs, val_labels, val_vids in dataloaders['test']:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_accuracy = calculate_accuracy(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                val_running_accuracy += val_accuracy
                val_total_batches += 1

        val_epoch_loss = val_running_loss / val_total_batches
        val_epoch_accuracy = val_running_accuracy / val_total_batches

        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_epoch_loss:.4f}, "
              f"Validation Accuracy: {val_epoch_accuracy:.2f}%\n")

        scheduler.step()

        # Save best model
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            epochs_no_improve = 0

            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"best_model_local_attn_ws{window_size}_epoch{epoch}_acc{val_epoch_accuracy:.2f}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Validation accuracy improved to {val_epoch_accuracy:.2f}%. Model saved to {checkpoint_path}\n")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).\n")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                early_stop = True
                break

    if not early_stop:
        final_model_path = os.path.join(checkpoint_dir, f'final_model_local_attn_ws{window_size}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")

    print(f"\n{'='*80}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition with Local Attention')
    parser.add_argument('--window_size', type=int, default=4, 
                        help='Local attention window size (default: 4)')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers (default: 6)')
    parser.add_argument('--config', type=str, default='configfiles/asl100.ini',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    mode = 'rgb'
    root = {'word': 'data/WLASL2000'}
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_100.json'
    weights = 'i3d_pretrained_100.pt'
    
    configs = Config(args.config)
    
    print("="*80)
    print("Sign Language Recognition with Local Attention")
    print("="*80)
    print(f"Configuration: {configs}")
    print(f"Window Size: {args.window_size} (each position attends to ±{args.window_size} neighbors)")
    print(f"Number of Layers: {args.num_layers}")
    print("="*80 + "\n")
    
    run(
        configs=configs, 
        mode=mode, 
        root=root, 
        save_model=save_model, 
        train_split=train_split, 
        pretrained_i3d_weights=weights,
        window_size=args.window_size, 
        num_layers=args.num_layers
    )
