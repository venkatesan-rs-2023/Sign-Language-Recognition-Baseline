import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torchvision import transforms  # removed: only Compose was used
import videotransforms

class Compose:
    """Minimal replacement for torchvision.Compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

import numpy as np
import datetime

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset
from torch.utils.data import WeightedRandomSampler

# Import the base feature extractor and the new CBAM model
from custom_models import I3DFeatureExtractor
from cbam_models import SignLanguageRecognitionModelCBAM

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy

def run(configs, run_dir: Path, num_epochs: int, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', save_model='', pretrained_i3d_weights=None):
    # Setup Data Augmentation
    train_transforms = Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = Compose([videotransforms.CenterCrop(224)])

    # Dataset Setup
    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    all_labels = np.array([label for _, label, _ in dataset])

    # Class Weighting and Sampler
    class_counts = np.bincount(all_labels, minlength=dataset.num_classes)
    epsilon = 1e-6
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    sample_weights = class_weights[all_labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), 
                                    num_samples=len(sample_weights),
                                    replacement=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, 
                                             sampler=sampler, shuffle=False, 
                                             num_workers=0, pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, 
                                                 shuffle=False, num_workers=2, pin_memory=False)
    
    dataloaders = {'train': dataloader, 'test': val_dataloader}

    # Initialize CBAM Model
    i3d = InceptionI3d(100, in_channels=3)
    i3d.load_state_dict(torch.load(pretrained_i3d_weights, map_location=torch.device('cpu'), weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    
    model = SignLanguageRecognitionModelCBAM(feature_extractor, dataset.num_classes)

    # Freeze I3D Feature Extractor (Same as baseline)
    for param in model.feature_extractor.feature_extractor.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer configuration: Include CBAM and Transformer parameters
    lr = 1e-4
    weight_decay = 1e-5  
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    
    # We ensure model.module is used to access parameters when using DataParallel
    base_model = model.module if hasattr(model, "module") else model
    optimizer = optim.Adam([
        {'params': base_model.cbam.parameters()},
        {'params': base_model.transformer.parameters()}
    ], lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # num_epochs is provided as an argument
    patience = 5
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_accuracy = 0
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_accuracy, total_batches = 0.0, 0.0, 0
        
        for batch_idx, (inputs, labels, vids) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

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
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloaders['train'])}], Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%")

        # Validation Logic
        model.eval()
        val_loss, val_acc, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for val_inputs, val_labels, _ in dataloaders['test']:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                val_acc += calculate_accuracy(val_outputs, val_labels)
                val_total += 1

        avg_val_acc = val_acc / val_total
        print(f"Epoch [{epoch+1}] Val Loss: {val_loss/val_total:.4f}, Val Acc: {avg_val_acc:.2f}%")

        scheduler.step()

        # Save a "last checkpoint" every epoch (useful for resuming/debugging)
        last_ckpt_path = checkpoint_dir / "last.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "epochs_no_improve": epochs_no_improve,
        }, last_ckpt_path)


        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            epochs_no_improve = 0
            checkpoint_path = checkpoint_dir / f"best_model_{epoch+1}_{avg_val_acc:.0f}.pth"
            torch.save(base_model.state_dict(), checkpoint_path)
            print(f"Validation accuracy improved. Model saved to {checkpoint_path}\n")
            # (message printed above with checkpoint path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--run_dir', type=str, default=None, help='Output directory for this run')
    args = parser.parse_args()

    mode = 'rgb'
    root = {'word': 'data/WLASL2000'}
    train_split = 'preprocess/nslt_100.json'
    weights = 'pretrained/I3D/i3d_pretrained_100.pt'
    config_file = 'configfiles/asl100.ini'

    configs = Config(config_file)

    if args.run_dir is None:
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = Path('runs') / f'cbam_100_{ts}'
    else:
        run_dir = Path(args.run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)

    run(configs=configs, run_dir=run_dir, num_epochs=args.epochs, mode=mode, root=root, train_split=train_split, pretrained_i3d_weights=weights)
