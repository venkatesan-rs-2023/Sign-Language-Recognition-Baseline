import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset
from torch.utils.data import WeightedRandomSampler

# Import the base feature extractor and the new CBAM model
from custom_models import I3DFeatureExtractor
from cbam_models import SignLanguageRecognitionModelCBAM

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy

def run(configs, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', save_model='', pretrained_i3d_weights=None):
    # Setup Data Augmentation
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

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
    model = nn.DataParallel(model)

    # Optimizer configuration: Include CBAM and Transformer parameters
    lr = 1e-4
    weight_decay = 1e-5  
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    
    # We ensure model.module is used to access parameters when using DataParallel
    optimizer = optim.Adam([
        {'params': model.module.cbam.parameters()},
        {'params': model.module.transformer.parameters()}
    ], lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 2 # Adjusted for typical training runs
    patience = 5
    checkpoint_dir = './checkpoints_cbam'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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

        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_cbam_model.pth"))
            print("Model Improved and Saved.\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

if __name__ == '__main__':
    mode = 'rgb'
    root = {'word': 'data/WLASL2000'}
    save_model = 'checkpoints_cbam/'
    train_split = 'preprocess/nslt_100.json'
    weights = 'i3d_pretrained_100.pt'
    config_file = 'configfiles/asl100.ini'

    configs = Config(config_file)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, pretrained_i3d_weights=weights)