import os
import argparse
from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from torchvision import transforms
import videotransforms


class Compose:
    """Minimal replacement for torchvision.transforms.Compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))

from custom_models import SignLanguageRecognitionModel, I3DFeatureExtractor  # Ensure your model script is imported

def calculate_accuracy(outputs, labels):
    # Get the index of the max log-probability
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy


def run(configs, run_dir: Path, num_epochs: int, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', save_model='', pretrained_i3d_weights=None):


    train_transforms = Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=1,
                                             pin_memory=True) # changing num_workers from 3 to 1, because it caused RAM OOM issues.

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=1,
                                                 pin_memory=False)
    
    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}


    i3d = InceptionI3d(300, in_channels=3)
    i3d.load_state_dict(torch.load(pretrained_i3d_weights, map_location=torch.device("cpu"), weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    num_classes = dataset.num_classes

    model = SignLanguageRecognitionModel(feature_extractor, num_classes)

    for param in model.feature_extractor.feature_extractor.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    lr = 1e-4
    weight_decay = 1e-5  
    criterion = nn.CrossEntropyLoss()
    base_model = model.module if hasattr(model, "module") else model
    optimizer = optim.Adam(base_model.transformer.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    patience = 5  # For early stopping

    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0
    early_stop = False

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


        if (epoch % 10 == 0):
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_accuracy = 0.0
            val_total_batches = 0

            with torch.no_grad():
                for val_batch_idx, (val_inputs, val_labels, val_vids) in enumerate(dataloaders['test']):
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

            # Scheduler step
            scheduler.step()

            # Check for improvement
            if val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = val_epoch_accuracy
                epochs_no_improve = 0

                # Save the best model
                checkpoint_path = checkpoint_dir / f"best_model_{epoch}_{val_epoch_accuracy:.0f}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Validation accuracy improved. Model saved to {checkpoint_path}\n")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).\n")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    early_stop = True
                    break

        # Save a 'last' checkpoint at the end of the epoch
        last_ckpt_path = checkpoint_dir / "last.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "epochs_no_improve": epochs_no_improve,
        }, last_ckpt_path)


    # Save the final model (always)
    final_model_path = checkpoint_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--run_dir', type=str, default=None, help='Directory to store logs/checkpoints for this run')
    args = parser.parse_args()

    mode = 'rgb'
    root = {'word': 'data/WLASL2000'}
    train_split = 'preprocess/nslt_300.json'
    weights = 'pretrained/I3D/i3d_pretrained_300.pt'
    config_file = 'configfiles/asl300.ini'

    # Default run directory: runs/vanilla_300_<timestamp>
    if args.run_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = Path('runs') / f'vanilla_300_{timestamp}'
    else:
        run_dir = Path(args.run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)

    configs = Config(config_file)
    run(configs=configs, run_dir=run_dir, num_epochs=args.epochs, mode=mode, root=root,
        train_split=train_split, pretrained_i3d_weights=weights)
