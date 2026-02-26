import os
import argparse

from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Commenting because torchvision might cause problems in cluster.
# from torchvision import transforms 
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset

from torch.utils.data import WeightedRandomSampler


from custom_models import SignLanguageRecognitionModel, I3DFeatureExtractor  # Ensure your model script is imported

# NEW - class to replace the transforms.Compose we imported from torchvision - which might cause problems in cluster.
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # manually setting it might cause issues on cluster, so commenting it.
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count()))) # manually setting it might cause issues on cluster, so commenting it.
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES")) # sanity check, not needed tho.

def calculate_accuracy(outputs, labels):
    # Get the index of the max log-probability
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy


def run(configs, run_dir, num_epochs, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', save_model='', pretrained_i3d_weights=None):


    train_transforms = Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)

    all_labels = [label for _, label, _ in dataset]
    all_labels = np.array(all_labels)

    class_counts = np.bincount(all_labels, minlength=dataset.num_classes)
    epsilon = 1e-6
    class_weights = 1.0 / (class_counts + epsilon)

    class_weights = class_weights / class_weights.sum() * len(class_counts)

    class_weights_tensor = torch.FloatTensor(class_weights)

    print(f"Class Weights: {class_weights_tensor}")

    sample_weights = class_weights[all_labels]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(weights=sample_weights_tensor, 
                                    num_samples=len(sample_weights_tensor),
                                    replacement=True)

    print (f"Sampler created with {len(sampler)} samples")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, 
                                             sampler=sampler, 
                                             shuffle=False, 
                                             num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=2,
                                                 pin_memory=False)
    
    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}


    i3d = InceptionI3d(100, in_channels=3)
    i3d.load_state_dict(torch.load(pretrained_i3d_weights, map_location=torch.device('cpu'), weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    num_classes = dataset.num_classes

    model = SignLanguageRecognitionModel(feature_extractor, num_classes)

    for param in model.feature_extractor.feature_extractor.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model.to(device)
    # model = nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) # On a cluster where Slurm allocates one GPU, DataParallel works but adds overhead. So put inside if condition.

    lr = 1e-4
    weight_decay = 1e-5  
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    base_model = model.module if hasattr(model, "module") else model
    optimizer = optim.Adam(base_model.transformer.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # num_epochs = 1 # Since, now we take num_epochs as argument.
    patience = 5  # For early stopping
    epochs_no_improve = 0 # Previously, You only define epochs_no_improve = 0 inside the “improved” branch, but you increment it in the “not improved” branch. So now initialising it here.

    #checkpoint_dir = './checkpoints'
    #os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True) # 2 Lines of code for proper output directory.
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

        # Code for Last checkpoint - NEW
        last_ckpt_path = os.path.join(checkpoint_dir, "last.pth")
        torch.save(
            {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "epochs_no_improve": epochs_no_improve,
            },
            last_ckpt_path,
        )
        print(f"Saved last checkpoint to {last_ckpt_path}\n", flush=True)
        # Code end for Last checkpoint - NEW
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
                # checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{epoch}_{val_epoch_accuracy:.0f}.pth")
                checkpoint_path = checkpoint_dir / f"best_model_{epoch}_{val_epoch_accuracy:.2f}.pth" # Code for proper directory.
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Validation accuracy improved. Model saved to {checkpoint_path}\n", flush=True)
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).\n")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    early_stop = True
                    # Code for Last checkpoint - NEW
                    last_ckpt_path = os.path.join(checkpoint_dir, "last.pth")
                    torch.save(
                        {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_accuracy": best_val_accuracy,
                        "epochs_no_improve": epochs_no_improve,
                        },
                        last_ckpt_path,
                    )
                    print(f"Early stopping. Saved last checkpoint to {last_ckpt_path}\n", flush=True)
                    # Code End for Last checkpoint
                    break


    if not early_stop:
        # Save the final model
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}\n", flush=True)


if __name__ == '__main__':
    # NEW - Code for proper directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1) # NEW - taking no.of.epochs as arg.
    args = parser.parse_args()

    # If run_dir is not provided, create one under ./runs with a timestamp
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("runs") / f"vanilla_100_{ts}"
    else:
        run_dir = Path(args.run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)
    # NEW - Code ends for proper directory

    mode = 'rgb'
    root = {'word': 'data/WLASL2000'}
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_100.json'
    weights = 'pretrained/I3D/i3d_pretrained_100.pt'
    config_file = 'configfiles/asl100.ini'

    configs = Config(config_file)
    run(configs=configs, run_dir=run_dir, num_epochs=args.epochs, mode=mode, root=root, save_model=save_model, train_split=train_split, pretrained_i3d_weights=weights)



def get_run_dir(run_dir_arg: str | None, default_name: str) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("runs") / f"{default_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir









