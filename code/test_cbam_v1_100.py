import os
import torch
import torch.nn as nn
from torchvision import transforms
import videotransforms
import numpy as np
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset_all import NSLT as Dataset
from custom_models import I3DFeatureExtractor  
# Import the CBAM model from your new file
from cbam_models import SignLanguageRecognitionModelCBAM 
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

def compute_topk_tp_fp(outputs, labels, num_classes, topk=(1, 5, 10)):
    topk_metrics = {k: {'TP': defaultdict(int), 'FP': defaultdict(int)} for k in topk}
    outputs_np = outputs.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    for k in topk:
        # Get indices of top k results
        topk_indices = np.argsort(outputs_np, axis=1)[:, -k:]
        for i in range(len(labels_np)):
            label = labels_np[i]
            if label in topk_indices[i]:
                topk_metrics[k]['TP'][label] += 1
            else:
                topk_metrics[k]['FP'][label] += 1
    return topk_metrics

def run_test():
    # 1. Setup Environment
    mode = 'rgb'
    num_classes = 100
    # Directory where your CBAM model was saved
    checkpoint_path = './checkpoints_cbam/best_cbam_model.pth' 
    root = 'data/WLASL2000'
    test_split = 'preprocess/nslt_100.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Initialize Model
    i3d = InceptionI3d(400, in_channels=3)
    feature_extractor = I3DFeatureExtractor(i3d)
    model = SignLanguageRecognitionModelCBAM(feature_extractor, num_classes)
    
    # 3. Load Trained Weights
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Remove 'module.' prefix if it was saved using DataParallel
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

    # 4. Prepare Dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(test_split, 'test', root, mode, test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_outputs = []
    all_labels = []

    # 5. Testing Loop
    print("Starting Evaluation...")
    with torch.no_grad():
        for data in dataloader:
            inputs, labels, video_id = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            all_outputs.append(outputs)
            all_labels.append(labels)

    # 6. Metrics Calculation
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate Top-K Accuracy
    topk = (1, 5, 10)
    metrics = compute_topk_tp_fp(all_outputs, all_labels, num_classes, topk)
    
    for k in topk:
        total_tp = sum(metrics[k]['TP'].values())
        acc = total_tp / len(all_labels) * 100
        print(f"Top-{k} Accuracy: {acc:.2f}%")

    # Detailed Reports
    preds = torch.max(all_outputs, 1)[1].cpu().numpy()
    truth = all_labels.cpu().numpy()
    
    print("\nDetailed Classification Report:")
    print(classification_report(truth, preds, digits=4))

if __name__ == '__main__':
    run_test()