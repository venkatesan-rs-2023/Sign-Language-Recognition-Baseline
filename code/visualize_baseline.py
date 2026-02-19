import torch
import cv2
import numpy as np
import os
from pytorch_i3d import InceptionI3d
from custom_models import I3DFeatureExtractor, SignLanguageRecognitionModel

# --- CONFIGURATION ---
VIDEO_PATH = 'sample_sign.mp4'
MODEL_PATH = './checkpoints/final_model.pth' 
OUTPUT_DIR = 'baseline_visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        # 1. MUST enable gradient for the input
        input_tensor.requires_grad = True
        
        # 2. Run forward pass MANUALLY without the no_grad decorator in custom_models.py
        # We access the internal modules directly to bypass the 'with torch.no_grad()'
        features = self.model.feature_extractor.feature_extractor(input_tensor)
        
        batch_size, channels, frames, height, width = features.shape
        pooled = torch.nn.functional.adaptive_avg_pool3d(features, (frames, 1, 1))
        pooled = pooled.view(batch_size, channels, frames).permute(2, 0, 1)
        
        output = self.model.transformer(pooled)
        
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        if self.gradients is None:
            raise ValueError("Gradients still not captured. Is Mixed_5c the right layer?")

        grads = self.gradients[0] 
        acts = self.activations[0]
        weights = torch.mean(grads, dim=(1, 2, 3)) 
        
        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()
        return cam

def visualize_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    i3d_base = InceptionI3d(400, in_channels=3)
    feature_extractor = I3DFeatureExtractor(i3d_base)
    model = SignLanguageRecognitionModel(feature_extractor, num_classes=100)
    
    # Loading weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.to(device).eval()

    # Identify the layer. In your I3D, it is usually inside the sequential extractor
    # Let's target Mixed_5c inside the OrderedDict
    target_layer = model.feature_extractor.feature_extractor._modules['Mixed_5c']
    
    gcam = GradCAM(model, target_layer)

    # Process Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.resize(frame, (224, 224)))
    cap.release()
    
    video_tensor = torch.from_numpy((np.array(frames).astype(np.float32) / 255.0) * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # Forward pass to get top prediction
    # We use the standard model forward here just for the index
    with torch.no_grad():
        output = model(video_tensor)
        pred_idx = torch.argmax(output).item()

    # Generate Heatmap (this function handles its own forward/backward)
    heatmap_3d = gcam.generate_heatmap(video_tensor, pred_idx)
    
    T_feat = heatmap_3d.shape[0]
    ratio = T_feat / len(frames)

    for sec in range(int(len(frames) / fps)):
        orig_idx = int(sec * fps)
        feat_idx = min(int(orig_idx * ratio), T_feat - 1)

        mask = heatmap_3d[feat_idx]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = np.power(mask, 2) 
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (224, 224))
        
        overlay = cv2.addWeighted(frames[orig_idx], 0.3, heatmap_color, 0.7, 0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"baseline_sec_{sec}.jpg"), overlay)

    print(f"Success! Baseline Grad-CAM results are in '{OUTPUT_DIR}'")

if __name__ == '__main__':
    visualize_baseline()