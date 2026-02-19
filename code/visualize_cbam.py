import torch
import cv2
import numpy as np
import os
from pytorch_i3d import InceptionI3d
from custom_models import I3DFeatureExtractor
from cbam_models import SignLanguageRecognitionModelCBAM

# --- CONFIGURATION ---
VIDEO_PATH = 'sample_sign.mp4'
MODEL_PATH = './checkpoints_cbam/best_cbam_model.pth'
OUTPUT_DIR = 'visualization_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_legend(img):
    h, w, _ = img.shape
    legend_w = 60
    gradient = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
    gradient = np.repeat(gradient, legend_w, axis=1)
    color_legend = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    cv2.putText(color_legend, "HIGH", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(color_legend, "LOW", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.hstack((img, color_legend))

def visualize_sequence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.resize(frame, (224, 224)))
    cap.release()
    
    video_np = np.array(frames).astype(np.float32)
    video_tensor = torch.from_numpy((video_np / 255.0) * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # 2. Load Model
    # Note: Ensure num_classes matches your training (100)
    i3d_base = InceptionI3d(400, in_channels=3) 
    feature_extractor = I3DFeatureExtractor(i3d_base)
    model = SignLanguageRecognitionModelCBAM(feature_extractor, num_classes=100)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.to(device).eval()

    # 3. Extract Attention
    with torch.no_grad():
        features = model.feature_extractor(video_tensor)
        refined = model.cbam.ChannelGate(features) * features
        spatial_masks = model.cbam.SpatialGate(refined) # Shape: [1, 1, T_feat, H, W]

    T_feat = spatial_masks.shape[2]
    T_orig = len(frames)
    # Ratio to map original frames to feature frames
    ratio = T_feat / T_orig 

    print(f"Original frames: {T_orig}, I3D Feature frames: {T_feat}")

# 4. Save 1 frame per second
    for sec in range(int(T_orig / fps)):
        orig_idx = int(sec * fps)
        feat_idx = min(int(orig_idx * ratio), T_feat - 1)

        # Get mask from CBAM Spatial Gate
        mask = spatial_masks[0, 0, feat_idx].cpu().numpy()
        
        # 1. Initial Normalization
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # 2. Clean 'Messiness' with Gaussian Blur 
        # Increase (7, 7) to (15, 15) if you want it even smoother
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        # 3. Re-normalize and apply Contrast Enhancement
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = np.power(mask, 1.5) 

        # 4. Create Heatmap and Resize
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # 5. PREVENT PATCHES: Convert everything to float [0.0, 1.0] for blending
        frame_float = frames[orig_idx].astype(np.float32) / 255.0
        heatmap_float = heatmap.astype(np.float32) / 255.0
        mask_3d = cv2.resize(mask, (224, 224))
        mask_3d = np.repeat(mask_3d[:, :, np.newaxis], 3, axis=2)

        # 6. Advanced Blending Logic:
        # We dim the background by the inverse of the mask
        # and "Screen" the heatmap over the top to avoid saturation patches
        background_dimmed = frame_float * (mask_3d * 0.7 + 0.2) 
        final_overlay = cv2.addWeighted(background_dimmed, 0.5, heatmap_float, 0.5, 0)

        # 7. Convert back to [0, 255] safely
        final_overlay = np.clip(final_overlay * 255, 0, 255).astype(np.uint8)

        final_img = add_legend(final_overlay)
        
        save_path = os.path.join(OUTPUT_DIR, f"sec_{sec}.jpg")
        cv2.imwrite(save_path, final_img)

    print(f"Enhanced visualization (no patches) saved to {OUTPUT_DIR}")
        
    """
    # 4. Save 1 frame per second
    for sec in range(int(T_orig / fps)):
        orig_idx = int(sec * fps)
        feat_idx = min(int(orig_idx * ratio), T_feat - 1)

        # Get mask and normalize
        mask = spatial_masks[0, 0, feat_idx].cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # Inside your visualization loop, after getting the mask:
        mask = spatial_masks[0, 0, feat_idx].cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # --- DARKER MASK ENHANCEMENTS ---
        # 1. Contrast Stretching: Square the mask to make low-attention areas darker
        mask = np.power(mask, 2) 
        
        # 2. Thresholding (Optional): Remove very weak attention signals
        mask[mask < 0.2] = 0 
        
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # 3. Adjust Alpha Blending: 
        # Lower the 'alpha' (0.3) for the original frame and 
        # increase 'beta' (0.7) for the heatmap to make it "pop"
        overlay = cv2.addWeighted(frames[orig_idx], 0.3, heatmap, 0.7, 0)
        
        # 4. Final Brightness Check: Darken the non-attention areas
        # We multiply the original frame by the mask so only attended areas stay bright
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask_3d = cv2.resize(mask_3d, (224, 224))
        darkened_bg = (frames[orig_idx] * (mask_3d + 0.1)).astype(np.uint8) 
        final_overlay = cv2.addWeighted(darkened_bg, 0.5, heatmap, 0.5, 0)

        final_img = add_legend(final_overlay)
        
        save_path = os.path.join(OUTPUT_DIR, f"sec_{sec}.jpg")
        cv2.imwrite(save_path, final_img)
    """
    
    print(f"Success! Check the '{OUTPUT_DIR}' folder for your images.")

if __name__ == '__main__':
    visualize_sequence()