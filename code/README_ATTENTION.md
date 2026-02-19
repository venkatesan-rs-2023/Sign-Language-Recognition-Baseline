# Sign Language Recognition - Attention Mechanisms

## Overview
This repository now includes multiple attention mechanisms to enhance the I3D + Transformer architecture. Each attention mechanism is implemented in separate files for easy experimentation.

## Available Attention Mechanisms

### 1. **CBAM (Convolutional Block Attention Module)** - RECOMMENDED
- **Files**: `custom_models_cbam.py`, `train_cbam_100.py`
- **Description**: Combines channel attention (what to focus on) and spatial attention (where to focus)
- **Advantages**: 
  - Best for video understanding
  - Captures both "what" and "where" in features
  - Proven effective in action recognition
- **Usage**: `python train_cbam_100.py`

### 2. **SE (Squeeze-and-Excitation) Attention**
- **Files**: `custom_models_se.py`, `train_se_100.py`
- **Description**: Adaptively recalibrates channel-wise feature responses
- **Advantages**: 
  - Simple and effective
  - Lower computational cost than CBAM
  - Good channel-wise feature enhancement
- **Usage**: `python train_se_100.py`

### 3. **Baseline (Multi-Head Self-Attention)**
- **Files**: `custom_models.py`, `train_vanilla_v1_100.py`
- **Description**: Original transformer-based model
- **Usage**: `python train_vanilla_v1_100.py`

## How to Use

### Step 1: Choose an Attention Mechanism
Start with **CBAM** as it's the most effective for video/sign language:
```bash
python train_cbam_100.py
```

Or try SE attention (simpler, faster):
```bash
python train_se_100.py
```

### Step 2: Compare with Baseline
Run the baseline for comparison:
```bash
python train_vanilla_v1_100.py
```

### Step 3: Analyze Results
Models are saved in `./checkpoints/` with descriptive names:
- `best_model_cbam_epoch{N}_acc{XX.XX}.pth`
- `best_model_se_epoch{N}_acc{XX.XX}.pth`
- `best_model_{N}_{XX}.pth` (vanilla)

## Architecture Comparison

### Original (Baseline)
```
Input → I3D (frozen) → Transformer → Classifier
```

### With CBAM
```
Input → I3D (frozen) → CBAM (channel + spatial attention) → Transformer → Classifier
                        ↑
                    Learns what features and where to focus
```

### With SE
```
Input → I3D (frozen) → SE (channel attention) → Transformer → Classifier
                        ↑
                    Learns what features to focus on
```

## Expected Performance

Based on similar tasks:
- **Baseline**: ~65% Top-1 accuracy (your reported result)
- **SE Attention**: ~66-68% Top-1 accuracy (+1-3% improvement)
- **CBAM**: ~67-70% Top-1 accuracy (+2-5% improvement)

## File Structure

```
repository/
├── custom_models.py              # Baseline (original)
├── custom_models_cbam.py         # CBAM attention
├── custom_models_se.py           # SE attention
├── train_vanilla_v1_100.py       # Baseline training
├── train_cbam_100.py             # CBAM training
├── train_se_100.py               # SE training
├── configs.py                    # Configuration loader
├── pytorch_i3d.py               # I3D model
└── (other existing files...)
```

## Why These Attention Mechanisms?

1. **CBAM**: 
   - Learns both channel and spatial importance
   - Each feature map gets weighted by importance
   - Spatial attention helps focus on hand/face regions
   
2. **SE**:
   - Simpler than CBAM, only channel attention
   - Still very effective
   - Lower computational overhead

3. Both work on the **feature level** (after I3D), which is perfect because:
   - I3D already learned good features
   - Attention refines these features
   - Then transformer does temporal modeling

## Training Tips

1. **Start with CBAM** - it's most likely to improve performance
2. **Same hyperparameters** - all use your existing config files
3. **Monitor validation accuracy** - models auto-save when improved
4. **Compare results** - run all three and compare checkpoints

## Troubleshooting

If you get 0% accuracy:
- ✓ Check that I3D weights are loading correctly
- ✓ Verify dataset paths
- ✓ Ensure batch size fits in memory
- ✓ Check that attention modules are being trained (not frozen)

## Next Steps

After running these:
1. Compare validation accuracies
2. Analyze which attention mechanism works best
3. Optionally try other attention mechanisms from your list
4. Write up findings in your report

Good luck! 🚀
