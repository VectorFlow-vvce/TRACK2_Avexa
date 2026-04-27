# TRACK2_Avexa

Off-road semantic segmentation for the Falcon by Duality AI hackathon.

## Overview

This project uses:
- `DINOv2 ViT-B/14` as the frozen visual backbone
- a lightweight `ConvNeXt-style segmentation head`
- a `10-class semantic segmentation` setup for off-road scene understanding

## Objective

Build a segmentation model for off-road scenes and improve robustness on visually similar terrain classes.

## Metrics

| Metric | Value |
|---|---:|
| Best Validation mIoU | `47.99%` |
| Final Evaluation Metric | `mAP50` |

Update these values with your final run if needed.

## Colab

| Resource | Link |
|---|---|
| Training Notebook / Colab | `PASTE_COLAB_LINK_HERE` |

## Project Structure

| Path | Description |
|---|---|
| `OPTIMIZED_TRAINING_COLAB.py` | Base Colab training script |
| `OPTIMIZED_TRAINING_COLAB_SAFE_V2.py` | Improved training script for higher mIoU |
| `OPTIMIZED_TRAINING_COLAB_SAFE_V3_FALLBACK.py` | Fallback stable training script |
| `evaluate_map50.py` | Local evaluation script for `mAP50` / IoU |
| `falcon_integration.py` | Inference wrapper for Falcon / deployment |
| `test_model.py` | Quick local model test |
| `test_on_unseen_images.py` | Evaluation on unseen images |
| `results/` | Saved plots, logs, and outputs |

## Dataset Layout

| Split | Folders |
|---|---|
| Train | `train/Color_Images`, `train/Segmentation` |
| Val | `val/Color_Images`, `val/Segmentation` |
| Test | `Offroad_Segmentation_testImages/Color_Images`, `Offroad_Segmentation_testImages/Segmentation` |

## Method

- `DINOv2 ViT-B/14` feature extractor
- `ConvNeXt-style` segmentation head
- `Focal + Dice loss`
- geometry-preserving preprocessing
- targeted class-confusion reduction for difficult terrain classes

## Bonus Challenge

### Problem
The model can confuse **Dry Grass** with **Flat Landscape**.

### Our Fix
We addressed this by:
- preserving scene geometry during preprocessing
- reducing overly aggressive color distortion
- adding a **targeted confusion-aware loss** between `Dry Grass` and `Landscape`

### Why it helps
This pushes the model to better separate visually similar ground classes instead of collapsing them into the same prediction.

## Run

| Task | Command / File |
|---|---|
| Train | `OPTIMIZED_TRAINING_COLAB_SAFE_V2.py` |
| Fallback Train | `OPTIMIZED_TRAINING_COLAB_SAFE_V3_FALLBACK.py` |
| Evaluate | `evaluate_map50.py` |
| Inference | `falcon_integration.py` |

## Notes

- Use `backbone_size="base"` for the new trained model.
- The training and inference class mapping are kept consistent.
- The repository includes both a stronger training version and a fallback stable version.
