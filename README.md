# Multi-Modal Semantic Segmentation Pipeline (Sentinel-1 & Sentinel-2)

## Project Goal

To build a robust and configurable PyTorch Lightning framework for semantic segmentation on complex geospatial data (Sen1Floods11 dataset), enabling systematic comparison between classic CNN architectures and Geo-Foundation Models (GFM).

---

## Technical Contributions and Methodology


### 1. Robust Data & Infrastructure Engineering

- **MLOps Integration:** Established full experiment lifecycle management using **MLflow**, including logging of all command-line parameters (`argparse`) and tracking performance across multiple runs for complete reproducibility.
- **Resumption:** Implemented a secure training resumption mechanism using PyTorch Lightning's checkpointing, ensuring continuity of long training jobs by restoring model, optimizer, and training states from a `.ckpt` file.
- **Data Pipeline:** Addressed critical challenges related to multi-modal data homogeneity using `GenericMultiModalDataModule` (TerraTorch):
    - **Channel consistency:** Ensured strict channel alignment for the model input by configuring dynamic `output_bands` to match the exact channel count expected by different pre-trained backbones (e.g., 13 channels for Sentinel-2 L1C for TerraMind).
    - **Artifact handling:** Configured the DataModule to process artifacts and non-data values, aligning `no_label_replace=255` with the `ignore_index` of the `CrossEntropyLoss` and `MulticlassJaccardIndex` to prevent fatal GPU assertion errors during training.

### 2. Advanced Training Strategies

- **Universal Model Wrapper (`UniversalSegmentationTask`):** Developed a single `pl.LightningModule` capable of handling radically different input formats (single concatenated tensor for CNNs, dictionary of tensors for TerraMind) through conditional dispatching.
- **Dynamic Fine-Tuning Orchestration (Custom Callback):** Implemented a custom **`EncoderFineTuning` Callback** to control the transfer learning process:
    - **Logic:** The strategy dynamically monitors the `val_loss` and initiates the transition from the "Encoder Frozen" phase to the "Full Fine-Tuning" phase when performance stagnates over a defined `patience` period.
    - **Optimizer Control:** Achieved efficient layer-wise fine-tuning by modifying the `AdamW` optimizer at the point of transition (`on_epoch_start`), correctly clearing and adding parameter groups to apply a lower Learning Rate (LR) to the base encoder.

### 3. Comprehensive Performance Analysis

- **Benchmarking:** The framework supports the comparison of **U-Net** and **DeepLabV3+** architectures against the **TerraMind** GFM.
- **Metrics Accuracy:** Utilized `torchmetrics` with the `average='none'` setting to log the accurate **IoU per class** (specifically for the challenging 'Water' class, Index 1), avoiding inflated scores typical of global metrics.
- **Qualitative Insight (`PredictionLogger`):** Created a custom callback to generate and log visual prediction artifacts in **MLflow** at every validation epoch, enabling intuitive debugging and visual identification of model failure modes (e.g., cloud shadow confusion, boundary errors).

---

## 💡 Expertise Focus

This project serves as the foundation for specializing in Geo-Foundation Models, providing a platform ready to implement and rigorously test advanced techniques such as **Thinking in Modalities (TiM)** and other generative/agentic strategies for contextual segmentation.
